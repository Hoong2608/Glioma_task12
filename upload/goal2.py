"""任务二在管线里的落点：拼接影像检测 + 重复影像检测 + 下游任务闸门。

三个部分，全部在 task1 内实现，不改动管线代码：

1. ``SpecialImageTask``（绑定到 ``goal2_stitched``，**排在任务链最前面**）
   一次算完两件事：

   * 拼接：按「当前层 vs 上下层均值」逐层比较（见 ``stitched.py``）；
   * 重复：当前检查各序列的中间层指纹是否命中的其它检查（见 ``middle_slice.py``），
     索引在处理的第一个检查时**先扫全库**建立。

   结果写进 ``context.diagnostics["goal2"]``，并在命中时写 ``context.diagnostics["goal2_gate"]``。

2. ``GatedStudyTask``：包装下游任务，看到闸门标记就**直接返回中性结果、不跑模型**，
   即赛方要求的「该病人不向下进入后续任务」。

3. ``DuplicatePairRecorder``（DatasetTask）：把各检查的命中关系汇总成
   ``duplicate_pairs.jsonl``（每例最多 200 个候选，与赛方稀疏性约束一致）。
"""
from __future__ import annotations

import logging
import os
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ._bootstrap import ensure_pipeline_root

ensure_pipeline_root()

from tasks.base import DatasetTask, StudyTask  # noqa: E402
from tasks.results import (  # noqa: E402
    DuplicatePair,
    DuplicateResult,
    Goal1Result,
    StitchedResult,
)

from .middle_slice import DuplicateIndex, ScanReport, dataset_root_of, signature_from_array
from .stitched import (  # noqa: E402
    DEFAULT_BAND,
    DEFAULT_SCALE,
    DEFAULT_THRESHOLD,
    METRICS,
    STATISTICS,
    probability_from_score,
    score_study,
)

logger = logging.getLogger(__name__)

GATE_KEY = "goal2_gate"
DIAGNOSTIC_KEY = "goal2"
SKIP_KEY = "goal2_skipped_tasks"
DEFAULT_GATED_FIELDS = ("goal3", "goal4", "goal5")


def _env_str(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Task2Config:
    """任务二的全部开关与阈值（``TASK2_*`` 环境变量）。"""

    stitched_enabled: bool = True
    stitched_statistic: str = "max"
    stitched_metric: str = "curvature"
    stitched_band: float = DEFAULT_BAND
    stitched_threshold: float = DEFAULT_THRESHOLD
    stitched_scale: float = DEFAULT_SCALE
    stitched_gate: bool = True

    duplicate_enabled: bool = True
    duplicate_scan: bool = True
    duplicate_scan_max_volumes: int = 0
    duplicate_gate: bool = True

    gated_fields: tuple[str, ...] = DEFAULT_GATED_FIELDS
    max_pairs_per_study: int = 200

    def __post_init__(self) -> None:
        if self.stitched_statistic not in STATISTICS:
            raise ValueError(f"stitched_statistic must be one of {STATISTICS}")
        if self.stitched_metric not in METRICS:
            raise ValueError(f"stitched_metric must be one of {METRICS}")
        if not 0.0 <= self.stitched_band < 0.5:
            raise ValueError("stitched_band must be within [0, 0.5)")
        allowed = {"goal1", "goal3", "goal4", "goal5"}
        unknown = [field for field in self.gated_fields if field not in allowed]
        if unknown:
            raise ValueError(
                f"TASK2_GATED_FIELDS 只能取 {sorted(allowed)}（goal2 是检测器自身），收到 {unknown}"
            )
        if self.stitched_threshold <= 0:
            raise ValueError("stitched_threshold must be > 0")
        if self.stitched_scale <= 0:
            raise ValueError("stitched_scale must be > 0")
        if self.duplicate_scan_max_volumes < 0:
            raise ValueError("duplicate_scan_max_volumes must be >= 0")
        if self.max_pairs_per_study < 1:
            raise ValueError("max_pairs_per_study must be >= 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Task2Config":
        env = os.environ if environ is None else environ
        fields = _env_str(env, "TASK2_GATED_FIELDS", ",".join(DEFAULT_GATED_FIELDS))
        return cls(
            stitched_enabled=_env_bool(env, "TASK2_STITCHED", True),
            stitched_statistic=_env_str(env, "TASK2_STITCHED_STAT", "max"),
            stitched_metric=_env_str(env, "TASK2_STITCHED_METRIC", "curvature"),
            stitched_band=_env_float(env, "TASK2_STITCHED_BAND", DEFAULT_BAND),
            stitched_threshold=_env_float(env, "TASK2_STITCHED_THRESHOLD", DEFAULT_THRESHOLD),
            stitched_scale=_env_float(env, "TASK2_STITCHED_SCALE", DEFAULT_SCALE),
            stitched_gate=_env_bool(env, "TASK2_STITCHED_GATE", True),
            duplicate_enabled=_env_bool(env, "TASK2_DUPLICATE", True),
            duplicate_scan=_env_bool(env, "TASK2_DUPLICATE_SCAN", True),
            duplicate_scan_max_volumes=_env_int(env, "TASK2_DUPLICATE_SCAN_MAX_VOLUMES", 0),
            duplicate_gate=_env_bool(env, "TASK2_DUPLICATE_GATE", True),
            gated_fields=tuple(item.strip() for item in fields.split(",") if item.strip()),
            max_pairs_per_study=_env_int(env, "TASK2_MAX_PAIRS", 200),
        )

    def describe(self) -> dict[str, object]:
        return {
            "stitched": {
                "enabled": self.stitched_enabled,
                "statistic": self.stitched_statistic,
                "metric": self.stitched_metric,
                "band": self.stitched_band,
                "threshold": self.stitched_threshold,
                "scale": self.stitched_scale,
                "gate": self.stitched_gate,
            },
            "duplicate": {
                "enabled": self.duplicate_enabled,
                "scan_whole_dataset": self.duplicate_scan,
                "scan_max_volumes": self.duplicate_scan_max_volumes,
                "gate": self.duplicate_gate,
            },
            "gated_fields": list(self.gated_fields),
            "max_pairs_per_study": self.max_pairs_per_study,
        }


def gate_reason(context) -> str | None:
    gate = context.diagnostics.get(GATE_KEY)
    return None if not gate else str(gate.get("reason", "goal2"))


def _mark_gate(context, reason: str, detail: dict) -> None:
    context.diagnostics[GATE_KEY] = {"reason": reason, **detail}
    context.warnings.append(f"goal2 gate: {reason}")


class SpecialImageTask(StudyTask[StitchedResult]):
    """任务二主任务：拼接检测 + 重复检测，并决定是否对下游任务上闸。"""

    name = "goal2_stitched"

    def __init__(self, config: Task2Config | None = None) -> None:
        self.config = config or Task2Config.from_env()
        self.index: DuplicateIndex | None = None
        self.scan_report = ScanReport()
        self.matches: dict[str, list[str]] = {}
        self._scanned_root: Path | None = None

    def load_model(self) -> None:
        logger.info("task2: %s", self.config.describe())

    # -- 重复影像：先扫全库，再逐例比对 -------------------------------------
    def _ensure_index(self, study) -> None:
        if self.index is not None or not self.config.duplicate_enabled:
            return
        self.index = DuplicateIndex()
        first = study.series[0]
        root = dataset_root_of(first.source_path, study.accession_number, first.series_uid)
        self._scanned_root = root
        if self.config.duplicate_scan:
            try:
                self.index, self.scan_report = DuplicateIndex.from_dataset(
                    root,
                    max_volumes=self.config.duplicate_scan_max_volumes,
                )
                logger.info(
                    "task2: middle-slice index built from %s -> %s",
                    root,
                    self.scan_report.as_dict(),
                )
            except Exception as exc:  # noqa: BLE001 - 扫描失败退回增量模式
                logger.error("task2: dataset scan failed on %s: %s", root, exc)
                self.index = DuplicateIndex()
                self.scan_report = ScanReport(root=str(root))
                self.scan_report.errors.append(f"{type(exc).__name__}: {exc}")
        else:
            self.scan_report = ScanReport(root=str(root))

    def _duplicate_matches(self, study) -> list[str]:
        if not self.config.duplicate_enabled:
            return []
        self._ensure_index(study)
        signatures = [signature_from_array(series.image) for series in study.series]
        assert self.index is not None
        matched = self.index.match(signatures, study.accession_number)
        # 增量模式（未全库扫描）下，本检查的指纹入库，供后续检查比对
        if not self.config.duplicate_scan:
            self.index.add(study.accession_number, signatures)
        return matched

    # -- 主流程 -------------------------------------------------------------
    def predict(self, context) -> StitchedResult:
        study = context.study
        config = self.config

        stitched_score = 0.0
        stitched_detail: dict[str, object] = {}
        if config.stitched_enabled:
            result = score_study(
                ((series.series_uid, series.image) for series in study.series),
                config.stitched_statistic,
                config.stitched_metric,
                config.stitched_band,
            )
            stitched_score = result.score
            stitched_detail = result.as_dict()
        probability = probability_from_score(
            stitched_score,
            config.stitched_threshold,
            config.stitched_scale,
        )
        is_stitched = stitched_score >= config.stitched_threshold

        matched = self._duplicate_matches(study)
        if matched:
            self.matches[study.accession_number] = matched
        is_duplicate = bool(matched)

        context.diagnostics[DIAGNOSTIC_KEY] = {
            "AccessionNumber": study.accession_number,
            "IsStitchedProb": probability,
            "stitched": {
                "score": stitched_score,
                "threshold": config.stitched_threshold,
                "statistic": config.stitched_statistic,
                "flagged": is_stitched,
                **stitched_detail,
            },
            "duplicate": {
                "flagged": is_duplicate,
                "matches": matched,
                "scanned_root": None if self._scanned_root is None else str(self._scanned_root),
                "index_studies": len(self.index) if self.index is not None else 0,
                "scan": self.scan_report.as_dict(),
            },
        }

        if is_duplicate and config.duplicate_gate:
            _mark_gate(context, "duplicate", {"matches": matched})
        elif is_stitched and config.stitched_gate:
            _mark_gate(context, "stitched", {"score": stitched_score})
        return StitchedResult(stitched_probability=float(probability))


class GatedStudyTask(StudyTask):
    """下游任务闸门：命中拼接/重复时直接给出中性结果，不执行模型。"""

    def __init__(
        self,
        inner: StudyTask,
        context_field: str,
        neutral: Callable[[object], object],
        *,
        gated_fields: Sequence[str] = DEFAULT_GATED_FIELDS,
    ) -> None:
        self.inner = inner
        self.context_field = context_field
        self.neutral = neutral
        self.gated_fields = tuple(gated_fields)
        self.name = inner.name

    def load_model(self) -> None:
        self.inner.load_model()

    def predict(self, context):
        if self.context_field in self.gated_fields:
            reason = gate_reason(context)
            if reason:
                skips = context.diagnostics.setdefault(SKIP_KEY, {})
                skips[self.context_field] = reason
                return self.neutral(context)
        return self.inner.predict(context)


class DuplicatePairRecorder(DatasetTask[DuplicateResult]):
    """把 ``SpecialImageTask`` 的命中结果汇总成 ``duplicate_pairs.jsonl``。"""

    name = "goal2_duplicate"

    def __init__(self, max_pairs_per_study: int = 200) -> None:
        self.max_pairs_per_study = max(1, int(max_pairs_per_study))
        self.reset()

    def reset(self) -> None:
        self._pairs: dict[tuple[str, str], float] = {}
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, study, context) -> None:
        detail = context.diagnostics.get(DIAGNOSTIC_KEY, {})
        matches = detail.get("duplicate", {}).get("matches", []) if detail else []
        for other in matches:
            key = tuple(sorted((study.accession_number, str(other))))
            if key[0] == key[1] or key in self._pairs:
                continue
            if self._counts[key[0]] >= self.max_pairs_per_study:
                continue
            if self._counts[key[1]] >= self.max_pairs_per_study:
                continue
            self._pairs[key] = 1.0
            self._counts[key[0]] += 1
            self._counts[key[1]] += 1

    def finalize(self) -> DuplicateResult:
        pairs = tuple(
            DuplicatePair(left, right, probability)
            for (left, right), probability in sorted(self._pairs.items())
        )
        logger.info("task2: %d duplicate pair(s) reported", len(pairs))
        return DuplicateResult(pairs=pairs)


# --------------------------------------------------------------------------
# 离线自检：在赛方数据上量一下两条规则到底有多准
# --------------------------------------------------------------------------
def _parse_gold_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip().replace("\t", ",")
        if not text or text.startswith("#"):
            continue
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if len(parts) >= 2 and parts[0].lower() not in {"src_img", "studyuid"}:
            pairs.append(tuple(sorted((parts[0], parts[1]))))
    return pairs


def evaluate_official_data(
    data_root: Path,
    *,
    annotation_root: Path | None = None,
    config: Task2Config | None = None,
    out_dir: Path | None = None,
    target_fpr: float = 0.05,
    max_volumes: int = 0,
) -> dict:
    """在赛方训练数据上评估任务二两条规则：拼接用 Composition，重复用 duplicate 金标准。"""
    from .dataset import find_special_root, iter_images, special_dir
    from .metrics import clean_report, metrics_report

    config = config or Task2Config.from_env()
    root = Path(data_root).expanduser()
    ann = annotation_root or find_special_root(root)
    composition = special_dir(ann, "composition") if ann else None
    duplicate_dir = special_dir(ann, "duplicate") if ann else None
    skips = tuple(path for path in (composition, duplicate_dir) if path)

    positives = [path for path in (iter_images(composition) if composition else [])]
    negatives = list(iter_images(ann, skip_dirs=skips)) if ann else []
    if ann is not None:
        negatives += iter_images(root, skip_dirs=((ann,) + skips))

    def score(path: Path) -> float | None:
        try:
            from .stitched import volume_stitched_score

            return volume_stitched_score(
                path, config.stitched_statistic, config.stitched_metric, config.stitched_band
            )
        except Exception:  # noqa: BLE001
            return None

    positive_scores = [value for value in (score(path) for path in positives) if value is not None]
    negative_scores = [value for value in (score(path) for path in (negatives[: max_volumes or len(negatives)])) if value is not None]

    report: dict[str, object] = {
        "data_root": str(root),
        "annotation_root": None if ann is None else str(ann),
        "config": config.describe(),
        "stitched": {
            "positives": len(positive_scores),
            "negatives": len(negative_scores),
        },
        "duplicate": {},
    }
    if positive_scores and negative_scores:
        labels = np.asarray([1.0] * len(positive_scores) + [0.0] * len(negative_scores))
        values = np.asarray(positive_scores + negative_scores)
        threshold = float(np.quantile(np.asarray(negative_scores), 1.0 - target_fpr))
        report["stitched"].update({
            "default_threshold": config.stitched_threshold,
            "suggested_threshold": round(threshold, 6),
            "suggested_recall": float((np.asarray(positive_scores) >= threshold).mean()),
            "suggested_negative_fpr": float((np.asarray(negative_scores) >= threshold).mean()),
            "positive_median": float(np.median(positive_scores)),
            "negative_median": float(np.median(negative_scores)),
            "ranking_metrics": clean_report(metrics_report(labels, values)),
        })
    else:
        report["stitched"]["error"] = "need annotation/Composition positives and normal negatives"

    if duplicate_dir is not None and config.duplicate_enabled:
        # 重复影像的检查号是 duplicate/ 下的一级目录；正常影像的检查号是标注根目录下的一级目录
        dup_index, dup_scan = DuplicateIndex.from_dataset(
            duplicate_dir,
            max_volumes=config.duplicate_scan_max_volumes,
        )
        normal_index, normal_scan = DuplicateIndex.from_dataset(
            ann if ann is not None else root,
            max_volumes=config.duplicate_scan_max_volumes,
            skip_dirs=tuple(path for path in (composition, duplicate_dir) if path),
        )
        gold_files = [
            path for path in sorted(duplicate_dir.rglob("*"))
            if path.is_file() and "nii" not in path.suffix.lower()
        ]
        gold: list[tuple[str, str]] = []
        for path in gold_files:
            gold.extend(_parse_gold_file(path))
        detected = _all_pairs(dup_index)
        normal_pairs = _all_pairs(normal_index)
        hits = [pair for pair in gold if pair in detected]
        report["duplicate"] = {
            "scanned": dup_scan.as_dict(),
            "index_studies": len(dup_index),
            "detected_pairs": len(detected),
            "detected": [list(pair) for pair in sorted(detected)][:10],
            "gold_files": [str(path) for path in gold_files],
            "gold_pairs": len(gold),
            "recall": None if not gold else round(len(hits) / len(gold), 6),
            "missed": [list(pair) for pair in gold if pair not in detected][:10],
            "normal_volumes_scanned": normal_scan.as_dict(),
            "normal_false_pairs": len(normal_pairs),
            "normal_false_pair_examples": [list(pair) for pair in sorted(normal_pairs)][:10],
        }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "goal2_evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _all_pairs(index: DuplicateIndex) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for accession in index.signatures:
        for other in index.match(index.signatures[accession], accession):
            pairs.add(tuple(sorted((accession, other))))
    return pairs


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="在赛方数据上自检任务二的两条规则")
    parser.add_argument("--data-root", type=Path, default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--max-volumes", type=int, default=0)
    parser.add_argument("--stitched-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--stitched-metric", default="curvature", choices=METRICS)
    parser.add_argument("--stitched-stat", default="max", choices=STATISTICS)
    parser.add_argument("--stitched-band", type=float, default=DEFAULT_BAND)
    args = parser.parse_args(argv)

    config = Task2Config(
        stitched_threshold=args.stitched_threshold,
        stitched_metric=args.stitched_metric,
        stitched_statistic=args.stitched_stat,
        stitched_band=args.stitched_band,
    )
    report = evaluate_official_data(
        args.data_root,
        annotation_root=args.annotation_root,
        config=config,
        out_dir=args.out_dir,
        target_fpr=args.target_fpr,
        max_volumes=args.max_volumes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
