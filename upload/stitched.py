"""任务二（拼接影像）：相邻层平均法。

规则（与赛方说明一致）：对每一条序列的每一层，取上下两层求平均，把当前层与该平均值比较，
差异过大即判为拼接影像。实现细节：

* 层方向取最短轴，与 task1 推理口径一致；
* 单层差异 ``d_i = mean(|I_i - (I_{i-1} + I_{i+1}) / 2|)``，再除以全卷亮度尺度
  ``mean(|I|)`` 做无量纲化，保证不同机器/序列之间可比；
* 检查级分数默认取该检查所有序列的**最大**单层差异（一层拼接就足以判定）；
* 分数按单调 logistic 映射成概率字段 ``IsStitchedProb``：
  ``p = sigmoid(scale * (score / threshold - 1))``，即分数等于阈值时 p = 0.5。

提供三种单层差异口径（``--metric`` / ``TASK2_STITCHED_METRIC``）：

* ``curvature``（默认，即上面的规则）：当前层与上下层均值之差；
* ``adjacent``：相邻层差 ``mean|I_i - I_{i-1}|``；
* ``local``：把层间差除以**该卷自身**的层间差中位数（±窗口），即「相对自己是否异常」，
  分母带亮度下限避免空层导致数值爆炸。

另外默认丢弃层方向两端各 ``band`` 比例的层（``--band`` / ``TASK2_STITCHED_BAND``，默认 0.1）：
任何体数据在 FOV 边缘（颈部、空气进入/离开视野）本来就会出现大的层间变化，
把它们计入会让正常影像被误判。

阈值可以用赛方数据标定（``annotation/Composition`` 为正类，其余正常影像为负类）::

    python -m task1.stitched --data-root /2026aicompetition/datasets/training \
        --out-dir /2026aicompetition/workspace/task1_runs --target-fpr 0.05
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .dataset import find_special_root, iter_images, special_dir
from .preprocess import to_float_volume

# 本地用真实体数据 + 人工拼接体数据标定的经验阈值，服务器上请用 `python -m task1.stitched` 重新标定。
DEFAULT_THRESHOLD = 0.16
DEFAULT_SCALE = 8.0
STATISTICS = ("max", "p99", "mean")
METRICS = ("curvature", "adjacent", "local")
LOCAL_WINDOW = 4
DEFAULT_BAND = 0.1


def _slice_axis(shape: Sequence[int]) -> int:
    return int(np.argmin(shape[:3]))


def slice_residual_scores(
    volume: np.ndarray | str | Path,
    metric: str = "curvature",
    band: float = DEFAULT_BAND,
) -> np.ndarray:
    """逐层差异分数（无量纲）。层方向 = 最短轴。"""
    array = to_float_volume(volume)
    if array.ndim < 3 or min(array.shape) < 1:
        return np.zeros(0, dtype=np.float32)
    moved = np.moveaxis(array, _slice_axis(array.shape), 0)
    count = moved.shape[0]
    if count < 3:
        return np.zeros(0, dtype=np.float32)
    previous, current, following = moved[:-2], moved[1:-1], moved[2:]
    axes = tuple(range(1, moved.ndim))
    curvature = np.abs(
        current.astype(np.float32) - 0.5 * (previous.astype(np.float32) + following.astype(np.float32))
    ).mean(axis=axes)
    adjacent = np.abs(np.diff(moved.astype(np.float32), axis=0)).mean(axis=axes)   # 长度 count-1
    scale = float(np.mean(np.abs(moved)))
    if not math.isfinite(scale) or scale <= 1e-6:
        scale = float(np.std(moved)) or 1.0

    if metric == "adjacent":
        raw = adjacent
    elif metric == "local":
        # 相对自身：curvature_i / 局部中位数，分母带亮度下限，避免空层导致数值爆炸
        floor = 0.02 * scale
        raw = np.zeros_like(curvature)
        for index in range(curvature.size):
            low = max(0, index - LOCAL_WINDOW)
            high = min(curvature.size, index + LOCAL_WINDOW + 1)
            baseline = max(float(np.median(curvature[low:high])), floor)
            raw[index] = curvature[index] / baseline
        return np.asarray(raw, dtype=np.float32)
    elif metric != "curvature":
        raise ValueError(f"metric must be one of {METRICS}")
    else:
        raw = curvature
    raw = np.asarray(raw / scale, dtype=np.float32)
    if 0.0 < band < 0.5 and raw.size > 2:
        margin = int(round(raw.size * band))
        if margin > 0 and raw.size - 2 * margin >= 1:
            raw = raw[margin:raw.size - margin]
    return raw


def _reduce(scores: np.ndarray, statistic: str) -> float:
    if scores.size == 0:
        return 0.0
    if statistic == "p99":
        return float(np.percentile(scores, 99))
    if statistic == "mean":
        return float(scores.mean())
    return float(scores.max())


def volume_stitched_score(
    volume: np.ndarray | str | Path,
    statistic: str = "max",
    metric: str = "curvature",
    band: float = DEFAULT_BAND,
) -> float:
    return _reduce(slice_residual_scores(volume, metric, band), statistic)


@dataclass(frozen=True)
class SeriesStitchedScore:
    series_uid: str
    score: float
    slice_index: int | None
    slices: int


@dataclass(frozen=True)
class StudyStitchedScore:
    score: float
    series: tuple[SeriesStitchedScore, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "series": [
                {
                    "series_uid": item.series_uid,
                    "score": item.score,
                    "worst_slice": item.slice_index,
                    "slices": item.slices,
                }
                for item in self.series
            ],
        }


def score_study(
    series: Iterable[tuple[str, np.ndarray | str | Path]],
    statistic: str = "max",
    metric: str = "curvature",
    band: float = DEFAULT_BAND,
) -> StudyStitchedScore:
    """一个检查（多条序列）的拼接分数 = 各序列的最大值。"""
    details: list[SeriesStitchedScore] = []
    best = 0.0
    for series_uid, source in series:
        try:
            scores = slice_residual_scores(source, metric, band)
        except Exception:  # noqa: BLE001 - 单条序列失败不影响判定
            details.append(SeriesStitchedScore(series_uid, 0.0, None, 0))
            continue
        value = _reduce(scores, statistic)
        worst = int(np.argmax(scores)) + 1 if scores.size else None
        details.append(SeriesStitchedScore(series_uid, value, worst, int(scores.size)))
        best = max(best, value)
    return StudyStitchedScore(best, tuple(details))


def probability_from_score(score: float, threshold: float, scale: float = DEFAULT_SCALE) -> float:
    """分数 -> [0,1] 概率：分数 = 阈值时 0.5，越大越接近 1。"""
    threshold = max(float(threshold), 1e-6)
    ratio = float(score) / threshold - 1.0
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, scale * ratio))))


def _score_scan(root: Path, annotation_root: Path | None, statistic: str,
                metric: str, band: float, max_volumes: int) -> tuple[list[float], list[float]]:
    ann = annotation_root or find_special_root(root)
    composition = special_dir(ann, "composition") if ann else None
    positives = list(iter_images(composition)) if composition else []
    negatives: list[Path] = []
    if ann is not None:
        skips = tuple(
            path for path in (special_dir(ann, "fake"), composition, special_dir(ann, "duplicate")) if path
        )
        negatives += iter_images(ann, skip_dirs=skips)
    negatives += iter_images(root, skip_dirs=tuple(path for path in (ann, composition) if path))

    def collect(paths: list[Path]) -> list[float]:
        values: list[float] = []
        for path in paths:
            if 0 < max_volumes <= len(values):
                break
            try:
                values.append(volume_stitched_score(path, statistic, metric, band))
            except Exception:  # noqa: BLE001
                continue
        return values

    return collect(positives), collect(negatives)


def main(argv: list[str] | None = None) -> int:
    from .metrics import clean_report, metrics_report

    parser = argparse.ArgumentParser(description="Calibrate the stitched-slice threshold")
    parser.add_argument("--data-root", type=Path, default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="写入 calibration.json")
    parser.add_argument("--statistic", choices=STATISTICS, default="max")
    parser.add_argument("--metric", choices=METRICS, default="curvature")
    parser.add_argument("--band", type=float, default=DEFAULT_BAND,
                        help="丢弃层方向两端各该比例的层（默认 0.1）")
    parser.add_argument("--target-fpr", type=float, default=0.05,
                        help="阈值取负类分数的 (1 - target_fpr) 分位")
    parser.add_argument("--max-volumes", type=int, default=0)
    args = parser.parse_args(argv)

    positives, negatives = _score_scan(
        args.data_root, args.annotation_root, args.statistic, args.metric, args.band, args.max_volumes
    )
    if not positives or not negatives:
        print(json.dumps({
            "positives": len(positives),
            "negatives": len(negatives),
            "error": "need both annotation/Composition positives and normal negatives",
        }, ensure_ascii=False, indent=2))
        return 2

    threshold = float(np.quantile(np.asarray(negatives), 1.0 - args.target_fpr))
    labels = np.asarray([1.0] * len(positives) + [0.0] * len(negatives))
    scores = np.asarray(positives + negatives)
    report = clean_report(metrics_report(labels, scores))
    flagged = np.asarray(negatives) >= threshold
    payload = {
        "statistic": args.statistic,
        "metric": args.metric,
        "band": args.band,
        "threshold": round(threshold, 6),
        "target_fpr": args.target_fpr,
        "positives": len(positives),
        "negatives": len(negatives),
        "negative_fpr_at_threshold": float(flagged.mean()),
        "recall_at_threshold": float((np.asarray(positives) >= threshold).mean()),
        "positive_score_median": float(np.median(positives)),
        "negative_score_median": float(np.median(negatives)),
        "ranking_metrics": report,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "stitched_calibration.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
