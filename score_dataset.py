"""目标一的独立批量打分脚本（不依赖比赛 HTTP 链路，便于自测与调权）。

两种输入方式：

* ``--dataset``：``<case>/<SeriesUid>/<SeriesUid>.nii.gz`` 的比赛布局，
  或平铺 ``*.nii(.gz)`` 的单检查目录；
* ``--manifest``：JSONL，每行至少含 ``path``（相对 ``--data-root``），
  可选 ``is_not_human_body`` 标签，给了标签就同时打印 AP / ROC-AUC / Recall@10%FPR。

输出：``<out-dir>/authenticity_scores.jsonl`` 与 ``<out-dir>/<Accession>/prediction.json``。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Task1Config
from .metrics import clean_report, metrics_report
from .preprocess import list_series
from .scorer import AuthenticityScorer

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def _discover(dataset: Path) -> list[tuple[str, list[tuple[str, Path]]]]:
    cases: list[tuple[str, list[tuple[str, Path]]]] = []
    for child in sorted(path for path in dataset.iterdir() if path.is_dir()):
        series = list_series(child)
        if series:
            cases.append((child.name, series))
    if cases:
        return cases
    series = list_series(dataset)
    if series:
        return [(dataset.name or "case", series)]
    return []


def _from_manifest(manifest: Path, data_root: Path) -> list[tuple[str, list[tuple[str, Path]]]]:
    cases = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        path = Path(record["path"])
        if not path.is_absolute():
            path = data_root / path
        stem = path.name
        for suffix in NIFTI_SUFFIXES:
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        cases.append((str(record.get("accession") or stem), [(stem, path)]))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="manifest 中相对路径的根目录")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=None,
                        help="覆盖 task1/weights 下的默认权重，可重复")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    config = Task1Config.from_env()
    if args.checkpoint:
        config = Task1Config(
            **{**config.__dict__, "weights": tuple(args.checkpoint)}
        )
    if args.device:
        config = Task1Config(**{**config.__dict__, "device": args.device})

    if args.manifest:
        data_root = (args.data_root or args.manifest.parent).resolve()
        cases = _from_manifest(args.manifest, data_root)
        labels = [
            json.loads(line).get("is_not_human_body")
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        cases = _discover(args.dataset.resolve())
        labels = [None] * len(cases)
    if args.limit:
        cases, labels = cases[: args.limit], labels[: args.limit]
    if not cases:
        parser.error("no NIfTI examinations found")

    scorer = AuthenticityScorer(config)
    scorer.load()
    print(json.dumps(scorer.describe(), ensure_ascii=False, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.out_dir / "authenticity_scores.jsonl"
    collected_scores: list[float] = []
    collected_labels: list[float] = []

    with scores_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (accession, series) in enumerate(cases, 1):
            started = time.perf_counter()
            result = scorer.score_study(series)
            probability = result.probability
            if probability is None:
                probability = config.fallback_probability
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            case_dir = args.out_dir / accession
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "prediction.json").write_text(
                json.dumps(
                    {
                        "AccessionNumber": accession,
                        "IsNotHumanBodyProb": round(float(probability), 6),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            record = {
                "AccessionNumber": accession,
                "IsNotHumanBodyProb": round(float(probability), 6),
                "ProcessingTime_ms": elapsed_ms,
                **result.as_dict(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            collected_scores.append(float(probability))
            if labels[index - 1] is not None:
                collected_labels.append(float(labels[index - 1]))
            print(
                f"[{index}/{len(cases)}] {accession} "
                f"IsNotHumanBodyProb={float(probability):.4f} "
                f"series={result.scored_series}/{len(result.series)} {elapsed_ms}ms",
                flush=True,
            )

    if collected_labels and len(collected_labels) == len(collected_scores):
        payload = clean_report(metrics_report(collected_labels, collected_scores))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        (args.out_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"scores: {scores_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
