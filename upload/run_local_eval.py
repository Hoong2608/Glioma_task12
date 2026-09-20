"""本地端到端跑一遍任务一管线（等价于一次 ``/call``，但不发 callback）。

在管线仓库根目录执行::

    python -m task1.run_local_eval --dataset <测试集> --output <answer 目录>

输出目录结构与比赛要求完全一致：``<answer>/<evaluation_id>/<AccessionNumber>/prediction.json``。
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from ._bootstrap import ensure_pipeline_root

ensure_pipeline_root()

from core.config import Settings  # noqa: E402
from core.runner import EvaluationJob, EvaluationRunner  # noqa: E402

from .config import Task1Config  # noqa: E402
from .pipeline_factory import build_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-id", default="task1-local")
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None,
                        help="把逐例 IsNotHumanBodyProb 审计记录写到这里（默认 <output>/../task1_scores.jsonl）")
    args = parser.parse_args(argv)

    config = Task1Config.from_env()
    output_root = args.output.resolve()
    log_root = (args.log_root or output_root.parent / "logs").resolve()
    settings = Settings(
        workspace=output_root.parent,
        answer_root=output_root,
        log_root=log_root,
        callback_url=None,
    )
    runner = EvaluationRunner(settings, pipeline=build_pipeline(config))
    result = runner.run(
        EvaluationJob(
            request_id=f"task1-{uuid.uuid4()}",
            evaluation_id=str(args.evaluation_id),
            dataset_path=args.dataset.resolve(),
        ),
        send_callback=False,
    )
    print(f"published: {result}")

    scores_path = args.scores or (output_root.parent / "task1_scores.jsonl")
    rows = []
    for case_dir in sorted(path for path in Path(result).iterdir() if path.is_dir()):
        payload = json.loads((case_dir / "prediction.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "AccessionNumber": payload["AccessionNumber"],
                "IsNotHumanBodyProb": payload["IsNotHumanBodyProb"],
                "IsStitchedProb": payload.get("IsStitchedProb"),
                "ProcessingTime_ms": payload.get("ProcessingTime_ms"),
            }
        )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows:
        print(
            f"  {row['AccessionNumber']}: IsNotHumanBodyProb={row['IsNotHumanBodyProb']:.4f} "
            f"({row['ProcessingTime_ms']} ms)"
        )
    print(f"scores: {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
