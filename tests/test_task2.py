"""任务二自检：拼接检测、重复影像索引、下游闸门、端到端输出。

在 ``task1`` 的上一层目录执行::

    python -m unittest discover -s task1/tests -t . -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TASK1_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = TASK1_ROOT.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from task1._bootstrap import ensure_pipeline_root  # noqa: E402

ensure_pipeline_root()

from task1.goal2 import GATE_KEY, SKIP_KEY, Task2Config  # noqa: E402
from task1.stitched import (  # noqa: E402
    DEFAULT_THRESHOLD,
    probability_from_score,
    slice_residual_scores,
    volume_stitched_score,
)

WEIGHTS = tuple(sorted((TASK1_ROOT / "weights").glob("*.pt")))


def _need_torch() -> bool:
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def synthetic_volume(seed: int, shape: tuple[int, int, int] = (16, 32, 32)) -> np.ndarray:
    """平滑的「正常」体数据：层间连续、带轻微噪声。"""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0.0, 20.0, shape[0])[:, None, None]
    blob = np.exp(-((np.linspace(-1, 1, shape[1]))[None, :, None] ** 2))
    return (100.0 + gradient * blob + rng.normal(0.0, 1.5, shape)).astype(np.float32)


def splice_volume(seed: int = 1, cut: int = 8, factor: float = 0.45) -> np.ndarray:
    """把两段不同亮度的体数据拼在一起，模拟拼接影像。"""
    volume = synthetic_volume(seed)
    top = volume[:cut]
    bottom = volume[cut:] * factor + 25.0
    return np.concatenate([top, bottom], axis=0).astype(np.float32)


def make_duplicate_dataset(root: Path, *, shape: tuple[int, int, int] = (24, 48, 48)) -> None:
    """ACC001 与 ACC002 逐像素相同（重复影像），ACC003 独立。"""
    import nibabel as nib

    base = synthetic_volume(3, shape)
    other = synthetic_volume(9, shape)
    for accession, image, suffix in (
        ("ACC001", base, ".nii.gz"),
        ("ACC002", base, ".nii"),          # 同一份像素、不同容器/后缀
        ("ACC003", other, ".nii.gz"),
    ):
        directory = root / accession / "T1"
        directory.mkdir(parents=True)
        nib.save(
            nib.Nifti1Image(image, np.eye(4)),
            str(directory / f"T1{suffix}"),
        )


class StitchedDetectionTest(unittest.TestCase):
    """拼接检测：正常体积分低，拼接处分数显著升高。"""

    def test_splice_scores_higher_than_normal(self) -> None:
        normal = synthetic_volume(1)
        spliced = splice_volume(1)

        normal_score = volume_stitched_score(normal)
        spliced_score = volume_stitched_score(spliced)
        self.assertLess(normal_score, spliced_score)
        self.assertGreater(spliced_score, normal_score * 3.0)
        self.assertGreater(normal_score, 0.0)

        # 最差层应落在拼接缝（第 cut 层）附近；band 裁剪会让索引略平移
        scores = slice_residual_scores(spliced)
        self.assertLessEqual(abs(int(np.argmax(scores)) + 1 - 8), 3)

    def test_default_threshold_separates_clear_cases(self) -> None:
        normal_probability = probability_from_score(
            volume_stitched_score(synthetic_volume(2)), DEFAULT_THRESHOLD
        )
        spliced_probability = probability_from_score(
            volume_stitched_score(splice_volume(2)), DEFAULT_THRESHOLD
        )
        self.assertLess(normal_probability, 0.5)
        self.assertGreater(spliced_probability, 0.5)
        self.assertLess(normal_probability, spliced_probability)

    def test_probability_is_monotone_and_bounded(self) -> None:
        values = [probability_from_score(score, DEFAULT_THRESHOLD) for score in (0.0, 0.05, 0.16, 0.4, 2.0)]
        self.assertEqual(sorted(values), values)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertAlmostEqual(0.5, probability_from_score(DEFAULT_THRESHOLD, DEFAULT_THRESHOLD), places=6)


class DuplicateIndexTest(unittest.TestCase):
    """重复影像：全库中间层指纹索引。"""

    def test_exact_copy_is_matched(self) -> None:
        from task1.middle_slice import DuplicateIndex, signature_from_array

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            make_duplicate_dataset(root)
            index, report = DuplicateIndex.from_dataset(root)

            self.assertEqual(3, report.studies)
            self.assertEqual(3, report.volumes)

            base = synthetic_volume(3, (24, 48, 48))
            other = synthetic_volume(9, (24, 48, 48))
            self.assertEqual(["ACC002"], index.match([signature_from_array(base)], "ACC001"))
            self.assertEqual(["ACC001"], index.match([signature_from_array(base)], "ACC002"))
            self.assertEqual([], index.match([signature_from_array(other)], "ACC003"))


class GateTest(unittest.TestCase):
    """下游闸门：命中拼接/重复后不再执行下游任务。"""

    def test_gated_wrapper_skips_inner_task(self) -> None:
        from pipeline.context import PipelineContext

        from task1.goal2 import GatedStudyTask
        from tasks.base import StudyTask
        from tasks.results import Goal1Result

        class Counter(StudyTask[Goal1Result]):
            name = "counter"

            def __init__(self) -> None:
                self.calls = 0

            def predict(self, context) -> Goal1Result:
                self.calls += 1
                return Goal1Result(not_human_probability=0.1)

        inner = Counter()
        gated = GatedStudyTask(
            inner,
            "goal3",
            lambda context: Goal1Result(not_human_probability=0.5),
            gated_fields=("goal3",),
        )
        context = PipelineContext(study=None)
        self.assertEqual(0.1, gated.predict(context).not_human_probability)
        self.assertEqual(1, inner.calls)

        context.diagnostics[GATE_KEY] = {"reason": "duplicate"}
        self.assertEqual(0.5, gated.predict(context).not_human_probability)
        self.assertEqual(1, inner.calls)          # 闸门生效，没有再次调用模型
        self.assertEqual("duplicate", context.diagnostics[SKIP_KEY]["goal3"])


class PipelineGoal2Test(unittest.TestCase):
    """任务二接入任务链后的行为与最终输出。"""

    def setUp(self) -> None:
        if not WEIGHTS:
            self.skipTest("weights missing")
        if not _need_torch():
            self.skipTest("torch/timm not installed")

    def _build(self):
        from task1.config import Task1Config
        from task1.pipeline_factory import build_pipeline

        return build_pipeline(
            Task1Config(weights=(WEIGHTS[0],), device="cpu", strict=True),
            Task2Config(gated_fields=("goal3", "goal4", "goal5")),
        )

    def test_duplicate_study_is_gated_and_reported(self) -> None:
        from data.loader import DatasetLoader

        # 管线 Loader 对未压缩 .nii 走内存映射，Windows 上删除临时目录会撞文件锁
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            dataset_path = Path(temporary) / "dataset"
            make_duplicate_dataset(dataset_path)

            pipeline = self._build()
            contexts, duplicates = pipeline.run(DatasetLoader().load(dataset_path))

            flagged = contexts["ACC002"]
            self.assertEqual("duplicate", flagged.diagnostics[GATE_KEY]["reason"])
            self.assertEqual(["ACC001"], flagged.diagnostics[GATE_KEY]["matches"])
            for field in ("goal3", "goal4", "goal5"):
                self.assertEqual("duplicate", flagged.diagnostics[SKIP_KEY][field])
            self.assertEqual("model", flagged.diagnostics["goal1"]["source"])   # goal1 仍执行

            clean = contexts["ACC003"]
            self.assertNotIn(GATE_KEY, clean.diagnostics)
            self.assertNotIn(SKIP_KEY, clean.diagnostics)

            pairs = {(pair.left_accession, pair.right_accession): pair.probability for pair in duplicates.pairs}
            self.assertEqual({("ACC001", "ACC002"): 1.0}, pairs)
            del contexts, duplicates

    def test_end_to_end_output_passes_validator(self) -> None:
        from core.config import Settings
        from core.runner import EvaluationJob, EvaluationRunner
        from data.loader import DatasetLoader
        from output.validator import OutputValidator

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            make_duplicate_dataset(dataset_path)
            settings = Settings(
                workspace=root / "workspace",
                answer_root=root / "workspace" / "answer",
                log_root=root / "workspace" / "logs",
                callback_url=None,
            )
            output = EvaluationRunner(settings, pipeline=self._build()).run(
                EvaluationJob("goal2-request", "goal2-evaluation", dataset_path),
                send_callback=False,
            )

            lines = [
                json.loads(line)
                for line in (output / "duplicate_pairs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, len(lines))
            self.assertEqual({"StudyUID", "StudyUID_dup", "PairProb"}, set(lines[0]))
            self.assertEqual({"ACC001", "ACC002"}, {lines[0]["StudyUID"], lines[0]["StudyUID_dup"]})
            self.assertEqual(1.0, lines[0]["PairProb"])

            for accession in ("ACC001", "ACC002", "ACC003"):
                payload = json.loads(
                    (output / accession / "prediction.json").read_text(encoding="utf-8")
                )
                self.assertIn("IsStitchedProb", payload)
                self.assertGreaterEqual(payload["IsStitchedProb"], 0.0)
                self.assertLessEqual(payload["IsStitchedProb"], 1.0)

            OutputValidator().validate(output, DatasetLoader().load(dataset_path))


if __name__ == "__main__":
    unittest.main()
