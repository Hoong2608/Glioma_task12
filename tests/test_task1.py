"""任务一自检：数据扫描、训练链路、与主文件夹方法等价、管线端到端。

在 ``task1`` 的上一层目录执行::

    python -m unittest discover -s task1/tests -t . -v
"""
from __future__ import annotations

import json
import os
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

REPO_ROOT = ensure_pipeline_root() or PLUGIN_ROOT

WEIGHTS = tuple(sorted((TASK1_ROOT / "weights").glob("*.pt")))


def _find_main_project_root() -> Path | None:
    """向上找带 ``models/authenticity_inference.py`` 的项目主文件夹。"""
    for base in (PLUGIN_ROOT, *PLUGIN_ROOT.parents):
        if (base / "models" / "authenticity_inference.py").is_file():
            return base
    return None


MAIN_PROJECT_ROOT = _find_main_project_root()
REFERENCE_MODULE = (
    None if MAIN_PROJECT_ROOT is None
    else MAIN_PROJECT_ROOT / "models" / "authenticity_inference.py"
)
PUBLIC_DIR = (
    None if MAIN_PROJECT_ROOT is None
    else MAIN_PROJECT_ROOT / "public_authenticity_data" / "raw" / "human_ixi" / "mri_IXI_480_2"
)
PUBLIC_VOLUMES = tuple(sorted(PUBLIC_DIR.glob("*.nii*"))) if PUBLIC_DIR and PUBLIC_DIR.is_dir() else ()
PUBLIC_VOLUME = PUBLIC_VOLUMES[0] if PUBLIC_VOLUMES else None


def _need_torch() -> bool:
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def make_official_tree(root: Path, *, normal_cases: int = 6, fake_cases: int = 6) -> None:
    """按赛方目录结构造一份小数据：annotation/{fake,Composition,duplicate,正常...}。"""
    import nibabel as nib

    rng = np.random.default_rng(0)

    def write(relative: str, seed: int) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image = rng.normal(100.0 + seed * 3.0, 20.0, (16, 32, 32)).astype(np.float32)
        nib.save(nib.Nifti1Image(image, np.eye(4)), str(path))

    for index in range(1, fake_cases + 1):
        write(f"annotation/fake/FAKE{index:02d}/T1/T1.nii.gz", index)
    for index in range(1, 3):
        write(f"annotation/Composition/COMP{index:02d}/T1/T1.nii.gz", index)
    for index in range(1, 3):
        write(f"annotation/duplicate/DUP{index:02d}/T1/T1.nii.gz", index)
    for index in range(1, normal_cases + 1):
        write(f"annotation/NORM{index:02d}/T1/T1.nii.gz", index)
    for index in range(1, 3):
        write(f"images/OTH{index:02d}/T1/T1.nii.gz", index)
    # 标注文件不应被当作输入影像
    write("annotation/NORM01/T1/T1_mask.nii.gz", 0)
    (root / "annotation" / "duplicate").mkdir(parents=True, exist_ok=True)
    (root / "annotation" / "duplicate" / "gold.txt").write_text(
        "DUP01, DUP02\n", encoding="utf-8"
    )


class OfficialDataTest(unittest.TestCase):
    """赛方目录扫描、标签与切分（不需要 torch）。"""

    def test_discovery_labels_and_splits(self) -> None:
        from task1.dataset import assign_splits, discover_records, summarize

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_official_tree(root)
            records = discover_records(root, composition="exclude", duplicate="exclude")
            summary = summarize(records)

            self.assertEqual(6, summary["positives"])
            self.assertEqual(8, summary["negatives"])
            self.assertEqual(14, summary["volumes"])
            self.assertTrue(all(record.label == 0 for record in records if record.group == "normal"))
            self.assertTrue(all(record.label == 1 for record in records if record.group == "fake"))
            self.assertFalse([record for record in records if record.group in {"composition", "duplicate"}])
            self.assertFalse([record for record in records if "mask" in Path(record.path).name])

            assign_splits(records, val_fraction=0.3, seed=1)
            split_summary = summarize(records)["by_split"]
            self.assertGreater(split_summary["val"]["positives"], 0)
            self.assertGreater(split_summary["train"]["positives"], 0)
            self.assertEqual(14, sum(info["volumes"] for info in split_summary.values()))
            self.assertGreater(split_summary["val"]["cases"], 0)

    def test_specials_can_be_used_as_negatives(self) -> None:
        from task1.dataset import discover_records, summarize

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_official_tree(root)
            records = discover_records(root, composition="negative", duplicate="negative")
            summary = summarize(records)
            self.assertEqual(6, summary["positives"])
            self.assertEqual(12, summary["negatives"])
            self.assertEqual({"composition": 2, "duplicate": 2}, {
                key: summary["by_group"][key] for key in ("composition", "duplicate")
            })


class ExtensionFallbackTest(unittest.TestCase):
    """后缀与实际压缩格式不符的 NIfTI 也要能读（赛方数据里出现过）。"""

    def test_plain_nifti_named_gz_is_readable(self) -> None:
        import nibabel as nib

        from task1.preprocess import load_volume

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
            plain = root / "T1CE.nii"
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(plain))
            mislabeled = root / "T1CE.nii.gz"
            mislabeled.write_bytes(plain.read_bytes())

            volume = load_volume(mislabeled)
            self.assertEqual(image.shape, volume.shape)
            np.testing.assert_allclose(image, volume)


class PreprocessingParityTest(unittest.TestCase):
    """推理预处理必须与主文件夹 ``models/authenticity_inference.py`` 完全一致。"""

    def setUp(self) -> None:
        if REFERENCE_MODULE is None or not REFERENCE_MODULE.is_file():
            self.skipTest("main project reference implementation not available")
        if PUBLIC_VOLUME is None:
            self.skipTest("no public NIfTI volume available for parity check")
        if not _need_torch():
            self.skipTest("torch/timm not installed")
        if str(MAIN_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(MAIN_PROJECT_ROOT))

    def test_volume_to_slices_matches_reference_bit_for_bit(self) -> None:
        from models.authenticity_inference import volume_to_slices as reference

        from task1.preprocess import volume_to_slices

        expected = reference(PUBLIC_VOLUME, 16, 224)
        actual = volume_to_slices(PUBLIC_VOLUME, 16, 224)
        self.assertEqual(expected.shape, actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


class ScorerParityTest(unittest.TestCase):
    """任务一的打分函数必须与主文件夹推理脚本给出同一个概率。"""

    def setUp(self) -> None:
        if not WEIGHTS or PUBLIC_VOLUME is None or REFERENCE_MODULE is None:
            self.skipTest("weights or public volume missing")
        if not _need_torch():
            self.skipTest("torch/timm not installed")
        if str(MAIN_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(MAIN_PROJECT_ROOT))

    def test_score_matches_reference_predictor(self) -> None:
        from models.authenticity_inference import AuthenticityPredictor

        from task1.config import Task1Config
        from task1.scorer import AuthenticityScorer

        weight = WEIGHTS[0]
        reference = AuthenticityPredictor(checkpoint=weight, device="cpu")
        scorer = AuthenticityScorer(Task1Config(weights=(weight,), device="cpu"))
        scorer.load()

        expected = reference.score_volume(PUBLIC_VOLUME)
        actual = scorer.score_path(PUBLIC_VOLUME)
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(float(expected), float(actual), places=6)


class Goal1ContractTest(unittest.TestCase):
    """Goal1 任务契约：概率有限、在 [0,1] 内，失败时明确降级。"""

    def setUp(self) -> None:
        if not WEIGHTS:
            self.skipTest("weights missing")
        if not _need_torch():
            self.skipTest("torch/timm not installed")

    @staticmethod
    def _context(seed: int = 0):
        from data.structures import Series, Study
        from pipeline.context import PipelineContext

        rng = np.random.default_rng(seed)
        volume = rng.normal(1.0, 0.25, (32, 64, 64)).astype(np.float32)
        series = Series(
            series_uid="T1CE",
            modality="T1 enhanced",
            image=volume,
            affine=np.eye(4),
            source_path=Path("T1CE.nii.gz"),
        )
        return PipelineContext(study=Study("ACC001", (series,)))

    def test_predict_returns_probability_from_model(self) -> None:
        from task1.config import Task1Config
        from task1.goal1 import NotHumanBodyTask

        task = NotHumanBodyTask(
            Task1Config(weights=(WEIGHTS[0],), device="cpu", strict=True)
        )
        task.load_model()
        context = self._context()
        result = task.predict(context)

        self.assertGreaterEqual(result.not_human_probability, 0.0)
        self.assertLessEqual(result.not_human_probability, 1.0)
        self.assertTrue(np.isfinite(result.not_human_probability))
        self.assertEqual("model", context.diagnostics["goal1"]["source"])

    def test_missing_weights_fall_back_outside_strict_mode(self) -> None:
        from task1.config import Task1Config
        from task1.goal1 import NotHumanBodyTask

        task = NotHumanBodyTask(
            Task1Config(
                weights=(TASK1_ROOT / "weights" / "does-not-exist.pt",),
                device="cpu",
                fallback_probability=0.25,
            )
        )
        task.load_model()
        context = self._context()
        result = task.predict(context)

        self.assertAlmostEqual(0.25, result.not_human_probability)
        self.assertTrue(context.warnings)
        self.assertEqual("fallback", context.diagnostics["goal1"]["source"])

    def test_missing_weights_raise_in_strict_mode(self) -> None:
        from task1.config import Task1Config
        from task1.goal1 import NotHumanBodyTask

        task = NotHumanBodyTask(
            Task1Config(
                weights=(TASK1_ROOT / "weights" / "does-not-exist.pt",),
                device="cpu",
                strict=True,
            )
        )
        with self.assertRaises(RuntimeError):
            task.load_model()


class TrainingSmokeTest(unittest.TestCase):
    """从零训练的最小闭环：赛方布局 -> train -> best.pt -> 推理可加载。"""

    def setUp(self) -> None:
        if not _need_torch():
            self.skipTest("torch/timm not installed")

    def test_train_from_scratch_and_load_checkpoint(self) -> None:
        from task1.config import Task1Config
        from task1.scorer import AuthenticityScorer
        from task1.train import main as train_main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "training"
            out_dir = root / "run"
            log_dir = root / "logs"
            make_official_tree(data_root)

            code = train_main([
                "--data-root", str(data_root),
                "--out-dir", str(out_dir),
                "--log-dir", str(log_dir),
                "--epochs", "1",
                "--batch-size", "2",
                "--slices-per-case", "4",
                "--image-size", "64",
                "--num-workers", "0",
                "--max-steps", "1",
                "--smoke-test",
                "--device", "cpu",
                "--val-fraction", "0.3",
            ])
            self.assertEqual(0, code)

            best = out_dir / "best.pt"
            self.assertTrue(best.is_file())
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual("scratch", summary["pretrained_from"])
            self.assertGreater(summary["data_summary"]["positives"], 0)

            log_lines = [
                json.loads(line)
                for line in (log_dir / "training.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(log_lines)
            for field in ("timestamp", "epoch", "step", "phase", "mode", "data_source"):
                self.assertIn(field, log_lines[0])

            scorer = AuthenticityScorer(Task1Config(weights=(best,), device="cpu"))
            scorer.load()
            record = next(
                line for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            probability = scorer.score_path(json.loads(record)["path"])
            self.assertIsNotNone(probability)
            self.assertGreaterEqual(float(probability), 0.0)
            self.assertLessEqual(float(probability), 1.0)


class PipelineIntegrationTest(unittest.TestCase):
    """真实 Goal1 + 占位任务组成完整管线，输出目录必须通过比赛校验。"""

    def setUp(self) -> None:
        if not WEIGHTS:
            self.skipTest("weights missing")
        if not _need_torch():
            self.skipTest("torch/timm not installed")

    @staticmethod
    def _make_dataset(root: Path) -> None:
        import nibabel as nib

        rng = np.random.default_rng(7)
        for index, accession in enumerate(("ACC001", "ACC002")):
            for series_uid in ("T1CE", "FLAIR"):
                directory = root / accession / series_uid
                directory.mkdir(parents=True)
                image = rng.normal(100.0 + index * 50.0, 20.0, (24, 48, 48)).astype(np.float32)
                nib.save(
                    nib.Nifti1Image(image, np.eye(4)),
                    str(directory / f"{series_uid}.nii.gz"),
                )

    def test_factory_pipeline_writes_valid_prediction(self) -> None:
        from core.config import Settings
        from core.runner import EvaluationJob, EvaluationRunner
        from data.loader import DatasetLoader
        from output.validator import OutputValidator

        from task1.config import Task1Config
        from task1.pipeline_factory import build_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)
            settings = Settings(
                workspace=root / "workspace",
                answer_root=root / "workspace" / "answer",
                log_root=root / "workspace" / "logs",
                callback_url=None,
            )
            pipeline = build_pipeline(
                Task1Config(weights=(WEIGHTS[0],), device="cpu", strict=True)
            )
            output = EvaluationRunner(settings, pipeline=pipeline).run(
                EvaluationJob("task1-request", "task1-evaluation", dataset_path),
                send_callback=False,
            )

            self.assertTrue((output / "duplicate_pairs.jsonl").is_file())
            for accession in ("ACC001", "ACC002"):
                payload = json.loads(
                    (output / accession / "prediction.json").read_text(encoding="utf-8")
                )
                probability = payload["IsNotHumanBodyProb"]
                self.assertIsInstance(probability, float)
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)

            OutputValidator().validate(output, DatasetLoader().load(dataset_path))

    def test_registry_builds_factory_without_pipeline_changes(self) -> None:
        """管线代码不能被改动，因此工厂必须能直接从仓库根目录导入。"""
        from core.registry import build_pipeline as registry_build_pipeline

        from task1.goal1 import NotHumanBodyTask
        from task1.goal2 import GatedStudyTask, SpecialImageTask

        previous = {
            "TASK1_WEIGHTS": os.environ.get("TASK1_WEIGHTS"),
            "TASK1_DEVICE": os.environ.get("TASK1_DEVICE"),
        }
        os.environ["TASK1_WEIGHTS"] = str(WEIGHTS[0])
        os.environ["TASK1_DEVICE"] = "cpu"
        try:
            pipeline = registry_build_pipeline("task1.pipeline_factory:build_pipeline")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        # 任务二必须排在任务链最前面，下游任务才来得及被闸门拦住
        self.assertEqual("goal2_stitched", pipeline.study_tasks[0].context_field)
        bindings = {binding.context_field: binding.task for binding in pipeline.study_tasks}
        self.assertIsInstance(bindings["goal2_stitched"], SpecialImageTask)
        goal1 = bindings["goal1"]
        self.assertEqual("goal1", goal1.name)
        self.assertIsInstance(goal1, GatedStudyTask)
        self.assertIsInstance(goal1.inner, NotHumanBodyTask)


if __name__ == "__main__":
    unittest.main()
