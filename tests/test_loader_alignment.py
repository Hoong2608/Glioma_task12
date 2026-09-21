"""任务一/二与新版管线 Loader 的对齐测试。

新版 ``data/loader.py`` 增加两条规则：

1. 序列目录里有多个 NIfTI 时，只读主名与目录名完全一致的 ``<目录名>.nii(.gz)``；
2. 根目录的 ``SeriesType.xlsx`` 会覆盖 ``Series.modality``（序列类型）。

task1 里有两处自己扫盘的地方（任务二全库中间层扫描、离线打分列序列），
必须与规则 1 一致，否则会把派生副本当成检查影像，造成重复影像误报。

在 ``task1`` 的上一层目录执行::

    python -m unittest discover -s task1/tests -t . -v
"""
from __future__ import annotations

import shutil
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

from core.exceptions import InvalidInputError  # noqa: E402
from data.loader import DatasetLoader  # noqa: E402


def _save_image(path: Path, value: float, shape: tuple[int, int, int] = (12, 24, 24)) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(value * 1000))
    image = rng.normal(100.0 + value, 5.0, shape).astype(np.float32)
    nib.save(nib.Nifti1Image(image, np.eye(4)), str(path))


def make_copy_dataset(root: Path) -> None:
    """ACC001 的序列目录里放一个「逐像素等于 ACC002 原文件」的派生副本。"""
    _save_image(root / "ACC001" / "SERIES-A" / "SERIES-A.nii.gz", 1.0)
    _save_image(root / "ACC002" / "SERIES-B" / "SERIES-B.nii.gz", 9.0)
    shutil.copyfile(
        root / "ACC002" / "SERIES-B" / "SERIES-B.nii.gz",
        root / "ACC001" / "SERIES-A" / "SERIES-A(1).nii.gz",
    )


class OriginalFileSelectionTest(unittest.TestCase):
    """派生副本不能进入任务二的重复影像比对。"""

    def test_group_dataset_ignores_derived_copies(self) -> None:
        from task1.middle_slice import DuplicateIndex, group_dataset

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary) / "dataset"
            make_copy_dataset(root)

            loaded = {
                study.accession_number: [series.source_path.name for series in study.series]
                for study in DatasetLoader().iter_studies(root)
            }
            self.assertEqual({"ACC001": ["SERIES-A.nii.gz"], "ACC002": ["SERIES-B.nii.gz"]}, loaded)

            grouped = group_dataset(root)
            self.assertEqual(["SERIES-A.nii.gz"], [p.name for _, p in grouped["ACC001"]])
            self.assertEqual(["SERIES-B.nii.gz"], [p.name for _, p in grouped["ACC002"]])

            index, report = DuplicateIndex.from_dataset(root)
            self.assertEqual(2, report.volumes)          # 不再把派生副本算成第三卷
            for accession in sorted(index.signatures):
                self.assertEqual(
                    [],
                    index.match(index.signatures[accession], accession),
                    f"{accession} 不应命中任何重复影像",
                )

    def test_original_is_kept_when_no_single_match(self) -> None:
        """没有同名原文件时保守保留全部（框架会在 IterStudies 阶段直接报错）。"""
        from task1.middle_slice import group_dataset

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary) / "dataset"
            _save_image(root / "ACC001" / "SERIES" / "first.nii", 1.0)
            _save_image(root / "ACC001" / "SERIES" / "second.nii", 2.0)

            with self.assertRaisesRegex(InvalidInputError, "exactly one original"):
                tuple(DatasetLoader().iter_studies(root))

            grouped = group_dataset(root)
            self.assertEqual(
                ["first.nii", "second.nii"],
                sorted(path.name for _, path in grouped["ACC001"]),
            )


class OfflineSeriesSelectionTest(unittest.TestCase):
    """离线打分选序列时优先原文件（含未压缩 .nii 的情况）。"""

    def test_list_series_prefers_uncompressed_original(self) -> None:
        from task1.preprocess import list_series

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            case = Path(temporary) / "ACC003"
            _save_image(case / "SERIES-C" / "SERIES-C.nii", 3.0)
            _save_image(case / "SERIES-C" / "SERIES-C(1).nii.gz", 7.0)
            _save_image(case / "SERIES-D" / "SERIES-D.nii.gz", 4.0)
            _save_image(case / "SERIES-D" / "SERIES-D-extra.nii.gz", 5.0)

            selected = dict(list_series(case))
            self.assertEqual("SERIES-C.nii", selected["SERIES-C"].name)
            self.assertEqual("SERIES-D.nii.gz", selected["SERIES-D"].name)

            # 与框架 Loader 看到的一致
            loaded = {
                series.series_uid: series.source_path.name
                for series in next(DatasetLoader().iter_studies(Path(temporary))).series
            }
            self.assertEqual(loaded, {uid: path.name for uid, path in selected.items()})


class SeriesTypeTest(unittest.TestCase):
    """SeriesType.xlsx 的类型要能进入任务二诊断信息。"""

    def setUp(self) -> None:
        try:
            import openpyxl  # noqa: F401
        except Exception:  # noqa: BLE001
            self.skipTest("openpyxl 未安装（新版管线 Loader 依赖它）")

    def test_series_type_reaches_goal2_diagnostics(self) -> None:
        from openpyxl import Workbook
        from pipeline.context import PipelineContext

        from task1.goal2 import DIAGNOSTIC_KEY, SpecialImageTask, Task2Config

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary) / "dataset"
            _save_image(root / "ACC001" / "SERIES-A" / "SERIES-A.nii.gz", 1.0)
            _save_image(root / "ACC001" / "SERIES-B" / "SERIES-B.nii.gz", 2.0)

            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["AccessionNumber", "SeriesUid", "SeriesType"])
            sheet.append(["ACC001", "SERIES-A", "T1CE (增强)"])
            workbook.save(root / "SeriesType.xlsx")

            study = next(DatasetLoader().iter_studies(root))
            task = SpecialImageTask(Task2Config(duplicate_scan=True))
            context = PipelineContext(study=study)
            task.predict(context)

            detail = context.diagnostics[DIAGNOSTIC_KEY]
            self.assertEqual("T1CE (增强)", detail["series_types"]["SERIES-A"])
            by_uid = {item["series_uid"]: item for item in detail["stitched"]["series"]}
            self.assertEqual("T1CE (增强)", by_uid["SERIES-A"]["series_type"])
            self.assertEqual("SERIES-B", by_uid["SERIES-B"]["series_type"])


if __name__ == "__main__":
    unittest.main()
