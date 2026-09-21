"""任务二（重复影像）：全库中间层指纹索引。

流程与赛方规则一致：

1. **先扫全库**：遍历数据集目录下每个检查的每条序列，只抽「中间一层」影像
   （最短轴作为层方向，取 ``n // 2`` 层），算一个确定性指纹（解码后像素的 SHA1）；
2. **再逐例比对**：当前检查的中间层指纹命中库里**其它**检查的指纹时，判定为重复影像。

指纹只取决于解码后的像素数据，因此「同一份影像的拷贝」必然命中，
不同病人的扫描几乎不可能逐像素相同。
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

NIFTI_SUFFIXES = (".nii", ".nii.gz")
MASK_HINTS = ("mask", "seg", "label", "roi")


def _stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def is_image(path: Path) -> bool:
    name = path.name.lower()
    return path.name.lower().endswith(NIFTI_SUFFIXES) and not any(h in name for h in MASK_HINTS)


def slice_axis(shape: Sequence[int]) -> int:
    """最短轴作为层方向（与 task1/preprocess.py 的推理口径一致）。"""
    return int(np.argmin(shape[:3]))


def middle_index(length: int) -> int:
    return max(0, length // 2)


def _fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha1(contiguous.tobytes()).hexdigest()


def middle_slice_from_array(volume: np.ndarray) -> np.ndarray:
    """内存体数据 -> 中间层（与 ``middle_slice_from_path`` 同规则）。"""
    array = np.asarray(volume)
    while array.ndim > 3:
        array = array[..., 0]
    if array.ndim < 2:
        raise ValueError(f"volume must be at least 2-D, got {array.shape}")
    axis = slice_axis(array.shape)
    index = middle_index(array.shape[axis])
    slicer = tuple(index if i == axis else slice(None) for i in range(array.ndim))
    return np.asarray(array[slicer])


def signature_from_array(volume: np.ndarray) -> str:
    return _fingerprint(middle_slice_from_array(volume))


def middle_slice_from_path(path: str | Path) -> np.ndarray:
    """从磁盘只读中间层（不把整个体数据读进内存）。"""
    import nibabel as nib

    image = nib.load(str(path))
    try:
        shape = [int(value) for value in image.shape]
        while len(shape) > 3:
            shape = shape[:-1]
        axis = slice_axis(shape)
        index = middle_index(shape[axis])
        slicer: list[object] = []
        for dim in range(image.ndim):
            if dim >= 3:
                slicer.append(0)          # 4-D 及其以上的尾部轴，与管线 Loader 一致取 0
            elif dim == axis:
                slicer.append(index)
            else:
                slicer.append(slice(None))
        data = np.asanyarray(image.dataobj[tuple(slicer)])
    finally:
        uncache = getattr(image, "uncache", None)
        if callable(uncache):
            try:
                uncache()
            except Exception:  # noqa: BLE001
                pass
    return np.asarray(data)


def signature_from_path(path: str | Path) -> str:
    return _fingerprint(middle_slice_from_path(path))


def select_original_files(root: Path, files: Sequence[Path]) -> list[Path]:
    """按新版管线 Loader 的「原文件」规则筛选 NIfTI（软版本）。

    规则（对应 ``data/loader.py::_select_original_nifti_files``）：

    * 相对根目录只剩 ≤2 段的文件（``<file>`` 或 ``<accession>/<file>``）全部保留；
    * 更深的序列目录里，优先保留主名与目录名完全一致的 ``<目录名>.nii(.gz)``，
      其余同目录文件（如 ``SERIES-A(1).nii.gz``）视为派生副本，不参与比对；
    * 找不到唯一同名原文件时保留该目录下全部文件。管线 Loader 会在更早的
      ``iter_studies`` 阶段直接报错，这里不再重复抛异常，避免自检脚本比主流程更严格。
    """
    selected: list[Path] = []
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in files:
        if len(path.relative_to(root).parts) <= 2:
            selected.append(path)
        else:
            grouped[path.parent].append(path)

    for directory, paths in grouped.items():
        if len(paths) == 1:
            selected.extend(paths)
            continue
        originals = [path for path in paths if _stem(path) == directory.name]
        selected.extend(originals if len(originals) == 1 else paths)
    return sorted(selected)


def group_dataset(
    root: str | Path,
    skip_dirs: Sequence[Path] = (),
) -> dict[str, list[tuple[str, Path]]]:
    """``{accession: [(series_uid, path), ...]}``：与管线 Loader 的发现规则一致。

    只保留「原文件」（见 ``select_original_files``），这样全库中间层指纹索引与
    管线实际加载的影像完全对应——否则序列目录里的派生副本会让重复检测误报。
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {}
    resolved_skips = [Path(path).resolve() for path in skip_dirs if path is not None]
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_image(path):
            continue
        if any(_is_within(path.resolve(), skip) for skip in resolved_skips):
            continue
        candidates.append(path)

    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for path in select_original_files(root, candidates):
        relative = path.relative_to(root)
        accession = relative.parts[0] if len(relative.parts) > 1 else _stem(path)
        if len(relative.parts) > 1 and path.parent != root / relative.parts[0]:
            series_uid = path.parent.name
        else:
            series_uid = _stem(path)
        grouped[accession].append((series_uid, path))
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def dataset_root_of(source_path: str | Path, accession: str, series_uid: str) -> Path:
    """从 ``Series.source_path`` 反推数据集根目录。"""
    path = Path(source_path).expanduser().resolve()
    if path.parent.name == series_uid and path.parent.parent.name == accession:
        return path.parent.parent.parent
    if path.parent.name == accession:
        return path.parent.parent
    return path.parent


@dataclass
class ScanReport:
    root: str | None = None
    studies: int = 0
    volumes: int = 0
    skipped: int = 0
    truncated: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "studies": self.studies,
            "volumes": self.volumes,
            "skipped": self.skipped,
            "truncated": self.truncated,
            "errors": self.errors[:5],
        }


class DuplicateIndex:
    """指纹 -> 检查号 的倒排表。"""

    def __init__(self) -> None:
        self.by_signature: dict[str, set[str]] = defaultdict(set)
        self.signatures: dict[str, set[str]] = defaultdict(set)

    def add(self, accession: str, signatures: Iterable[str]) -> None:
        for signature in signatures:
            self.by_signature[signature].add(accession)
            self.signatures[accession].add(signature)

    def match(self, signatures: Iterable[str], accession: str) -> list[str]:
        """返回与 ``accession`` 中间层重合的**其它**检查号（去重排序）。"""
        matched: set[str] = set()
        for signature in signatures:
            matched.update(self.by_signature.get(signature, ()))
        matched.discard(accession)
        return sorted(matched)

    def __len__(self) -> int:
        return len(self.signatures)

    @classmethod
    def from_array(cls, accession: str, volumes: Iterable[np.ndarray]) -> "DuplicateIndex":
        index = cls()
        index.add(accession, (signature_from_array(volume) for volume in volumes))
        return index

    @classmethod
    def from_dataset(
        cls,
        root: str | Path,
        *,
        max_volumes: int = 0,
        accession_prefixes: Sequence[str] = (),
        skip_dirs: Sequence[Path] = (),
    ) -> tuple["DuplicateIndex", ScanReport]:
        """扫描整个数据集建立索引；``max_volumes > 0`` 时限制扫描条数。"""
        index = cls()
        report = ScanReport(root=str(Path(root).expanduser()))
        grouped = group_dataset(root, skip_dirs=skip_dirs)
        if not grouped:
            return index, report

        budget = max_volumes if max_volumes > 0 else None
        for accession, series in grouped.items():
            if accession_prefixes and not accession.startswith(tuple(accession_prefixes)):
                continue
            signatures: list[str] = []
            for series_uid, path in series:
                if budget is not None and report.volumes >= budget:
                    report.truncated = True
                    break
                try:
                    signatures.append(signature_from_path(path))
                except Exception as exc:  # noqa: BLE001
                    report.skipped += 1
                    if len(report.errors) < 20:
                        report.errors.append(f"{path}: {type(exc).__name__}: {exc}")
                    continue
                report.volumes += 1
            if signatures:
                index.add(accession, signatures)
                report.studies += 1
            if report.truncated:
                break
        return index, report


def iter_pairs(matches: dict[str, list[str]]) -> Iterator[tuple[str, str]]:
    """把「检查号 -> 命中列表」整理成去重后的无序对。"""
    seen: set[tuple[str, str]] = set()
    for accession, matched in matches.items():
        for other in matched:
            key = tuple(sorted((accession, other)))
            if key[0] == key[1] or key in seen:
                continue
            seen.add(key)
            yield key
