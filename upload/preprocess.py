"""任务一的体数据预处理（与主文件夹推理脚本保持逐位一致）。

对应主文件夹的 ``models/authenticity_inference.py``：percentile 截断归一化 →
最短轴作为层方向 → 去掉近空层 → uniform 抽 K 层 → 双线性缩放到 224² →
复制为 3 通道。训练侧在 ``datasets/authenticity.py``，两侧必须同步修改。

与主文件夹的唯一差别是入口：这里既接受 NIfTI 路径，也直接接受管线
``Series.image`` 已经在内存中的数组，避免对同一份影像二次读盘。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NIFTI_SUFFIXES = (".nii", ".nii.gz")
DEFAULT_SLICES_PER_CASE = 16
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MIN_STD = 0.05


def _alternate_suffix(path: Path) -> Path | None:
    """``x.nii.gz`` <-> ``x.nii``：赛方数据里偶有后缀与实际压缩格式不符的文件。"""
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return path.with_name(name[: -len(".gz")])
    if name.lower().endswith(".nii"):
        return path.with_name(name + ".gz")
    return None


def load_volume(path: str | Path) -> np.ndarray:
    """读取 NIfTI 为 float32 体数据，折叠多余的 4-D 轴。"""
    import nibabel as nib

    path = Path(path)
    try:
        image = nib.load(str(path))
    except Exception as first_error:  # noqa: BLE001
        alternative = _alternate_suffix(path)
        if alternative is None or not alternative.is_file():
            raise
        logger.warning(
            "task1: %s cannot be read (%s: %s); reading %s instead",
            path,
            type(first_error).__name__,
            first_error,
            alternative,
        )
        image = nib.load(str(alternative))
    try:
        data = np.asarray(image.dataobj, dtype=np.float32)
    finally:
        # 释放 nibabel 持有的文件句柄：长跑时避免句柄泄漏，
        # 也避免 Windows 上后续无法删除/覆盖已读文件。
        uncache = getattr(image, "uncache", None)
        if callable(uncache):
            try:
                uncache()
            except Exception:  # noqa: BLE001 - 释放失败不影响已读出的数据
                pass
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim == 2:
        data = data[:, :, None]
    volume = _sanitize(data)
    if isinstance(volume, np.memmap) or not volume.flags.owndata:
        # nibabel 对未压缩 .nii 默认走内存映射；显式拷成内存数组，
        # 避免文件被长期映射占用（Windows 上会锁住文件，长跑也不该依赖页缓存）。
        volume = np.array(volume, dtype=np.float32, copy=True)
    return volume


def _sanitize(data: np.ndarray) -> np.ndarray:
    """统一成 3-D float32，并把 NaN/Inf 置零（训练数据不含这些值，是空操作）。"""
    array = np.asarray(data, dtype=np.float32)
    while array.ndim > 3:
        array = array[..., 0]
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"volume must be 3-D after collapsing, got {array.shape}")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def to_float_volume(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, (str, Path)):
        return load_volume(source)
    return _sanitize(source)


def normalize_volume(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(volume, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(volume)) if np.isfinite(volume).any() else 0.0
        hi = lo + 1.0
    volume = np.clip(volume, lo, hi)
    std = float(volume.std())
    return (volume - float(volume.mean())) / (std if std > 1e-6 else 1.0)


def slice_axis(volume: np.ndarray) -> int:
    """最短轴作为层方向（对任何来源都适用）。"""
    return int(np.argmin(volume.shape))


def usable_indices(volume: np.ndarray, axis: int, min_std: float = DEFAULT_MIN_STD) -> np.ndarray:
    others = tuple(i for i in range(volume.ndim) if i != axis)
    with np.errstate(invalid="ignore"):
        stds = np.nan_to_num(volume.std(axis=others), nan=0.0)
    keep = np.flatnonzero(stds > min_std)
    if keep.size == 0:
        keep = np.arange(volume.shape[axis])
    return keep


def pick_uniform_positions(count: int, k: int) -> np.ndarray:
    """确定性的 uniform 取样——与训练侧验证模式一致。"""
    if count <= 0:
        return np.zeros(k, dtype=int)
    pos = np.round(np.linspace(0, count - 1, k)).astype(int)
    return np.clip(pos, 0, count - 1)


def sample_positions(count: int, k: int, mode: str = "uniform", rng=None) -> np.ndarray:
    """训练用层采样：``random``（训练）/ ``uniform``（验证）/ ``center``。

    推理走 ``pick_uniform_positions``，因此这里的训练随机性不会影响推理口径。
    """
    if count <= 0:
        return np.zeros(k, dtype=int)
    if mode == "random":
        if rng is None:
            raise ValueError("random sampling requires an rng")
        pos = np.array([rng.randrange(count) for _ in range(k)])
        pos.sort()
    elif mode == "center":
        pos = np.round(np.linspace(0.3, 0.7, k) * (count - 1)).astype(int)
    else:
        pos = pick_uniform_positions(count, k)
    return np.clip(pos, 0, count - 1)


def resize2d(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape
    if (height, width) == (size, size):
        return image.astype(np.float32, copy=False)
    yi = np.linspace(0, height - 1, size)
    xi = np.linspace(0, width - 1, size)
    y0 = np.floor(yi).astype(int)
    x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = (yi - y0)[:, None].astype(np.float32)
    wx = (xi - x0)[None, :].astype(np.float32)
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def volume_to_slices(
    source: str | Path | np.ndarray,
    k: int = DEFAULT_SLICES_PER_CASE,
    size: int = DEFAULT_IMAGE_SIZE,
    min_std: float = DEFAULT_MIN_STD,
    mode: str = "uniform",
    rng=None,
    augment=None,
) -> np.ndarray:
    """体数据 -> ``(K, 3, size, size)`` float32，等价于训练期验证输入。"""
    volume = normalize_volume(to_float_volume(source))
    axis = slice_axis(volume)
    usable = usable_indices(volume, axis, min_std)
    if mode == "uniform" and rng is None:
        positions = usable[pick_uniform_positions(usable.size, k)]
    else:
        positions = usable[sample_positions(usable.size, k, mode, rng)]
    stack = np.moveaxis(np.take(volume, positions, axis=axis), axis, 0)
    images = np.stack([resize2d(stack[i], size) for i in range(stack.shape[0])])
    if augment is not None:
        images = augment(images, rng)
    return np.ascontiguousarray(
        np.repeat(images[:, None, :, :], 3, axis=1),
        dtype=np.float32,
    )


def list_series(case_dir: str | Path) -> list[tuple[str, Path]]:
    """列出一个检查目录下的 ``[(series_uid, nifti_path), ...]``。

    主布局是比赛格式 ``<case>/<SeriesUid>/<SeriesUid>.nii.gz``；
    平铺的 ``*.nii(.gz)`` 目录作为兜底，方便直接跑公开代理数据。
    """
    case_dir = Path(case_dir)
    items: list[tuple[str, Path]] = []
    if not case_dir.is_dir():
        return items

    for series_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
        candidate = series_dir / f"{series_dir.name}.nii.gz"
        if candidate.is_file():
            items.append((series_dir.name, candidate))
            continue
        nested = sorted(
            path for path in series_dir.iterdir()
            if path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES)
        )
        if nested:
            items.append((series_dir.name, nested[0]))
    if items:
        return items

    for path in sorted(case_dir.iterdir()):
        if path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES):
            stem = path.name
            for suffix in NIFTI_SUFFIXES:
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            items.append((stem, path))
    return items
