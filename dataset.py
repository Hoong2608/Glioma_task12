"""赛方训练数据的发现、标签构建与切片数据集。

按《公共数据集格式说明》，训练数据根目录下有一个 ``annotation`` 目录，
里面除三个特殊目录外的子目录都是**正常影像**（目标一负类）：

```text
/2026aicompetition/datasets/training/
├── annotation/
│   ├── fake/          # 目标一阳性：假人体 / 非人体影像
│   ├── Composition/   # 目标二阳性：拼接影像（目标一默认排除）
│   ├── duplicate/     # 目标二：重复影像（含金标准，目标一默认排除）
│   └── <其它目录>/     # 正常影像 = 目标一负类
└── <其它目录>/         # 正常影像 = 目标一负类
```

目录名按大小写不敏感的前缀匹配（``fake`` / ``composition`` / ``duplicate``），
文件名含 ``mask`` / ``seg`` / ``label`` / ``roi`` 的 NIfTI 视为标注而非输入影像，
与管线 Loader 的过滤规则一致。

先体检数据（不需要 torch）::

    python -m task1.dataset --data-root /2026aicompetition/datasets/training \
        --out-dir /2026aicompetition/workspace/task1_runs
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .preprocess import resize2d, volume_to_slices

NIFTI_SUFFIXES = (".nii", ".nii.gz")
MASK_HINTS = ("mask", "seg", "label", "roi")
SPECIAL_KINDS = ("fake", "composition", "duplicate")
SPECIAL_POLICIES = ("exclude", "negative")


# --------------------------------------------------------------------------
# 数据发现
# --------------------------------------------------------------------------
@dataclass
class Record:
    """一个训练样本 = 一条序列体数据。"""

    path: str
    accession: str
    series_uid: str
    label: int
    group: str
    split: str = "train"

    def as_dict(self) -> dict:
        return asdict(self)


def _stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def is_image(path: Path) -> bool:
    """NIfTI 且不是标注/掩膜。"""
    if not is_nifti(path):
        return False
    name = path.name.lower()
    return not any(hint in name for hint in MASK_HINTS)


def iter_images(root: Path, skip_dirs: Sequence[Path] = ()) -> list[Path]:
    """递归收集输入影像，跳过 ``skip_dirs`` 子树（以及其中的全部内容）。"""
    if not root.is_dir():
        return []
    resolved_skips = [path.resolve() for path in skip_dirs if path is not None]
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_image(path):
            continue
        parent = path.resolve()
        if any(_is_within(parent, skip) for skip in resolved_skips):
            continue
        found.append(path)
    return found


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def special_dir(base: Path, kind: str) -> Path | None:
    """在 ``base`` 下按大小写不敏感前缀找 ``fake`` / ``composition`` / ``duplicate``。"""
    if not base.is_dir():
        return None
    for child in sorted(path for path in base.iterdir() if path.is_dir()):
        if child.name.strip().lower().startswith(kind):
            return child
    return None


def find_special_root(data_root: Path, kinds: Sequence[str] = SPECIAL_KINDS) -> Path | None:
    """定位含任一特殊目录（``fake`` / ``composition`` / ``duplicate``）的标注根目录。"""
    if not data_root.is_dir():
        return None
    candidates = [data_root, data_root / "annotation"]
    candidates += sorted(path for path in data_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if any(special_dir(candidate, kind) is not None for kind in kinds):
            return candidate
    for child in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for grandchild in sorted(path for path in child.iterdir() if path.is_dir()):
            if any(special_dir(grandchild, kind) is not None for kind in kinds):
                return grandchild
    return None


def find_annotation_root(data_root: Path) -> Path | None:
    """定位含 ``fake`` 目录的标注根目录（任务一训练用）。"""
    return find_special_root(data_root, kinds=("fake",))


def _identity(path: Path, base: Path) -> tuple[str, str]:
    """``(accession, series_uid)``：一级目录为检查号，二级目录为序列号。"""
    try:
        relative = path.relative_to(base)
    except ValueError:
        relative = Path(path.name)
    parts = relative.parts
    if len(parts) >= 3:
        return parts[0], parts[-2]
    if len(parts) == 2:
        return parts[0], _stem(path)
    return _stem(path), _stem(path)


def discover_records(
    data_root: Path,
    *,
    annotation_root: Path | None = None,
    composition: str = "exclude",
    duplicate: str = "exclude",
    limit: int = 0,
) -> list[Record]:
    """扫描赛方数据，返回带标签的样本列表。

    ``composition`` / ``duplicate`` 取 ``exclude``（默认，目标一不使用这两类
    影像）或 ``negative``（当作真人体负类）。
    """
    if composition not in SPECIAL_POLICIES or duplicate not in SPECIAL_POLICIES:
        raise ValueError(f"policy must be one of {SPECIAL_POLICIES}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")

    root = data_root.resolve()
    ann = (annotation_root or find_annotation_root(root))
    if ann is not None:
        ann = ann.resolve()
        if not _is_within(ann, root) and ann != root:
            root = ann.parent

    fake_dir = special_dir(ann, "fake") if ann else None
    composition_dir = special_dir(ann, "composition") if ann else None
    duplicate_dir = special_dir(ann, "duplicate") if ann else None
    if fake_dir is None:
        raise FileNotFoundError(
            f"cannot find the 'fake' annotation folder under {root}; "
            "pass --annotation-root explicitly if the layout differs"
        )

    records: list[Record] = []

    for path in iter_images(fake_dir):
        accession, series_uid = _identity(path, fake_dir)
        records.append(Record(str(path), accession, series_uid, 1, "fake"))

    negative_roots: list[tuple[Path, tuple[Path, ...]]] = []
    if ann is not None:
        negative_roots.append(
            (ann, tuple(path for path in (fake_dir, composition_dir, duplicate_dir) if path))
        )
    negative_roots.append(
        (root, tuple(path for path in (ann, fake_dir, composition_dir, duplicate_dir) if path))
    )

    seen: set[str] = set()
    for base, skips in negative_roots:
        for path in iter_images(base, skip_dirs=skips):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            accession, series_uid = _identity(path, base)
            records.append(Record(str(path), accession, series_uid, 0, "normal"))

    for kind, directory, policy in (
        ("composition", composition_dir, composition),
        ("duplicate", duplicate_dir, duplicate),
    ):
        if directory is None or policy != "negative":
            continue
        for path in iter_images(directory):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            accession, series_uid = _identity(path, directory)
            records.append(Record(str(path), accession, series_uid, 0, kind))

    records.sort(key=lambda item: (item.label, item.group, item.path))
    if limit:
        from collections import Counter

        kept: list[Record] = []
        quota = Counter()
        for record in records:
            if quota[record.label] < limit:
                kept.append(record)
                quota[record.label] += 1
        records = kept
    return records


def assign_splits(
    records: list[Record],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[Record]:
    """按**检查号**分层切分 train/val/test，同一检查不会跨集合。"""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("invalid val/test fractions")
    rng = random.Random(seed)
    cases: dict[int, dict[str, list[Record]]] = {}
    for record in records:
        cases.setdefault(record.label, {}).setdefault(record.accession, []).append(record)

    for label, grouped in cases.items():
        accessions = sorted(grouped)
        rng.shuffle(accessions)
        total = len(accessions)
        n_test = int(round(total * test_fraction)) if test_fraction else 0
        n_val = int(round(total * val_fraction)) if val_fraction else 0
        if total > 1:
            n_val = max(1, min(n_val, total - 1 - n_test))
            n_test = max(0, min(n_test, total - 1 - n_val))
        else:
            n_val = n_test = 0
        for index, accession in enumerate(accessions):
            if index < n_test:
                split = "test"
            elif index < n_test + n_val:
                split = "val"
            else:
                split = "train"
            for record in grouped[accession]:
                record.split = split
    return records


def summarize(records: Iterable[Record]) -> dict:
    records = list(records)
    by_split: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        subset = [record for record in records if record.split == split]
        if not subset:
            continue
        by_split[split] = {
            "volumes": len(subset),
            "cases": len({record.accession for record in subset}),
            "positives": sum(record.label for record in subset),
        }
    groups: dict[str, int] = {}
    for record in records:
        groups[record.group] = groups.get(record.group, 0) + 1
    return {
        "volumes": len(records),
        "cases": len({record.accession for record in records}),
        "positives": sum(record.label for record in records),
        "negatives": sum(1 for record in records if record.label == 0),
        "by_group": groups,
        "by_split": by_split,
    }


def write_manifest(records: Sequence[Record], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: Path) -> list[Record]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(Record(**json.loads(line)))
    return records


# --------------------------------------------------------------------------
# 数据集与增广（与主文件夹 datasets/authenticity.py 的增广一致）
# --------------------------------------------------------------------------
@dataclass
class SliceConfig:
    k: int = 16
    size: int = 224
    min_std: float = 0.05
    train: bool = False
    slice_mode: str = "random"
    flip_prob: float = 0.5
    intensity_jitter: float = 0.1
    bias_prob: float = 0.3
    bias_strength: float = 0.25
    blur_prob: float = 0.2
    noise_prob: float = 0.3
    noise_sigma: float = 0.05
    gamma_prob: float = 0.2


def _box_blur(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    acc = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]
    return (acc / 9.0).astype(np.float32)


def _low_frequency_field(shape: tuple[int, int], rng: random.Random, cells: int = 4) -> np.ndarray:
    small = np.array(
        [[rng.uniform(0.7, 1.3) for _ in range(cells)] for _ in range(cells)],
        dtype=np.float32,
    )
    return resize2d(small, shape[0])


def augment_slices(images: np.ndarray, rng: random.Random, config: SliceConfig) -> np.ndarray:
    if rng.random() < config.flip_prob:
        images = images[:, :, ::-1]
    if rng.random() < config.intensity_jitter:
        images = images * rng.uniform(1 - config.intensity_jitter, 1 + config.intensity_jitter)
    if rng.random() < config.bias_prob:
        field = _low_frequency_field(images.shape[1:], rng)
        images = images * (1.0 + config.bias_strength * (field - 1.0))
    if rng.random() < config.blur_prob:
        images = np.stack([_box_blur(image) for image in images])
    if rng.random() < config.noise_prob:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, config.noise_sigma, images.shape
        )
        images = images + np.asarray(noise, dtype=np.float32)
    if rng.random() < config.gamma_prob:
        gamma = rng.uniform(0.7, 1.4)
        lo, hi = float(images.min()), float(images.max())
        if hi > lo:
            images = np.clip((images - lo) / (hi - lo), 0, 1) ** gamma * (hi - lo) + lo
    return np.ascontiguousarray(images, dtype=np.float32)


class OfficialSlices:
    """一个样本 = 一个检查的一条序列（K 张 2.5D 切片）。"""

    def __init__(
        self,
        records: Sequence[Record],
        split: str,
        config: SliceConfig,
        seed: int = 0,
    ) -> None:
        self.records = [record for record in records if record.split == split]
        self.split = split
        self.config = config
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> np.ndarray:
        return np.asarray([record.label for record in self.records], dtype=np.float32)

    def accessions(self) -> list[str]:
        return [record.accession for record in self.records]

    def __getitem__(self, index: int) -> dict:
        import torch

        record = self.records[index]
        config = self.config
        rng = random.Random(f"{self.seed}:{self.split}:{self.epoch}:{index}")
        mode = config.slice_mode
        if not config.train and mode == "random":
            mode = "uniform"
        augment = None
        if config.train:
            augment = lambda images, generator: augment_slices(images, generator, config)  # noqa: E731
        slices = volume_to_slices(
            record.path,
            config.k,
            config.size,
            config.min_std,
            mode=mode,
            rng=rng if mode != "uniform" else None,
            augment=augment,
        )
        return {
            "slices": torch.from_numpy(slices),
            "label": float(record.label),
            "accession": record.accession,
            "series_uid": record.series_uid,
            "group": record.group,
            "path": record.path,
        }


def collate(batch: list[dict]) -> dict:
    import torch

    return {
        "slices": torch.stack([item["slices"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "accession": [item["accession"] for item in batch],
        "series_uid": [item["series_uid"] for item in batch],
        "group": [item["group"] for item in batch],
        "path": [item["path"] for item in batch],
    }


# --------------------------------------------------------------------------
# 命令行：只做数据体检，不训练
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect official competition data")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="把 manifest.jsonl 写到这里（默认只打印）")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--composition", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--duplicate", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--limit-per-class", type=int, default=0)
    args = parser.parse_args(argv)

    annotation_root = args.annotation_root or find_annotation_root(args.data_root)
    records = discover_records(
        args.data_root,
        annotation_root=annotation_root,
        composition=args.composition,
        duplicate=args.duplicate,
        limit=args.limit_per_class,
    )
    assign_splits(
        records,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    summary = summarize(records)
    summary["data_root"] = str(args.data_root)
    summary["annotation_root"] = None if annotation_root is None else str(annotation_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_dir:
        path = write_manifest(records, args.out_dir / "manifest.jsonl")
        print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
