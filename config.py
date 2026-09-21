"""任务一的运行配置。

配置全部来自环境变量（前缀 ``TASK1_``），因此比赛容器里不需要改代码，
只要在启动命令或 ``task1/configs/task1.env.example`` 中声明即可。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = PACKAGE_ROOT / "weights"
DEFAULT_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

SERIES_AGGREGATIONS = ("max", "mean")
MODEL_AGGREGATIONS = ("mean", "max")

DEFAULT_DATA_ROOT = Path("/2026aicompetition/datasets/training")
DEFAULT_WORKSPACE = Path("/2026aicompetition/workspace")


def default_data_root() -> Path:
    """赛方训练数据根目录：``TASK1_DATA_ROOT`` 优先。"""
    raw = os.environ.get("TASK1_DATA_ROOT")
    return Path(raw).expanduser() if raw else DEFAULT_DATA_ROOT


def default_run_dir() -> Path:
    """训练产物目录：``TASK1_RUN_DIR`` 优先，默认落在队伍存储下。"""
    raw = os.environ.get("TASK1_RUN_DIR")
    if raw:
        return Path(raw).expanduser()
    workspace = Path(os.environ.get("COMPETITION_WORKSPACE", DEFAULT_WORKSPACE))
    return workspace / "task1_runs"


def default_log_dir(out_dir: Path | None = None) -> Path:
    """比赛日志目录：``TASK1_LOG_DIR`` > ``$COMPETITION_WORKSPACE/logs`` > ``<out-dir>/logs``。"""
    raw = os.environ.get("TASK1_LOG_DIR")
    if raw:
        return Path(raw).expanduser()
    workspace = os.environ.get("COMPETITION_WORKSPACE")
    if workspace:
        return Path(workspace).expanduser() / "logs"
    base = Path(out_dir) if out_dir is not None else PACKAGE_ROOT
    return base / "logs"


def default_weights() -> tuple[Path, ...]:
    """``task1/weights`` 下的全部 ``*.pt``，按文件名排序。"""
    if not DEFAULT_WEIGHTS_DIR.is_dir():
        return ()
    return tuple(sorted(DEFAULT_WEIGHTS_DIR.glob("*.pt")))


def _env_str(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_paths(
    environ: Mapping[str, str],
    name: str,
    default: tuple[Path, ...],
) -> tuple[Path, ...]:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    parts = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    return tuple(Path(item).expanduser() for item in parts)


@dataclass(frozen=True)
class Task1Config:
    """任务一的模型与聚合配置。

    Attributes
    ----------
    weights:
        一个或多个 ``best.pt``。多个权重按 ``model_aggregation`` 集成。
    device / batch_slices:
        推理设备与每次前向的切片数，显存不足时调小 ``batch_slices``。
    slices_per_case / image_size:
        0 表示沿用权重文件里记录的训练值（推荐），非 0 会覆盖。
    series_aggregation:
        一个检查含多条序列时，如何在序列级概率上汇总（``max``/``mean``）。
        训练与评测脚本使用 ``max``：只要有一条可疑序列就足以标记该检查。
    model_aggregation:
        多权重（多随机种子）之间的汇总方式。
    fallback_probability:
        序列全部不可读时的兜底概率；排序指标下必须对每一例都输出连续值。
    strict:
        ``True`` 时权重缺失/加载失败直接抛错（开发与自检用）；
        ``False`` 时记录错误并按 ``fallback_probability`` 降级，保证服务仍能启动。
    """

    weights: tuple[Path, ...] = ()
    device: str = "auto"
    batch_slices: int = 32
    slices_per_case: int = 0
    image_size: int = 0
    min_std: float = 0.05
    series_aggregation: str = "max"
    model_aggregation: str = "mean"
    fallback_probability: float = 0.5
    strict: bool = False

    def __post_init__(self) -> None:
        if self.batch_slices < 1:
            raise ValueError("batch_slices must be >= 1")
        if self.slices_per_case < 0 or self.image_size < 0:
            raise ValueError("slices_per_case and image_size must be >= 0")
        if self.series_aggregation not in SERIES_AGGREGATIONS:
            raise ValueError(
                f"series_aggregation must be one of {SERIES_AGGREGATIONS}"
            )
        if self.model_aggregation not in MODEL_AGGREGATIONS:
            raise ValueError(
                f"model_aggregation must be one of {MODEL_AGGREGATIONS}"
            )
        if not 0.0 <= self.fallback_probability <= 1.0:
            raise ValueError("fallback_probability must be within [0, 1]")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "Task1Config":
        env = os.environ if environ is None else environ
        return cls(
            weights=_env_paths(env, "TASK1_WEIGHTS", default_weights()),
            device=_env_str(env, "TASK1_DEVICE", "auto"),
            batch_slices=_env_int(env, "TASK1_BATCH_SLICES", 32),
            slices_per_case=_env_int(env, "TASK1_SLICES_PER_CASE", 0),
            image_size=_env_int(env, "TASK1_IMAGE_SIZE", 0),
            min_std=_env_float(env, "TASK1_MIN_STD", 0.05),
            series_aggregation=_env_str(env, "TASK1_SERIES_AGGREGATION", "max"),
            model_aggregation=_env_str(env, "TASK1_MODEL_AGGREGATION", "mean"),
            fallback_probability=_env_float(env, "TASK1_FALLBACK_PROB", 0.5),
            strict=_env_bool(env, "TASK1_STRICT", False),
        )

    def resolved_weights(self) -> tuple[Path, ...]:
        """把相对权重路径按当前工作目录展开，便于日志和错误信息展示。"""
        return tuple(path.resolve() for path in self.weights)

    def describe(self) -> dict[str, object]:
        return {
            "weights": [str(path) for path in self.resolved_weights()],
            "device": self.device,
            "batch_slices": self.batch_slices,
            "slices_per_case": self.slices_per_case or "from-checkpoint",
            "image_size": self.image_size or "from-checkpoint",
            "min_std": self.min_std,
            "series_aggregation": self.series_aggregation,
            "model_aggregation": self.model_aggregation,
            "fallback_probability": self.fallback_probability,
            "strict": self.strict,
        }
