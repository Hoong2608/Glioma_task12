"""任务一的推理入口：把训练好的权重变成检查级 ``IsNotHumanBodyProb``。

聚合方式与训练侧验证完全一致：先对一条序列的 K 张切片取 ``K // 2`` 个最高
**logit** 求均值再 sigmoid（不是先 sigmoid 再取 top-k），然后多个权重
（多随机种子）取均值，最后在一个检查的多条序列上取最大值。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import DEFAULT_BACKBONE, Task1Config
from .preprocess import to_float_volume, volume_to_slices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesScore:
    series_uid: str
    probability: float | None
    error: str | None = None


@dataclass(frozen=True)
class StudyScore:
    probability: float | None
    series: tuple[SeriesScore, ...]

    @property
    def scored_series(self) -> int:
        return sum(1 for item in self.series if item.probability is not None)

    def as_dict(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "scored_series": self.scored_series,
            "series_count": len(self.series),
            "series": [
                {
                    "series_uid": item.series_uid,
                    "probability": item.probability,
                    "error": item.error,
                }
                for item in self.series
            ],
        }


@dataclass
class _LoadedModel:
    path: Path
    net: object
    image_size: int
    slices_per_case: int
    val_ap: float | None
    epoch: int | None


class AuthenticityScorer:
    """加载一个或多个 ``best.pt`` 并给体数据/检查打分。"""

    def __init__(self, config: Task1Config | None = None) -> None:
        self.config = config or Task1Config.from_env()
        self._torch = None
        self._device = None
        self._models: list[_LoadedModel] = []
        self.load_errors: list[str] = []
        self.loaded = False

    # -- 加载 ---------------------------------------------------------------
    def load(self) -> None:
        """加载全部可用权重；一个都加载不到时抛 ``RuntimeError``。"""
        import torch  # 局部导入：只有真正推理时才需要 PyTorch

        from .model import AuthenticityNet

        self._torch = torch
        self._device = self._resolve_device(torch, self.config.device)
        self._models = []
        self.load_errors = []

        for path in self.config.weights:
            try:
                state = self._read_checkpoint(torch, path)
                backbone = str(state.get("backbone") or DEFAULT_BACKBONE)
                frequency_branch = bool(state.get("frequency_branch", True))
                image_size = int(state.get("image_size") or self.config.image_size or 224)
                slices_per_case = int(
                    state.get("slices_per_case") or self.config.slices_per_case or 16
                )
                net = AuthenticityNet(
                    backbone=backbone,
                    pretrained=False,  # 权重来自本地文件，推理时不联网
                    frequency_branch=frequency_branch,
                )
                missing, unexpected = net.load_state_dict(state["model"], strict=False)
                if missing or unexpected:
                    raise RuntimeError(
                        "checkpoint does not match the network definition "
                        f"(missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]})"
                    )
                net = net.to(self._device).eval()
            except Exception as exc:  # noqa: BLE001 - 逐个权降级，错误进入日志
                detail = f"{path}: {type(exc).__name__}: {exc}"
                self.load_errors.append(detail)
                logger.error("task1: cannot load checkpoint %s", detail)
                continue
            self._models.append(
                _LoadedModel(
                    path=Path(path),
                    net=net,
                    image_size=image_size,
                    slices_per_case=slices_per_case,
                    val_ap=state.get("val_ap"),
                    epoch=state.get("epoch"),
                )
            )
            logger.info(
                "task1: loaded %s (backbone=%s, image_size=%d, slices=%d, val_ap=%s)",
                path,
                backbone,
                image_size,
                slices_per_case,
                state.get("val_ap"),
            )

        if not self._models:
            raise RuntimeError(
                "no authenticity checkpoint could be loaded: "
                + ("; ".join(self.load_errors) or "TASK1_WEIGHTS is empty")
            )
        self.loaded = True

    @staticmethod
    def _read_checkpoint(torch, path: Path) -> dict:
        if not Path(path).is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.0 没有 weights_only
            state = torch.load(str(path), map_location="cpu")
        if not isinstance(state, dict) or "model" not in state:
            raise RuntimeError(f"unexpected checkpoint layout in {path}")
        return state

    @staticmethod
    def _resolve_device(torch, device: str):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    @property
    def device(self) -> str | None:
        return None if self._device is None else str(self._device)

    @property
    def available(self) -> bool:
        return bool(self._models)

    def describe(self) -> dict[str, object]:
        return {
            "loaded_models": [
                {
                    "path": str(model.path),
                    "image_size": model.image_size,
                    "slices_per_case": model.slices_per_case,
                    "val_ap": model.val_ap,
                    "epoch": model.epoch,
                }
                for model in self._models
            ],
            "load_errors": list(self.load_errors),
            "device": self.device,
            "config": self.config.describe(),
        }

    # -- 打分 ---------------------------------------------------------------
    def _logits(self, handle: _LoadedModel, stack: np.ndarray):
        torch = self._torch
        flat = torch.from_numpy(np.ascontiguousarray(stack))
        outputs = []
        for start in range(0, flat.shape[0], self.config.batch_slices):
            chunk = flat[start:start + self.config.batch_slices].to(
                self._device,
                non_blocking=True,
            )
            outputs.append(handle.net(chunk).detach().float().cpu())
        return torch.cat(outputs)

    def score_array(self, volume: np.ndarray | str | Path) -> float | None:
        """一个 3-D 体数据 -> ``P(非真人体)``；不可打分时返回 ``None``。"""
        if not self._models:
            return None
        torch = self._torch
        array = to_float_volume(volume)
        if array.size == 0:
            return None

        stacks: dict[tuple[int, int], np.ndarray] = {}
        probabilities: list[float] = []
        with torch.inference_mode():
            for handle in self._models:
                key = (handle.slices_per_case, handle.image_size)
                stack = stacks.get(key)
                if stack is None:
                    stack = volume_to_slices(
                        array,
                        handle.slices_per_case,
                        handle.image_size,
                        self.config.min_std,
                    )
                    stacks[key] = stack
                if stack.shape[0] == 0:
                    continue
                logits = self._logits(handle, stack)
                top_k = max(1, logits.shape[0] // 2)
                probabilities.append(
                    float(torch.sigmoid(logits.topk(top_k).values.mean()))
                )
        if not probabilities:
            return None
        return self._aggregate_models(probabilities)

    def _aggregate_models(self, probabilities: Sequence[float]) -> float:
        if self.config.model_aggregation == "max":
            return float(max(probabilities))
        return float(np.mean(probabilities))

    def score_path(self, path: str | Path) -> float | None:
        try:
            return self.score_array(path)
        except Exception as exc:  # noqa: BLE001 - 单条序列失败不中断整个检查
            logger.warning(
                "task1: cannot score volume %s: %s: %s",
                path,
                type(exc).__name__,
                exc,
            )
            return None

    def score_study(self, series: Iterable[tuple[str, object]]) -> StudyScore:
        """``[(series_uid, ndarray|path), ...]`` -> 检查级概率。"""
        scored: list[SeriesScore] = []
        for series_uid, source in series:
            try:
                probability = self.score_array(source)
                error = None if probability is not None else "unreadable volume"
            except Exception as exc:  # noqa: BLE001
                probability = None
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("task1: study series %s failed: %s", series_uid, error)
            scored.append(SeriesScore(series_uid, probability, error))

        values = [item.probability for item in scored if item.probability is not None]
        if not values:
            return StudyScore(None, tuple(scored))
        if self.config.series_aggregation == "mean":
            return StudyScore(float(np.mean(values)), tuple(scored))
        return StudyScore(float(max(values)), tuple(scored))

    def score_case_dir(self, case_dir: str | Path) -> StudyScore:
        """``<case>/<SeriesUid>/<SeriesUid>.nii.gz`` 目录 -> 检查级概率。"""
        from .preprocess import list_series

        return self.score_study(list_series(case_dir))
