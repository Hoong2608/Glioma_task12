"""任务一在管线里的落点：真实的 Goal1（真假人体）任务。

只替换 ``StudyTask``，不碰 HTTP 路由、Writer、Validator 和 callback，
因此输出格式与输出目录结构和原有 Dummy 基线完全一致。
"""
from __future__ import annotations

import logging

from ._bootstrap import ensure_pipeline_root

ensure_pipeline_root()

from tasks.base import StudyTask  # noqa: E402
from tasks.results import Goal1Result  # noqa: E402

from .config import Task1Config  # noqa: E402
from .preprocess import series_types  # noqa: E402

logger = logging.getLogger(__name__)

FALLBACK_WARNING = "task1 goal1 fallback: no readable series"


class NotHumanBodyTask(StudyTask[Goal1Result]):
    """用主文件夹训练的真实性模型预测 ``IsNotHumanBodyProb``。

    输入直接取管线 ``Series.image``（Loader 已读入内存），不重复读盘；
    某条序列打不了分就跳过，全部不可读时输出 ``fallback_probability``，
    因为目标一是排序指标，必须对每一例都给出连续概率。
    """

    name = "goal1"

    def __init__(self, config: Task1Config | None = None) -> None:
        self.config = config or Task1Config.from_env()
        self.scorer = None
        self.load_error: str | None = None

    def load_model(self) -> None:
        from .scorer import AuthenticityScorer

        try:
            scorer = AuthenticityScorer(self.config)
            scorer.load()
        except Exception as exc:  # noqa: BLE001
            self.scorer = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            if self.config.strict:
                raise
            logger.error(
                "task1 goal1: model unavailable, every study falls back to %.3f (%s)",
                self.config.fallback_probability,
                self.load_error,
            )
            return
        self.scorer = scorer
        self.load_error = None
        logger.info("task1 goal1: ready on %s", scorer.device)

    def predict(self, context) -> Goal1Result:
        study = context.study
        types = series_types(study)
        if self.scorer is None:
            context.warnings.append(
                f"{FALLBACK_WARNING} ({self.load_error or 'model not loaded'})"
            )
            context.diagnostics["goal1"] = {
                "source": "fallback",
                "probability": self.config.fallback_probability,
                "load_error": self.load_error,
                "series_types": types,
            }
            return Goal1Result(not_human_probability=self.config.fallback_probability)

        result = self.scorer.score_study(
            (series.series_uid, series.image) for series in study.series
        )
        probability = result.probability
        detail = result.as_dict()
        for item in detail.get("series", []):
            item["series_type"] = types.get(str(item.get("series_uid", "")), "")
        if probability is None:
            probability = self.config.fallback_probability
            detail["source"] = "fallback"
            context.warnings.append(f"{FALLBACK_WARNING} for {study.accession_number}")
        else:
            detail["source"] = "model"
        context.diagnostics["goal1"] = detail
        return Goal1Result(not_human_probability=float(probability))
