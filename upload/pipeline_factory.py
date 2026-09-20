"""任务一 / 任务二的管线工厂（``COMPETITION_PIPELINE_FACTORY`` 的入口）。

``task1`` 就在仓库根目录内，所以**不需要改动任何管线文件**：把环境变量
``COMPETITION_PIPELINE_FACTORY`` 指向本模块，``core.registry.build_pipeline()``
即可从仓库根目录导入它。

任务链顺序（由本工厂决定，管线只按顺序调用）：

1. ``goal2_stitched``：任务二「拼接 + 重复」检测，命中就写闸门标记；
2. ``goal1``：任务一真实性模型（可被闸门跳过）；
3. ``goal3`` / ``goal5`` / ``goal4``：任务三～五，当前仍是占位任务，
   但已经挂在闸门后面——命中拼接/重复的检查不会执行它们。

启动示例（容器内，工作目录 = 仓库根目录）::

    export COMPETITION_PIPELINE_FACTORY=task1.pipeline_factory:build_pipeline
    export TASK1_WEIGHTS=/2026aicompetition/workspace/task1_runs/best.pt
    export TASK2_GATED_FIELDS=goal3,goal4,goal5
    python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from ._bootstrap import ensure_pipeline_root

ensure_pipeline_root()

from pipeline.inference import InferencePipeline, StudyTaskBinding  # noqa: E402
from tasks.dummy.study_tasks import (  # noqa: E402
    DummyGoal3Task,
    DummyGoal4Task,
    DummyGoal5Task,
)
from tasks.results import Goal1Result  # noqa: E402

from .config import Task1Config  # noqa: E402
from .goal1 import NotHumanBodyTask  # noqa: E402
from .goal2 import (  # noqa: E402
    DuplicatePairRecorder,
    GatedStudyTask,
    SpecialImageTask,
    Task2Config,
)

_DUMMY_GOAL3 = DummyGoal3Task()
_DUMMY_GOAL4 = DummyGoal4Task()
_DUMMY_GOAL5 = DummyGoal5Task()


def build_pipeline(
    config: Task1Config | None = None,
    task2_config: Task2Config | None = None,
) -> InferencePipeline:
    """任务一（真实 Goal1）+ 任务二（拼接/重复闸门）+ 任务三～五占位任务。

    任务三～五的真实模型到位后，把 ``GatedStudyTask`` 里的 Dummy 换成对应 Task 即可；
    闸门逻辑不需要改。
    """
    task1_config = config or Task1Config.from_env()
    task2 = task2_config or Task2Config.from_env()
    fallback = task1_config.fallback_probability
    gated_fields = task2.gated_fields

    return InferencePipeline(
        study_tasks=(
            # 任务二先运行，才能在下游任务之前决定是否上闸
            StudyTaskBinding("goal2_stitched", SpecialImageTask(task2)),
            StudyTaskBinding(
                "goal1",
                GatedStudyTask(
                    NotHumanBodyTask(task1_config),
                    "goal1",
                    lambda context: Goal1Result(not_human_probability=fallback),
                    gated_fields=gated_fields,
                ),
            ),
            StudyTaskBinding(
                "goal3",
                GatedStudyTask(
                    _DUMMY_GOAL3,
                    "goal3",
                    _DUMMY_GOAL3.predict,
                    gated_fields=gated_fields,
                ),
            ),
            StudyTaskBinding(
                "goal5",
                GatedStudyTask(
                    _DUMMY_GOAL5,
                    "goal5",
                    _DUMMY_GOAL5.predict,
                    gated_fields=gated_fields,
                ),
            ),
            StudyTaskBinding(
                "goal4",
                GatedStudyTask(
                    _DUMMY_GOAL4,
                    "goal4",
                    _DUMMY_GOAL4.predict,
                    gated_fields=gated_fields,
                ),
            ),
        ),
        duplicate_task=DuplicatePairRecorder(task2.max_pairs_per_study),
    )
