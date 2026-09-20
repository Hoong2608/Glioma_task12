"""让 ``task1`` 既能作为管线插件，也能被单独运行。

比赛推理服务从管线仓库根目录启动（``python -m uvicorn app.server:app``），
此时 ``pipeline``、``tasks``、``data`` 已经在 ``sys.path`` 上。
``task1`` 现在就在仓库根目录内，因此正常情况下什么都不用做；
但为了兼容「直接执行脚本」或「把 task1 放到仓库外层」的用法，
这里仍会把管线根目录补进 ``sys.path``。
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _candidates() -> tuple[Path, ...]:
    parent = PACKAGE_ROOT.parent
    return (
        parent,                                  # task1 位于仓库根目录内（当前布局）
        parent / "Glioma_recognition-main",      # task1 与仓库同级（旧布局）
        parent.parent,
    )


def find_pipeline_root() -> Path | None:
    """定位包含 ``pipeline/inference.py`` 的管线仓库根目录。"""
    for candidate in _candidates():
        if (candidate / "pipeline" / "inference.py").is_file():
            return candidate
    return None


def ensure_pipeline_root() -> Path | None:
    """把管线根目录插入 ``sys.path``，返回该目录（找不到时返回 ``None``）。"""
    root = find_pipeline_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
