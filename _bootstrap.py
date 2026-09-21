"""让 ``task1`` 无论放在哪里都能找到管线仓库并正常导入。

两种布局都支持：

* ``<仓库根>/task1/``（task1 在仓库内）；
* ``<上一层>/task1/`` 与 ``<上一层>/<仓库目录>/`` 并列（task1 在仓库外，
  ``<仓库目录>`` 名字可以是 ``Glioma_recognition``、``Glioma_recognition-main``
  或任意包含 ``pipeline/inference.py`` 的目录）。

定位到仓库根后会把它插入 ``sys.path``，这样 ``pipeline`` / ``tasks`` / ``data`` /
``core`` 这些包才能被导入。
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _candidates() -> tuple[Path, ...]:
    parent = PACKAGE_ROOT.parent
    candidates: list[Path] = [
        parent,                                   # task1 与仓库同级（当前布局）
        parent / "Glioma_recognition-main",        # 常见解包目录名
        parent / "Glioma_recognition",             # 服务器上的目录名
        parent.parent,                             # 再上一层兜底
    ]
    if parent.is_dir():
        # 兜底：扫描同级目录里任何含 pipeline/inference.py 的仓库
        try:
            candidates.extend(sorted(path for path in parent.iterdir() if path.is_dir()))
        except OSError:
            pass
    return tuple(candidates)


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
