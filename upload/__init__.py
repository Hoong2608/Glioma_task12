"""任务一：评审目标一「真人体 vs 伪造影像（假人体/非人体）」。

本包把项目主文件夹（``E:\\111new\\Comp``）里已经训练好的真实性判别模型接入
``Glioma_recognition-main`` 的比赛推理管线，替代原来的 Dummy Goal1，
为每个检查号输出 ``IsNotHumanBodyProb``。

包内不含训练代码，也不在导入时加载 PyTorch；权重与推理链路分别放在
``task1/weights/`` 与 ``task1/scorer.py``。
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
