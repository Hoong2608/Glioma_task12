"""排序指标（纯 numpy 实现，不引入额外依赖）。

目标一是排序评测（Partial AUC-PR / Recall@10%FPR），所以训练、验证、
离线打分共用同一套实现，避免「训练时看着好、评测口径不同」的偏差。
"""
from __future__ import annotations

import numpy as np

# NumPy 2.x 用 ``trapezoid`` 取代了 ``trapz``。
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """标准 AP（PR 曲线下面积，阶梯规则）。样本全同类时返回 ``nan``。"""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels.sum()
    if positives == 0 or positives == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / positives
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def partial_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    min_recall: float = 0.5,
) -> float:
    """限制在 recall ∈ [min_recall, 1] 的部分 AUC-PR，并按 (1 - min_recall) 归一化。"""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels.sum()
    if positives == 0 or positives == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / positives
    keep = recall >= min_recall
    if not keep.any():
        return float("nan")
    recall = np.concatenate([[min_recall], recall[keep]])
    precision = np.concatenate([[precision[keep][0]], precision[keep]])
    area = float(_trapezoid(precision, recall))
    return area / max(1e-12, 1.0 - min_recall)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def recall_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.10,
) -> float:
    """假阳性率不超过 ``max_fpr`` 时能达到的最大召回率。"""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    recall = np.cumsum(labels) / positives
    fpr = np.cumsum(1.0 - labels) / negatives
    valid = fpr <= max_fpr
    return float(recall[valid].max()) if valid.any() else 0.0


def metrics_report(
    labels: np.ndarray,
    scores: np.ndarray,
    min_recall: float = 0.5,
    max_fpr: float = 0.10,
) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "average_precision": average_precision(labels, scores),
        "partial_ap": partial_average_precision(labels, scores, min_recall),
        "partial_ap_min_recall": float(min_recall),
        "roc_auc": roc_auc(labels, scores),
        "recall_at_fpr": recall_at_fpr(labels, scores, max_fpr),
        "recall_at_fpr_limit": float(max_fpr),
    }


def clean_report(report: dict[str, float]) -> dict[str, float | int | None]:
    """把 ``nan`` 转成 ``None``，方便写 JSON。"""
    cleaned: dict[str, float | int | None] = {}
    for key, value in report.items():
        if isinstance(value, float) and np.isnan(value):
            cleaned[key] = None
        elif isinstance(value, float):
            cleaned[key] = round(value, 6)
        else:
            cleaned[key] = value
    return cleaned
