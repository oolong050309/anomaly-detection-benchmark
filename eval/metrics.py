"""评估指标实现：AUC-ROC / AUC-PR / F1@best。

所有函数遵循统一约定：
- ``y_true`` 一维整型 0/1 数组（1=异常）
- ``y_score`` 一维浮点数组，约定越大越异常
- 单类别 ``y_true`` 时返回 NaN 并发 ``UserWarning``，不抛异常
- 输入非法（含 NaN/Inf、长度不一致、非 0/1 标签）时抛 ``ValueError``
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _validate_metric_inputs(
    y_true: Any, y_score: Any
) -> tuple[np.ndarray, np.ndarray]:
    if y_true is None or y_score is None:
        raise ValueError("y_true and y_score must not be None")

    y_true_arr = np.asarray(y_true).ravel()
    y_score_arr = np.asarray(y_score).ravel()

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError(
            f"y_true and y_score length mismatch: "
            f"{y_true_arr.shape[0]} vs {y_score_arr.shape[0]}"
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true is empty")

    # y_true 必须是 0/1 整数
    unique_values = np.unique(y_true_arr)
    if not set(unique_values.tolist()).issubset({0, 1}):
        raise ValueError(
            f"y_true must contain only 0/1 labels, got {unique_values.tolist()}"
        )

    # y_score 必须有限
    y_score_arr = y_score_arr.astype(np.float64, copy=False)
    if not np.isfinite(y_score_arr).all():
        raise ValueError("y_score contains NaN or Inf")

    return y_true_arr.astype(np.int64, copy=False), y_score_arr


# ---------------------------------------------------------------------------
# AUC-ROC / AUC-PR
# ---------------------------------------------------------------------------


def auc_roc(y_true: Any, y_score: Any) -> float:
    """Receiver Operating Characteristic 曲线下面积。

    单类别（y_true 全 0 或全 1）时返回 NaN 并发 UserWarning。
    """
    y_t, y_s = _validate_metric_inputs(y_true, y_score)
    if np.unique(y_t).size < 2:
        warnings.warn(
            "Only one class present in y_true; AUC-ROC is undefined, returning NaN",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(roc_auc_score(y_t, y_s))


def auc_pr(y_true: Any, y_score: Any) -> float:
    """Precision-Recall 曲线下面积（Average Precision）。

    单类别 y_true 时返回 NaN 并发 UserWarning。
    """
    y_t, y_s = _validate_metric_inputs(y_true, y_score)
    if np.unique(y_t).size < 2:
        warnings.warn(
            "Only one class present in y_true; AUC-PR is undefined, returning NaN",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(average_precision_score(y_t, y_s))


# ---------------------------------------------------------------------------
# F1@best
# ---------------------------------------------------------------------------


def f1_at_best(y_true: Any, y_score: Any) -> tuple[float, float]:
    """遍历 ``precision_recall_curve`` 输出的所有阈值，返回 (best_f1, best_threshold)。

    当 ``y_true`` 全为 0 时，没有正样本，返回 ``(0.0, 0.0)``，不抛异常。
    """
    y_t, y_s = _validate_metric_inputs(y_true, y_score)
    if y_t.sum() == 0:
        # 全负样本：任何阈值的召回率都为 0/0，f1=0
        return 0.0, 0.0

    precision, recall, thresholds = precision_recall_curve(y_t, y_s)
    # precision/recall 长度比 thresholds 多 1（最后一个点对应 recall=0 边界）
    p = precision[:-1]
    r = recall[:-1]
    # 避免 (p+r)=0 引发除零
    denom = p + r
    f1 = np.zeros_like(p)
    nz = denom > 0
    f1[nz] = 2.0 * p[nz] * r[nz] / denom[nz]
    if f1.size == 0:
        return 0.0, 0.0
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(thresholds[idx])


# ---------------------------------------------------------------------------
# 一站式聚合
# ---------------------------------------------------------------------------


def evaluate_all(y_true: Any, y_score: Any) -> dict[str, float]:
    """一次返回 AUC-ROC / AUC-PR / F1@best / 最优阈值。

    Returns
    -------
    dict
        keys: ``auc_roc``, ``auc_pr``, ``f1_best``, ``best_threshold``。
    """
    f1_v, thr_v = f1_at_best(y_true, y_score)
    return {
        "auc_roc": auc_roc(y_true, y_score),
        "auc_pr": auc_pr(y_true, y_score),
        "f1_best": f1_v,
        "best_threshold": thr_v,
    }
