"""训练集污染注入工具。

Exp-2 需要比较算法在训练集被污染时的鲁棒性。本模块分别实现无监督场景的
异常样本保留比例控制，以及有监督场景的标签对称翻转。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def contaminate_unsupervised(
    X_train: Any,
    y_train: Any,
    contamination_rate: float,
    seed: int = 42,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """为无监督算法构造指定污染率的训练集。

    `contamination_rate=0.0` 表示只保留正常样本。其他值会从异常样本中抽样，
    让输出训练集中的异常比例尽量接近目标污染率。
    """

    if not 0 <= contamination_rate < 1:
        raise ValueError("contamination_rate must be in [0, 1)")
    X = np.asarray(X_train)
    y = np.asarray(y_train).astype(int).reshape(-1)
    rng = np.random.default_rng(seed)

    normal_idx = np.flatnonzero(y == 0)
    anomaly_idx = np.flatnonzero(y == 1)
    if contamination_rate == 0 or len(anomaly_idx) == 0:
        selected = normal_idx
    else:
        target_anomalies = int(round(contamination_rate * len(normal_idx) / (1 - contamination_rate)))
        target_anomalies = min(max(target_anomalies, 1), len(anomaly_idx))
        selected_anomalies = rng.choice(anomaly_idx, size=target_anomalies, replace=False)
        selected = np.concatenate([normal_idx, selected_anomalies])
    rng.shuffle(selected)
    meta = {
        "mode": "unsupervised_keep_anomaly_ratio",
        "contamination_rate": float(contamination_rate),
        "seed": int(seed),
        "n_input": int(len(y)),
        "n_output": int(len(selected)),
        "n_anomalies_output": int(np.sum(y[selected] == 1)),
    }
    return X[selected], y[selected], meta


def contaminate_supervised_labels(
    y_train: Any,
    flip_rate: float,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """对监督学习标签做 0/1 对称翻转。"""

    if not 0 <= flip_rate <= 1:
        raise ValueError("flip_rate must be in [0, 1]")
    y = np.asarray(y_train).astype(int).reshape(-1).copy()
    rng = np.random.default_rng(seed)
    n_flip = int(round(len(y) * flip_rate))
    flip_idx = rng.choice(np.arange(len(y)), size=n_flip, replace=False) if n_flip else np.array([], dtype=int)
    y[flip_idx] = 1 - y[flip_idx]
    meta = {
        "mode": "supervised_symmetric_label_flip",
        "flip_rate": float(flip_rate),
        "seed": int(seed),
        "n_input": int(len(y)),
        "n_flipped": int(n_flip),
    }
    return y, meta


def contaminate_supervised(
    X_train: Any,
    y_train: Any,
    flip_rate: float,
    seed: int = 42,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """保持特征不变，只返回翻转后的监督标签。"""

    y_flipped, meta = contaminate_supervised_labels(y_train, flip_rate, seed)
    return X_train, y_flipped, meta
