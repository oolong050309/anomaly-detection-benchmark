"""数据适配器的共享工具。

本模块放置跨 ADBench、TSB-AD、GADBench 都会用到的轻量工具：
路径解析、统一返回对象、训练集标准化、分层划分和 EDA 摘要统计。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


DEFAULT_SEED = int(os.getenv("AD_SEED", "42"))
DEFAULT_TEST_SIZE = float(os.getenv("AD_TEST_SIZE", "0.3"))


@dataclass
class DatasetBundle:
    """所有数据适配器共享的统一返回对象。

    只要数据能表示成样本矩阵，就提供 `X_train`、`X_test`、`y_train`、
    `y_test` 四元组。图数据还会把原始图对象、mask 等附加信息放进
    `extras`，避免破坏统一接口。
    """

    name: str
    modality: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    extras: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[Any, Any, Any, Any]:
        """返回算法侧最常用的四元组。"""

        return self.X_train, self.X_test, self.y_train, self.y_test


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_root(data_root: Optional[os.PathLike[str] | str] = None) -> Path:
    """解析数据根目录：优先参数，其次环境变量，最后使用本地 `data/`。"""

    if data_root:
        return Path(data_root).expanduser()
    env_root = os.getenv("AD_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return get_project_root() / "data"


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """只用训练集统计量标准化，避免测试集信息泄漏。"""

    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    return (X_train - mean) / scale, (X_test - mean) / scale, {
        "standardization": "zscore_train_only",
        "mean_shape": tuple(mean.shape),
        "scale_shape": tuple(scale.shape),
    }


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """无第三方依赖的二分类分层训练/测试划分。"""

    y = np.asarray(y).astype(int).reshape(-1)
    X = np.asarray(X)
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} != {len(y)}")

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        if len(idx) > 1:
            n_test = min(max(n_test, 1), len(idx) - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def summarize_array(X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """生成可 JSON 序列化的数组型数据 EDA 摘要。"""

    arr = np.asarray(X)
    summary: Dict[str, Any] = {
        "n_samples": int(arr.shape[0]) if arr.ndim else 0,
        "n_features": int(arr.shape[1]) if arr.ndim > 1 else 1,
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
    }
    if arr.size:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            summary.update(
                {
                    "value_min": float(np.min(finite)),
                    "value_max": float(np.max(finite)),
                    "value_mean": float(np.mean(finite)),
                    "value_std": float(np.std(finite)),
                }
            )
    if y is not None:
        labels = np.asarray(y).astype(int).reshape(-1)
        anomalies = int(np.sum(labels == 1))
        summary.update(
            {
                "n_anomalies": anomalies,
                "anomaly_rate": float(anomalies / len(labels)) if len(labels) else 0.0,
            }
        )
    return summary


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None
