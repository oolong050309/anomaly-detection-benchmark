"""算法基类家族。

定义异常检测算法的统一接口，由所有 21 个 Detector 子类继承：

- ``BaseDetector``       : 表格无监督算法的通用基类，模板方法模式。
- ``SupervisedDetector`` : 强制 ``fit`` 提供 ``y`` 标签。
- ``TimeSeriesDetector`` : 放宽校验，接受一维序列或 ``(n_windows, window_size)``。
- ``GraphDetector``      : 重写 ``fit`` 接受图对象（PyG ``Data``）。

子类只需实现 ``_fit`` / ``_decision_function`` 两个钩子，
拟合状态、阈值化、输入校验等公共逻辑由基类统一提供。
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
from sklearn.exceptions import NotFittedError


# ---------------------------------------------------------------------------
# BaseDetector
# ---------------------------------------------------------------------------


class BaseDetector(abc.ABC):
    """所有异常检测算法的抽象基类。

    Parameters
    ----------
    contamination : float, default=0.1
        训练集中异常样本的预估比例，用于 ``predict`` 的阈值化；取值 ``(0, 0.5]``。
    random_state : int | None, default=42
        随机种子，子类应在 ``_fit`` 中传递给底层模型确保可复现。

    Attributes
    ----------
    is_fitted_ : bool
        ``fit`` 成功后由基类自动置为 True。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
    ) -> None:
        if not isinstance(contamination, (int, float)):
            raise ValueError(
                f"contamination must be a float, got {type(contamination).__name__}"
            )
        if not (0.0 < float(contamination) <= 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5], got {contamination!r}"
            )
        self.contamination = float(contamination)
        self.random_state = random_state
        self.is_fitted_: bool = False

    # ---- public API (template methods) ----

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> "BaseDetector":
        """拟合异常检测器。

        子类不应重写本方法；统一通过 ``_fit`` 钩子实现自己的训练逻辑。
        """
        X_arr = self._validate_input(X)
        if y is not None:
            y = self._validate_y(y, n_samples=X_arr.shape[0])
        self._fit(X_arr, y, **kwargs)
        self.is_fitted_ = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的异常分数（越大越异常）。"""
        self._check_fitted()
        X_arr = self._validate_input(X)
        scores = self._decision_function(X_arr)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if scores.shape[0] != X_arr.shape[0]:
            raise RuntimeError(
                f"{type(self).__name__}._decision_function returned scores of "
                f"shape {scores.shape}, expected ({X_arr.shape[0]},)"
            )
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """基于 ``contamination`` 比例对 ``decision_function`` 输出阈值化。

        默认实现：取分数 top ``ceil(contamination * n)`` 标 1，其余标 0。
        分数并列于阈值时，所有 ``>= threshold`` 的样本都标 1（保持幂等性）。
        """
        scores = self.decision_function(X)
        n = scores.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        k = int(np.ceil(self.contamination * n))
        if k <= 0:
            return np.zeros(n, dtype=np.int64)
        if k >= n:
            return np.ones(n, dtype=np.int64)
        # 第 k 大的分数 = 降序排列后第 k-1 个，等价于升序的第 (n-k) 个
        threshold = float(np.partition(scores, n - k)[n - k])
        return (scores >= threshold).astype(np.int64)

    # ---- abstract hooks ----

    @abc.abstractmethod
    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        """子类训练钩子。基类已完成输入校验与状态管理。"""

    @abc.abstractmethod
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """子类评分钩子。返回长度等于 X 行数的一维浮点数组。"""

    # ---- helpers ----

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                f"{type(self).__name__} not fitted; call fit() before "
                "decision_function() or predict()."
            )

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """表格场景的默认校验：要求二维 float ndarray，无 NaN/Inf。"""
        if X is None:
            raise ValueError("X must not be None")
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be a 2D array, got shape {X_arr.shape}"
            )
        if X_arr.size == 0:
            raise ValueError("X must contain at least one sample")
        if not np.issubdtype(X_arr.dtype, np.number):
            raise ValueError(
                f"X must have a numeric dtype, got {X_arr.dtype}"
            )
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or Inf values")
        return X_arr.astype(np.float64, copy=False)

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y_arr = np.asarray(y).ravel()
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"y has {y_arr.shape[0]} samples but X has {n_samples}"
            )
        return y_arr


# ---------------------------------------------------------------------------
# SupervisedDetector
# ---------------------------------------------------------------------------


class SupervisedDetector(BaseDetector):
    """有监督异常检测器基类。

    覆盖 ``fit`` 强制 ``y`` 必须提供，且必须包含两个类别（0/1）。
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> "SupervisedDetector":
        if y is None:
            raise ValueError(
                f"{type(self).__name__} is supervised; fit() requires y"
            )
        y_arr = np.asarray(y).ravel()
        unique = np.unique(y_arr)
        # 必须是 {0, 1} 子集
        if not set(unique.tolist()).issubset({0, 1}):
            raise ValueError(
                f"y must contain only 0/1 labels, got unique values {unique.tolist()}"
            )
        if unique.size < 2:
            raise ValueError(
                f"{type(self).__name__} requires both classes (0 and 1) in y, "
                f"got only {unique.tolist()}"
            )
        return super().fit(X, y_arr, **kwargs)


# ---------------------------------------------------------------------------
# TimeSeriesDetector
# ---------------------------------------------------------------------------


class TimeSeriesDetector(BaseDetector):
    """时序专属基类。

    输入约定：
    - 一维 ``(T,)`` 单序列；
    - 或二维 ``(n_windows, window_size)`` 窗口化数据，由上游 adapter 切窗。
    """

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        if X is None:
            raise ValueError("X must not be None")
        X_arr = np.asarray(X)
        if X_arr.ndim not in (1, 2):
            raise ValueError(
                f"Time-series X must be 1D or 2D, got shape {X_arr.shape}"
            )
        if X_arr.size == 0:
            raise ValueError("X must contain at least one timestep")
        if not np.issubdtype(X_arr.dtype, np.number):
            raise ValueError(
                f"X must have a numeric dtype, got {X_arr.dtype}"
            )
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or Inf values")
        return X_arr.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# GraphDetector
# ---------------------------------------------------------------------------


class GraphDetector(BaseDetector):
    """图专属基类。

    重写 ``fit`` / ``decision_function`` 签名，第一个参数是图对象（PyG Data 或 DGL）。
    ``predict`` 在图层语义不直接可用：调用方应根据 train/val/test mask
    自行处理节点级标签输出。
    """

    # ---- override public API to accept a graph object ----

    def fit(self, graph: Any, **kwargs: Any) -> "GraphDetector":  # type: ignore[override]
        graph = self._validate_graph(graph)
        self._fit(graph, None, **kwargs)
        self.is_fitted_ = True
        return self

    def decision_function(self, graph: Any) -> np.ndarray:  # type: ignore[override]
        self._check_fitted()
        graph = self._validate_graph(graph)
        scores = self._decision_function(graph)
        return np.asarray(scores, dtype=np.float64).ravel()

    def predict(self, graph: Any) -> np.ndarray:  # type: ignore[override]
        scores = self.decision_function(graph)
        n = scores.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        k = int(np.ceil(self.contamination * n))
        if k <= 0:
            return np.zeros(n, dtype=np.int64)
        if k >= n:
            return np.ones(n, dtype=np.int64)
        threshold = float(np.partition(scores, n - k)[n - k])
        return (scores >= threshold).astype(np.int64)

    # ---- helpers ----

    def _validate_graph(self, graph: Any) -> Any:
        """校验图对象是否可用；若为 DGL，则尝试转换为 PyG。"""
        if graph is None:
            raise ValueError("graph must not be None")
        # PyG Data 对象
        if hasattr(graph, "edge_index") and hasattr(graph, "x"):
            if graph.edge_index is None or graph.x is None:
                raise ValueError(
                    "graph.edge_index and graph.x must both be present"
                )
            return graph
        # DGL 图：尝试转 PyG
        if hasattr(graph, "edges") and hasattr(graph, "ndata"):
            return self._dgl_to_pyg(graph)
        raise ValueError(
            f"Unsupported graph type {type(graph).__name__}; expected "
            "torch_geometric.data.Data or dgl.DGLGraph"
        )

    @staticmethod
    def _dgl_to_pyg(g: Any) -> Any:
        """把 DGL 图转换为 PyG Data 对象。"""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "torch_geometric is required to convert DGL graphs"
            ) from e

        src, dst = g.edges()
        edge_index = torch.stack(
            [src.long(), dst.long()], dim=0
        )
        x = g.ndata.get("feature", g.ndata.get("feat", None))
        if x is None:
            raise ValueError("DGL graph has no 'feature' or 'feat' in ndata")
        y = g.ndata.get("label", None)
        data = Data(x=x.float(), edge_index=edge_index)
        if y is not None:
            data.y = y.long()
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            if mask_name in g.ndata:
                setattr(data, mask_name, g.ndata[mask_name].bool())
        return data
