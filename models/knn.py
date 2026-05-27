"""KNN-AD 包装。

委托 ``pyod.models.knn.KNN``，默认使用 k 近邻最大距离作为分数。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseDetector


class KNNDetector(BaseDetector):
    """基于 k 近邻距离的异常检测。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
        KNN-AD 本身确定性，参数保留只为接口一致。
    n_neighbors : int, default=5
    method : str, default="largest"
        - "largest"：第 k 个近邻的距离（最远近邻法）
        - "mean"：前 k 个近邻距离的均值
        - "median"：前 k 个近邻距离的中位数
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        n_neighbors: int = 5,
        method: str = "largest",
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.method = method
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        from pyod.models.knn import KNN

        try:
            self._model = KNN(
                n_neighbors=self.n_neighbors,
                method=self.method,
                contamination=self.contamination,
                **self._algo_kwargs,
            )
            self._model.fit(X)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.decision_function(X)
