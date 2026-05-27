"""LOF（Local Outlier Factor）包装。

委托 ``pyod.models.lof.LOF``。LOF 基于密度比，无随机性。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseDetector


class LOFDetector(BaseDetector):
    """Local Outlier Factor 异常检测器。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
        LOF 是确定性算法，参数保留只为接口一致。
    n_neighbors : int, default=20
        构建 LOF 时使用的近邻数。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        n_neighbors: int = 20,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.n_neighbors = n_neighbors
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        from pyod.models.lof import LOF

        try:
            self._model = LOF(
                n_neighbors=self.n_neighbors,
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
