"""Isolation Forest 包装。

委托 ``pyod.models.iforest.IForest``。IForest 有随机性，random_state 透传。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseDetector


class IForestDetector(BaseDetector):
    """Isolation Forest 异常检测器。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
        透传给 PyOD 底层的 IForest。
    n_estimators : int, default=100
        森林中树的数量。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        n_estimators: int = 100,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.n_estimators = n_estimators
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        from pyod.models.iforest import IForest

        try:
            self._model = IForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
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
