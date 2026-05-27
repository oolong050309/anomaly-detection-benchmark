"""Deep SVDD 包装（PyOD）。

委托 ``pyod.models.deep_svdd.DeepSVDD``。基于 PyTorch 实现。
``n_features`` 在 ``_fit`` 内自动从 ``X.shape[1]`` 推断。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.autoencoder import _set_torch_seed
from models.base import BaseDetector


class DeepSVDDDetector(BaseDetector):
    """Deep Support Vector Data Description（基于深度边界的异常检测）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    epochs : int, default=20
    batch_size : int, default=64
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        epochs: int = 20,
        batch_size: int = 64,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        _set_torch_seed(self.random_state)
        np.random.seed(self.random_state if self.random_state is not None else 42)

        from pyod.models.deep_svdd import DeepSVDD

        n_features = X.shape[1]

        try:
            self._model = DeepSVDD(
                n_features=n_features,
                epochs=self.epochs,
                batch_size=self.batch_size,
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
