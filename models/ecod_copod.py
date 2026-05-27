"""ECOD + COPOD 包装。

两者均为基于经验累积分布的确定性异常检测算法，
不需要 random_state（保留为接口一致性参数）。

- ECOD: Empirical Cumulative Distribution-based Outlier Detection (TKDE 2022)
- COPOD: Copula-Based Outlier Detection (ICDM 2020)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseDetector


class _ECOPBase(BaseDetector):
    """ECOD/COPOD 的共享逻辑。"""

    _UNDERLYING_CLS: Any = None  # 由子类覆盖

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        try:
            self._model = self._UNDERLYING_CLS(
                contamination=self.contamination, **self._algo_kwargs
            )
            self._model.fit(X)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.decision_function(X)


class ECODDetector(_ECOPBase):
    """Empirical Cumulative Distribution-based Outlier Detection."""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        **algo_kwargs: Any,
    ) -> None:
        from pyod.models.ecod import ECOD

        type(self)._UNDERLYING_CLS = ECOD
        super().__init__(
            contamination=contamination, random_state=random_state, **algo_kwargs
        )


class COPODDetector(_ECOPBase):
    """Copula-Based Outlier Detection."""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        **algo_kwargs: Any,
    ) -> None:
        from pyod.models.copod import COPOD

        type(self)._UNDERLYING_CLS = COPOD
        super().__init__(
            contamination=contamination, random_state=random_state, **algo_kwargs
        )
