"""One-Class SVM 包装（Schölkopf et al., NeurIPS 1999）。

委托 ``pyod.models.ocsvm.OCSVM``。底层是 libsvm，单线程顺序求解器。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseDetector


class OCSVMDetector(BaseDetector):
    """One-Class SVM 异常检测器。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
        libsvm 求解器，参数保留为接口一致。
    kernel : str, default="rbf"
    nu : float, default=0.1
        异常率上界估计；通常与 contamination 一致即可。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        kernel: str = "rbf",
        nu: float = 0.1,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.kernel = kernel
        self.nu = nu
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        from pyod.models.ocsvm import OCSVM

        try:
            self._model = OCSVM(
                kernel=self.kernel,
                nu=self.nu,
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
