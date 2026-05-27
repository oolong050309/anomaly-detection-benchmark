"""MiniRocket + RidgeClassifierCV 时序有监督异常检测。

委托 ``sktime.transformations.panel.rocket.MiniRocket`` 提取卷积核特征，
然后用岭回归分类。``decision_function`` 返回 RidgeClassifierCV 的连续打分
（``decision_function``，而非概率），值越大表示属于正类（异常）的可能性越高。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import SupervisedDetector, TimeSeriesDetector


class MiniRocketDetector(TimeSeriesDetector, SupervisedDetector):
    """MiniRocket 特征 + RidgeClassifierCV 二分类。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    num_kernels : int, default=10000
        MiniRocket 的卷积核数量。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        num_kernels: int = 10000,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.num_kernels = num_kernels
        self._transformer: Any | None = None
        self._classifier: Any | None = None

    @staticmethod
    def _to_3d(X: np.ndarray) -> np.ndarray:
        """sktime panel 格式：(n_instances, n_channels, n_timepoints)。"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            # (n_instances, n_timepoints) -> (n_instances, 1, n_timepoints)
            return X[:, np.newaxis, :]
        if X.ndim == 3:
            return X
        raise ValueError(
            f"MiniRocketDetector expects 2D or 3D X, got shape {X.shape}"
        )

    def _fit(self, X, y=None, **kwargs):
        try:
            from sklearn.linear_model import RidgeClassifierCV
            from sktime.transformations.panel.rocket import MiniRocket
        except ImportError as e:
            raise RuntimeError(
                "[MiniRocketDetector] sktime/sklearn 未安装"
            ) from e

        X3 = self._to_3d(X)
        try:
            self._transformer = MiniRocket(
                num_kernels=self.num_kernels, random_state=self.random_state
            )
            features = self._transformer.fit_transform(X3)
            self._classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            self._classifier.fit(features, y)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        assert self._transformer is not None and self._classifier is not None
        X3 = self._to_3d(X)
        features = self._transformer.transform(X3)
        # RidgeClassifierCV.decision_function 在二分类下返回 (n,) 连续分数
        scores = self._classifier.decision_function(features)
        return np.asarray(scores, dtype=np.float64).ravel()

    # ---- 重写 _validate_input：MiniRocket 接受 2D / 3D ----

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim not in (2, 3):
            raise ValueError(
                f"MiniRocketDetector expects 2D or 3D X, got shape {X_arr.shape}"
            )
        if X_arr.size == 0:
            raise ValueError("X is empty")
        if not np.issubdtype(X_arr.dtype, np.number):
            raise ValueError(f"X must be numeric, got {X_arr.dtype}")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or Inf")
        return X_arr.astype(np.float64, copy=False)
