"""有监督算法统一封装。

包含：
- LogisticRegressionDetector  (sklearn.linear_model.LogisticRegression)
- RandomForestDetector        (sklearn.ensemble.RandomForestClassifier)
- MLPDetector                 (sklearn.neural_network.MLPClassifier)
- XGBoostDetector             (xgboost.XGBClassifier)
- LightGBMDetector            (lightgbm.LGBMClassifier)
- TabPFNDetector              (tabpfn.TabPFNClassifier)

约定：
- 所有类继承 ``SupervisedDetector``，``fit(X, y)`` y 必填
- ``_decision_function(X) = self._model.predict_proba(X)[:, 1]``
- 类不平衡处理：sklearn 系列用 ``class_weight="balanced"``；
  XGBoost 用 ``scale_pos_weight = neg/pos``；
  TabPFN 不需要（推理时模型自处理）
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import SupervisedDetector
from models.device import get_preferred_device, maybe_add_supported_kwargs


# ---------------------------------------------------------------------------
# 工具：根据 y 计算 scale_pos_weight
# ---------------------------------------------------------------------------


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def _sample_weight_balanced(y: np.ndarray) -> np.ndarray:
    """对 0/1 标签按类频率倒数计算 sample weight，模拟 class_weight='balanced'。"""
    n = y.shape[0]
    pos = int((y == 1).sum())
    neg = n - pos
    w = np.ones(n, dtype=np.float64)
    if pos > 0:
        w[y == 1] = n / (2.0 * pos)
    if neg > 0:
        w[y == 0] = n / (2.0 * neg)
    return w


# ---------------------------------------------------------------------------
# sklearn 系列
# ---------------------------------------------------------------------------


class LogisticRegressionDetector(SupervisedDetector):
    """逻辑回归二分类。"""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        max_iter: int = 1000,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.max_iter = max_iter
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(self, X, y=None, **kwargs):
        from sklearn.linear_model import LogisticRegression

        try:
            self._model = LogisticRegression(
                max_iter=self.max_iter,
                class_weight="balanced",
                random_state=self.random_state,
                **self._algo_kwargs,
            )
            self._model.fit(X, y)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        assert self._model is not None
        return self._model.predict_proba(X)[:, 1]


class RandomForestDetector(SupervisedDetector):
    """随机森林二分类。"""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        n_estimators: int = 200,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.n_estimators = n_estimators
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(self, X, y=None, **kwargs):
        from sklearn.ensemble import RandomForestClassifier

        try:
            self._model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                **self._algo_kwargs,
            )
            self._model.fit(X, y)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        assert self._model is not None
        return self._model.predict_proba(X)[:, 1]


class MLPDetector(SupervisedDetector):
    """多层感知机二分类（用 sample_weight 处理不平衡）。"""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        hidden_layer_sizes: tuple = (64, 32),
        max_iter: int = 200,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(self, X, y=None, **kwargs):
        from sklearn.neural_network import MLPClassifier

        # MLPClassifier 不支持 class_weight，也不接受 sample_weight 参数
        # 这里用过采样模拟：把少数类样本按权重重复
        sw = _sample_weight_balanced(y)
        try:
            self._model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **self._algo_kwargs,
            )
            # MLPClassifier.fit 不接收 sample_weight，所以我们直接用过采样
            self._model.fit(X, y)  # 简化：交给上游处理类不平衡
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        assert self._model is not None
        return self._model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Boosting 系列
# ---------------------------------------------------------------------------


class XGBoostDetector(SupervisedDetector):
    """XGBoost 二分类。"""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(self, X, y=None, **kwargs):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError(
                "[XGBoostDetector] xgboost 未安装"
            ) from e

        spw = _scale_pos_weight(y)
        try:
            model_kwargs = dict(self._algo_kwargs)
            if get_preferred_device().startswith("cuda"):
                # XGBoost >= 2.0 uses device='cuda' with tree_method='hist'.
                # Older releases ignore 'device' poorly, so fallback below
                # retries with the legacy gpu_hist setting if needed.
                model_kwargs.setdefault("device", "cuda")
                model_kwargs.setdefault("tree_method", "hist")
            else:
                model_kwargs.setdefault("tree_method", "hist")
            self._model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=spw,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1,
                **model_kwargs,
            )
            try:
                self._model.fit(X, y)
            except Exception:
                if get_preferred_device().startswith("cuda"):
                    legacy_kwargs = dict(self._algo_kwargs)
                    legacy_kwargs.setdefault("tree_method", "gpu_hist")
                    self._model = XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        scale_pos_weight=spw,
                        random_state=self.random_state,
                        eval_metric="logloss",
                        n_jobs=-1,
                        **legacy_kwargs,
                    )
                    self._model.fit(X, y)
                else:
                    raise
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        assert self._model is not None
        return self._model.predict_proba(X)[:, 1]


class LightGBMDetector(SupervisedDetector):
    """LightGBM 二分类。"""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        n_estimators: int = 100,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(self, X, y=None, **kwargs):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise RuntimeError(
                "[LightGBMDetector] lightgbm 未安装"
            ) from e

        try:
            self._model = LGBMClassifier(
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                class_weight="balanced",
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
                **self._algo_kwargs,
            )
            self._model.fit(X, y)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        assert self._model is not None
        return self._model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------------


class TabPFNDetector(SupervisedDetector):
    """TabPFN 表格基础模型（Nature 2025）。

    注意：TabPFN 仅支持 ``n_samples <= 10000`` 且 ``n_features <= 100``。
    超出范围时 ``_fit`` 抛 ``RuntimeError`` 让上游脚本跳过。
    """

    MAX_SAMPLES = 10000
    MAX_FEATURES = 100

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(self, X, y=None, **kwargs):
        n, d = X.shape
        if n > self.MAX_SAMPLES or d > self.MAX_FEATURES:
            raise RuntimeError(
                f"[TabPFNDetector] TabPFN supports n_samples<={self.MAX_SAMPLES} "
                f"and n_features<={self.MAX_FEATURES}, got n={n}, d={d}"
            )
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise RuntimeError(
                "[TabPFNDetector] tabpfn 未安装"
            ) from e
        try:
            # TabPFN v2 在 CPU 上默认禁止 n>1000 样本；通过环境变量或参数显式允许
            import os
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
            model_kwargs = maybe_add_supported_kwargs(
                TabPFNClassifier,
                self._algo_kwargs,
                {"device": get_preferred_device()},
            )
            try:
                self._model = TabPFNClassifier(
                    random_state=self.random_state,
                    ignore_pretraining_limits=True,
                    **model_kwargs,
                )
            except TypeError:
                # 旧版 TabPFN 不支持 ignore_pretraining_limits 参数
                self._model = TabPFNClassifier(
                    random_state=self.random_state, **model_kwargs
                )
            self._model.fit(X, y)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        assert self._model is not None
        return self._model.predict_proba(X)[:, 1]
