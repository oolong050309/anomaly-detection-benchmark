"""任务 9：6 个有监督算法封装的快速集成测试。"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from models.supervised import (
    LightGBMDetector,
    LogisticRegressionDetector,
    MLPDetector,
    RandomForestDetector,
    TabPFNDetector,
    XGBoostDetector,
)


# 5 个常规有监督检测器（TabPFN 单独测，避免触发模型权重下载）
SUPERVISED_CLS = [
    pytest.param(LogisticRegressionDetector, {}, id="LR"),
    pytest.param(RandomForestDetector, {"n_estimators": 30}, id="RF"),
    pytest.param(MLPDetector, {"max_iter": 30}, id="MLP"),
    pytest.param(XGBoostDetector, {"n_estimators": 30}, id="XGBoost"),
    pytest.param(LightGBMDetector, {"n_estimators": 30}, id="LightGBM"),
]


@pytest.fixture
def small_supervised_dataset():
    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=4,
        weights=[0.85, 0.15], random_state=0,
    )
    return X.astype(np.float64), y.astype(np.int64)


@pytest.mark.parametrize("Cls,kwargs", SUPERVISED_CLS)
def test_supervised_basic(Cls, kwargs, small_supervised_dataset):
    X, y = small_supervised_dataset
    det = Cls(contamination=0.15, random_state=42, **kwargs)
    det.fit(X, y)
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()
    # predict_proba 输出应在 [0, 1]
    assert (0.0 <= scores).all() and (scores <= 1.0).all()


@pytest.mark.parametrize("Cls,kwargs", SUPERVISED_CLS)
def test_supervised_requires_y(Cls, kwargs, small_supervised_dataset):
    X, _ = small_supervised_dataset
    det = Cls(random_state=42, **kwargs)
    with pytest.raises(ValueError):
        det.fit(X)


@pytest.mark.parametrize("Cls,kwargs", SUPERVISED_CLS)
def test_supervised_rejects_single_class_y(Cls, kwargs, small_supervised_dataset):
    X, _ = small_supervised_dataset
    y_single = np.zeros(X.shape[0], dtype=np.int64)
    det = Cls(random_state=42, **kwargs)
    with pytest.raises(ValueError):
        det.fit(X, y_single)


@pytest.mark.parametrize("Cls,kwargs", [
    pytest.param(LogisticRegressionDetector, {}, id="LR"),
    pytest.param(RandomForestDetector, {"n_estimators": 30}, id="RF"),
    pytest.param(XGBoostDetector, {"n_estimators": 30}, id="XGBoost"),
    pytest.param(LightGBMDetector, {"n_estimators": 30}, id="LightGBM"),
])
def test_supervised_idempotent(Cls, kwargs, small_supervised_dataset):
    """LR/RF/XGB/LGBM 在固定 random_state 下两次 fit 结果一致。"""
    X, y = small_supervised_dataset
    s1 = Cls(random_state=42, **kwargs).fit(X, y).decision_function(X)
    s2 = Cls(random_state=42, **kwargs).fit(X, y).decision_function(X)
    assert np.allclose(s1, s2, rtol=1e-4)


def test_tabpfn_rejects_too_many_samples():
    rng = np.random.RandomState(0)
    X = rng.randn(11000, 5)
    y = (rng.rand(11000) > 0.9).astype(np.int64)
    if y.sum() == 0 or y.sum() == 11000:
        y[0], y[-1] = 0, 1
    det = TabPFNDetector(random_state=42)
    with pytest.raises(RuntimeError, match="TabPFN supports"):
        det.fit(X, y)


def test_tabpfn_rejects_too_many_features():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 150)
    y = (rng.rand(500) > 0.85).astype(np.int64)
    if y.sum() == 0 or y.sum() == 500:
        y[0], y[-1] = 0, 1
    det = TabPFNDetector(random_state=42)
    with pytest.raises(RuntimeError, match="TabPFN supports"):
        det.fit(X, y)
