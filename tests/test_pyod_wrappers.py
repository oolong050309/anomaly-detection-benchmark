"""任务 7 + 8：8 个 PyOD 包装类的快速集成测试。"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from models.autoencoder import AutoEncoderDetector
from models.base import BaseDetector
from models.deep_svdd import DeepSVDDDetector
from models.ecod_copod import COPODDetector, ECODDetector
from models.iforest import IForestDetector
from models.knn import KNNDetector
from models.lof import LOFDetector
from models.ocsvm import OCSVMDetector


# 6 个浅层 PyOD 包装：固定容易跑过的小数据集
SHALLOW = [LOFDetector, KNNDetector, IForestDetector, ECODDetector, COPODDetector, OCSVMDetector]

# 2 个深度包装：epoch 数小一点防止测试慢
DEEP = [
    pytest.param(AutoEncoderDetector, {"epoch_num": 3}, id="AutoEncoder"),
    pytest.param(DeepSVDDDetector, {"epochs": 3}, id="DeepSVDD"),
]


@pytest.fixture
def small_dataset():
    X, _ = make_classification(
        n_samples=200, n_features=8, n_informative=4,
        weights=[0.9, 0.1], random_state=0,
    )
    return X.astype(np.float64)


@pytest.mark.parametrize("Cls", SHALLOW)
def test_shallow_wrappers_basic(Cls, small_dataset):
    X = small_dataset
    det = Cls(contamination=0.1, random_state=42)
    assert isinstance(det, BaseDetector)
    det.fit(X)
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("Cls", SHALLOW)
def test_shallow_wrappers_predict_label_set(Cls, small_dataset):
    X = small_dataset
    det = Cls(contamination=0.1, random_state=42).fit(X)
    labels = det.predict(X)
    assert set(np.unique(labels).tolist()).issubset({0, 1})


def test_iforest_idempotent(small_dataset):
    """IForest 在固定 random_state 下应该可复现。"""
    X = small_dataset
    s1 = IForestDetector(random_state=42).fit(X).decision_function(X)
    s2 = IForestDetector(random_state=42).fit(X).decision_function(X)
    assert np.allclose(s1, s2)


def test_ecod_copod_deterministic(small_dataset):
    """ECOD/COPOD 是确定性算法。"""
    X = small_dataset
    e1 = ECODDetector().fit(X).decision_function(X)
    e2 = ECODDetector().fit(X).decision_function(X)
    assert np.allclose(e1, e2)
    c1 = COPODDetector().fit(X).decision_function(X)
    c2 = COPODDetector().fit(X).decision_function(X)
    assert np.allclose(c1, c2)


def test_runtime_error_on_invalid_input():
    """构造 LOF 在 n=1 时会拒绝的情形（fit 一个样本）。
    底层异常应被重抛为 RuntimeError。"""
    X = np.array([[1.0, 2.0, 3.0]])  # n=1
    det = LOFDetector(n_neighbors=20)
    with pytest.raises(RuntimeError):
        det.fit(X)


@pytest.mark.parametrize("Cls,kwargs", DEEP)
def test_deep_wrappers_basic(Cls, kwargs, small_dataset):
    X = small_dataset
    det = Cls(contamination=0.1, random_state=42, **kwargs)
    det.fit(X)
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()
