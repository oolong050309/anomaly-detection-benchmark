"""BaseDetector / SupervisedDetector / TimeSeriesDetector / GraphDetector 测试。"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from models.base import (
    BaseDetector,
    GraphDetector,
    SupervisedDetector,
    TimeSeriesDetector,
)


# ---------------------------------------------------------------------------
# 测试用的最简 Detector 实现
# ---------------------------------------------------------------------------


class _DummyDetector(BaseDetector):
    """最简 Detector：用样本到训练均值的 L2 距离作为分数。"""

    def _fit(self, X, y=None, **kwargs):
        self.mean_ = X.mean(axis=0)

    def _decision_function(self, X):
        return np.linalg.norm(X - self.mean_, axis=1)


class _DummySupervised(SupervisedDetector):
    def _fit(self, X, y=None, **kwargs):
        self.pos_mean_ = X[y == 1].mean(axis=0)

    def _decision_function(self, X):
        return -np.linalg.norm(X - self.pos_mean_, axis=1)


class _DummyTS(TimeSeriesDetector):
    def _fit(self, X, y=None, **kwargs):
        self.mu_ = float(X.mean())

    def _decision_function(self, X):
        if X.ndim == 1:
            return np.abs(X - self.mu_)
        return np.abs(X - self.mu_).mean(axis=1)


class _DummyGraph(GraphDetector):
    def _fit(self, graph, y=None, **kwargs):
        self.n_nodes_ = graph.x.shape[0]

    def _decision_function(self, graph):
        # 返回每个节点的 |x| 范数
        import numpy as np
        return np.linalg.norm(np.asarray(graph.x), axis=1)


# ---------------------------------------------------------------------------
# BaseDetector
# ---------------------------------------------------------------------------


def test_contamination_validation():
    with pytest.raises(ValueError):
        _DummyDetector(contamination=0)
    with pytest.raises(ValueError):
        _DummyDetector(contamination=0.6)
    with pytest.raises(ValueError):
        _DummyDetector(contamination=-0.1)
    # 边界值合法
    _DummyDetector(contamination=0.5)
    _DummyDetector(contamination=0.001)


def test_not_fitted_error_decision_function():
    det = _DummyDetector()
    X = np.random.RandomState(0).randn(10, 3)
    with pytest.raises(NotFittedError):
        det.decision_function(X)


def test_not_fitted_error_predict():
    det = _DummyDetector()
    X = np.random.RandomState(0).randn(10, 3)
    with pytest.raises(NotFittedError):
        det.predict(X)


def test_fit_sets_is_fitted_flag():
    det = _DummyDetector()
    assert det.is_fitted_ is False
    X = np.random.RandomState(0).randn(20, 3)
    det.fit(X)
    assert det.is_fitted_ is True


def test_input_validation_rejects_nan():
    det = _DummyDetector()
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(ValueError):
        det.fit(X)


def test_input_validation_rejects_inf():
    det = _DummyDetector()
    X = np.array([[1.0, 2.0], [np.inf, 3.0]])
    with pytest.raises(ValueError):
        det.fit(X)


def test_input_validation_rejects_1d():
    det = _DummyDetector()
    X = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        det.fit(X)


def test_input_validation_rejects_empty():
    det = _DummyDetector()
    X = np.empty((0, 3))
    with pytest.raises(ValueError):
        det.fit(X)


def test_decision_function_shape():
    det = _DummyDetector()
    X = np.random.RandomState(0).randn(50, 4)
    det.fit(X)
    scores = det.decision_function(X)
    assert scores.shape == (50,)
    assert scores.dtype == np.float64


def test_predict_label_set():
    det = _DummyDetector(contamination=0.1)
    X = np.random.RandomState(0).randn(100, 4)
    det.fit(X)
    labels = det.predict(X)
    assert labels.shape == (100,)
    assert set(np.unique(labels).tolist()).issubset({0, 1})


def test_predict_contamination_count():
    det = _DummyDetector(contamination=0.1)
    X = np.random.RandomState(0).randn(100, 4)
    det.fit(X)
    labels = det.predict(X)
    # 阳性数 >= ceil(0.1 * 100) = 10（并列时可能略多）
    assert labels.sum() >= 10


def test_predict_idempotent_deterministic():
    det = _DummyDetector(contamination=0.1)
    X = np.random.RandomState(0).randn(100, 4)
    det.fit(X)
    p1 = det.predict(X)
    p2 = det.predict(X)
    assert np.array_equal(p1, p2)


def test_predict_monotonic_with_contamination():
    X = np.random.RandomState(0).randn(200, 4)
    det1 = _DummyDetector(contamination=0.05).fit(X)
    det2 = _DummyDetector(contamination=0.20).fit(X)
    pos1 = det1.predict(X).sum()
    pos2 = det2.predict(X).sum()
    assert pos1 <= pos2


# ---------------------------------------------------------------------------
# SupervisedDetector
# ---------------------------------------------------------------------------


def test_supervised_requires_y():
    det = _DummySupervised()
    X = np.random.RandomState(0).randn(20, 3)
    with pytest.raises(ValueError):
        det.fit(X)
    with pytest.raises(ValueError):
        det.fit(X, y=None)


def test_supervised_requires_two_classes():
    det = _DummySupervised()
    X = np.random.RandomState(0).randn(20, 3)
    y_single = np.zeros(20, dtype=int)
    with pytest.raises(ValueError):
        det.fit(X, y_single)


def test_supervised_rejects_non_binary_y():
    det = _DummySupervised()
    X = np.random.RandomState(0).randn(20, 3)
    y_multi = np.array([0, 1, 2] * 7)[:20]
    with pytest.raises(ValueError):
        det.fit(X, y_multi)


def test_supervised_happy_path():
    det = _DummySupervised(contamination=0.1)
    rng = np.random.RandomState(0)
    X = rng.randn(50, 3)
    y = (rng.rand(50) > 0.7).astype(int)
    if y.sum() == 0 or y.sum() == 50:
        y[0], y[-1] = 0, 1
    det.fit(X, y)
    scores = det.decision_function(X)
    assert scores.shape == (50,)


# ---------------------------------------------------------------------------
# TimeSeriesDetector
# ---------------------------------------------------------------------------


def test_timeseries_accepts_1d():
    det = _DummyTS(contamination=0.1)
    X = np.random.RandomState(0).randn(100)
    det.fit(X)
    scores = det.decision_function(X)
    assert scores.shape == (100,)


def test_timeseries_accepts_2d_windows():
    det = _DummyTS(contamination=0.1)
    X = np.random.RandomState(0).randn(20, 50)  # 20 个窗口，每窗口长 50
    det.fit(X)
    scores = det.decision_function(X)
    assert scores.shape == (20,)


def test_timeseries_rejects_3d():
    det = _DummyTS()
    X = np.random.RandomState(0).randn(5, 10, 2)
    with pytest.raises(ValueError):
        det.fit(X)


def test_timeseries_rejects_nan():
    det = _DummyTS()
    X = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError):
        det.fit(X)


# ---------------------------------------------------------------------------
# GraphDetector
# ---------------------------------------------------------------------------


def test_graph_rejects_none():
    det = _DummyGraph()
    with pytest.raises(ValueError):
        det.fit(None)


def test_graph_rejects_unsupported_type():
    det = _DummyGraph()
    with pytest.raises(ValueError):
        det.fit("not a graph")


@pytest.mark.skipif(
    pytest.importorskip("torch_geometric", reason="PyG not installed") is None,
    reason="PyG not installed",
)
def test_graph_pyg_happy_path():
    pytest.importorskip("torch_geometric")
    import torch
    from torch_geometric.data import Data

    x = torch.randn(10, 4)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
    )
    g = Data(x=x, edge_index=edge_index)
    det = _DummyGraph(contamination=0.1)
    det.fit(g)
    scores = det.decision_function(g)
    assert scores.shape == (10,)
