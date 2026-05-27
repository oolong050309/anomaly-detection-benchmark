"""任务 10：4 个时序算法的快速集成测试。

使用合成 sin 波序列，故意在某些位置注入异常点。
"""

from __future__ import annotations

import numpy as np
import pytest

from models.timeseries.lstm_ae import LSTMAutoEncoderDetector
from models.timeseries.lstm_supervised import LSTMSupervisedDetector
from models.timeseries.matrix_profile import MatrixProfileDetector
from models.timeseries.minirocket import MiniRocketDetector


def _make_synthetic_series(T: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 20 * np.pi, T)
    seq = np.sin(t) + 0.05 * rng.randn(T)
    # 注入异常
    seq[200:210] += 5.0
    seq[400:405] -= 5.0
    return seq.astype(np.float64)


def _make_windowed_dataset(
    n_windows: int = 60, w: int = 50, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n_windows, w) * 0.1
    # 一半窗口加正弦信号，另一半加正弦+异常脉冲
    t = np.linspace(0, 4 * np.pi, w)
    base = np.sin(t)
    y = np.zeros(n_windows, dtype=np.int64)
    for i in range(n_windows):
        X[i] += base
        if i % 5 == 0:  # 20% 异常
            X[i, w // 2] += 3.0
            y[i] = 1
    return X.astype(np.float64), y


# ---------------------------------------------------------------------------
# MatrixProfile
# ---------------------------------------------------------------------------


def test_matrix_profile_basic():
    seq = _make_synthetic_series()
    det = MatrixProfileDetector(window_size=50, contamination=0.1)
    det.fit(seq)
    scores = det.decision_function(seq)
    assert scores.shape == (seq.size,)
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# MiniRocket
# ---------------------------------------------------------------------------


def test_minirocket_basic():
    X, y = _make_windowed_dataset()
    det = MiniRocketDetector(num_kernels=500, random_state=42)
    det.fit(X, y)
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()


def test_minirocket_requires_y():
    X, _ = _make_windowed_dataset()
    det = MiniRocketDetector(num_kernels=500, random_state=42)
    with pytest.raises(ValueError):
        det.fit(X)


# ---------------------------------------------------------------------------
# LSTM-AE
# ---------------------------------------------------------------------------


def test_lstm_ae_basic():
    X, _ = _make_windowed_dataset(n_windows=40, w=30)
    det = LSTMAutoEncoderDetector(
        hidden_size=8, num_layers=1, epochs=2, batch_size=16, random_state=42
    )
    det.fit(X)
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
    assert (scores >= 0).all()
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# LSTM Supervised
# ---------------------------------------------------------------------------


def test_lstm_supervised_basic():
    X, y = _make_windowed_dataset(n_windows=40, w=30)
    det = LSTMSupervisedDetector(
        hidden_size=8, num_layers=1, epochs=2, batch_size=16, random_state=42
    )
    det.fit(X, y)
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_lstm_supervised_requires_y():
    X, _ = _make_windowed_dataset(n_windows=40, w=30)
    det = LSTMSupervisedDetector(
        hidden_size=8, num_layers=1, epochs=2, batch_size=16, random_state=42
    )
    with pytest.raises(ValueError):
        det.fit(X)
