"""eval/metrics.py 单元测试。"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from eval.metrics import auc_pr, auc_roc, evaluate_all, f1_at_best


def test_auc_roc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc_roc(y, s) == 1.0


def test_auc_roc_inverted():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert auc_roc(y, s) == 0.0


def test_auc_roc_random_in_range():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=200)
    if y.sum() == 0 or y.sum() == 200:
        y[0], y[-1] = 0, 1
    s = rng.randn(200)
    val = auc_roc(y, s)
    assert 0.0 <= val <= 1.0


def test_auc_roc_single_class_returns_nan():
    y = np.zeros(10, dtype=int)
    s = np.random.RandomState(0).randn(10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = auc_roc(y, s)
    assert np.isnan(val)
    assert len(w) >= 1
    assert issubclass(w[-1].category, UserWarning)


def test_auc_pr_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_pr(y, s) == 1.0


def test_auc_pr_single_class_returns_nan():
    y = np.zeros(10, dtype=int)
    s = np.random.RandomState(0).randn(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(auc_pr(y, s))


def test_f1_at_best_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    f1, thr = f1_at_best(y, s)
    assert f1 == pytest.approx(1.0)


def test_f1_at_best_in_range():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, size=100)
    if y.sum() == 0:
        y[0] = 1
    s = rng.randn(100)
    f1, thr = f1_at_best(y, s)
    assert 0.0 <= f1 <= 1.0


def test_f1_at_best_all_negatives():
    y = np.zeros(20, dtype=int)
    s = np.random.RandomState(0).randn(20)
    f1, thr = f1_at_best(y, s)
    assert f1 == 0.0


def test_evaluate_all_returns_four_keys():
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=100)
    if y.sum() == 0 or y.sum() == 100:
        y[0], y[-1] = 0, 1
    s = rng.randn(100)
    result = evaluate_all(y, s)
    assert set(result.keys()) == {
        "auc_roc",
        "auc_pr",
        "f1_best",
        "best_threshold",
    }


def test_invalid_y_true_non_binary():
    y = np.array([0, 1, 2, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        auc_roc(y, s)


def test_invalid_y_score_nan():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, np.nan, 0.3, 0.4])
    with pytest.raises(ValueError):
        auc_roc(y, s)


def test_invalid_y_score_inf():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, np.inf, 0.3, 0.4])
    with pytest.raises(ValueError):
        auc_roc(y, s)


def test_invalid_length_mismatch():
    y = np.array([0, 1, 0])
    s = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        auc_roc(y, s)


def test_invalid_empty():
    y = np.array([], dtype=int)
    s = np.array([], dtype=float)
    with pytest.raises(ValueError):
        auc_roc(y, s)


def test_auc_roc_inversion_symmetry():
    """AUC(y, -s) ≈ 1 - AUC(y, s)（无并列时严格成立）。"""
    rng = np.random.RandomState(7)
    y = rng.randint(0, 2, size=200)
    if y.sum() == 0 or y.sum() == 200:
        y[0], y[-1] = 0, 1
    s = rng.randn(200)
    a = auc_roc(y, s)
    b = auc_roc(y, -s)
    assert a + b == pytest.approx(1.0, abs=1e-9)
