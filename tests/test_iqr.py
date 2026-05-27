"""IQRDetector 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from models.statistical import IQRDetector


def test_basic_fit_and_score():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    det = IQRDetector(contamination=0.1).fit(X)
    scores = det.decision_function(X)
    assert scores.shape == (100,)
    assert (scores >= 0).all()


def test_invalid_aggregation():
    with pytest.raises(ValueError):
        IQRDetector(aggregation="median")


def test_three_aggregations_run():
    X = np.random.RandomState(0).randn(50, 3)
    for agg in ("max", "mean", "sum"):
        det = IQRDetector(aggregation=agg).fit(X)
        s = det.decision_function(X)
        assert s.shape == (50,)


def test_constant_column_no_contribution():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 4)
    X[:, 1] = 5.0  # 第 2 列常量
    det = IQRDetector(aggregation="sum").fit(X)
    # 常量列被检测到
    assert det.constant_mask_[1]
    assert not det.constant_mask_[[0, 2, 3]].any()
    scores = det.decision_function(X)
    # 不应该出现 NaN/Inf
    assert np.isfinite(scores).all()


def test_outliers_get_higher_scores():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    # 前 10 个明显异常
    X[:10] += 10.0
    det = IQRDetector(aggregation="max").fit(X)
    scores = det.decision_function(X)
    # 异常样本的均值分数应明显高于正常样本
    assert scores[:10].mean() > scores[10:].mean() + 1.0


def test_idempotent():
    X = np.random.RandomState(0).randn(50, 3)
    det = IQRDetector().fit(X)
    s1 = det.decision_function(X)
    s2 = det.decision_function(X)
    assert np.array_equal(s1, s2)


def test_predict_returns_binary():
    X = np.random.RandomState(0).randn(100, 4)
    det = IQRDetector(contamination=0.1).fit(X)
    labels = det.predict(X)
    assert set(np.unique(labels).tolist()).issubset({0, 1})
