"""Tests for ``models._param_count.count_parameters``.

每种 detector 类型（PyTorch / sklearn linear / sklearn 树集成 /
XGBoost / LightGBM / OCSVM / TabPFN-like / MiniRocket / parameterless）
都给一个最小可复现实例，确认参数量推断正确。
"""

from __future__ import annotations

import numpy as np
import pytest

from models._param_count import count_parameters


# --------------------------------------------------------------------- helpers


class _FakeDetector:
    """模拟 detector：用动态子类名给 ``count_parameters`` 的 algorithm-hint 用。"""

    def __init__(self, model):
        self._model = model

    @classmethod
    def named(cls, model, name):
        # 用 type() 动态建子类，类名直接进 type(self).__name__
        sub = type(name, (cls,), {})
        return sub(model)


# --------------------------------------------------------------------- PyTorch


def test_torch_module_counted_exactly():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    det = _FakeDetector.named(model, "AutoEncoderDetector")
    # Linear(10,5) = 10*5 + 5 = 55；Linear(5,1) = 5*1 + 1 = 6 → 61
    assert count_parameters(det) == 61


def test_torch_nested_model_counted():
    """PyOD-style 嵌套 ``detector._model.model`` → 参数量从内层抓出。"""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Wrapper:
        def __init__(self):
            self.model = nn.Linear(4, 2)

    det = _FakeDetector.named(Wrapper(), "WrappedTorchDetector")
    # Linear(4,2) = 4*2 + 2 = 10
    assert count_parameters(det) == 10


# --------------------------------------------------------------------- sklearn


def test_sklearn_logistic_regression_counted():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 6))
    y = (rng.random(50) > 0.5).astype(int)
    lr = LogisticRegression().fit(X, y)
    det = _FakeDetector.named(lr, "LogisticRegressionDetector")
    # binary LR: coef_.shape (1, 6) → 6； intercept_ size 1 → 7
    assert count_parameters(det) == 7


def test_sklearn_mlp_counted():
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    y = (rng.random(30) > 0.5).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=20, random_state=0).fit(X, y)
    det = _FakeDetector.named(mlp, "MLPDetector")
    # weights: (4,8)=32, (8,4)=32, (4,1)=4 → 68
    # biases:  (8,)=8, (4,)=4, (1,)=1   → 13
    # total = 81
    assert count_parameters(det) == 81


def test_sklearn_random_forest_counted():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    y = (rng.random(40) > 0.5).astype(int)
    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    det = _FakeDetector.named(rf, "RandomForestDetector")
    n = count_parameters(det)
    # 5 棵树，每棵至少 1 个 root → 总节点 >= 5
    assert n is not None and n >= 5
    # 不应大到离谱
    assert n <= sum(t.tree_.node_count for t in rf.estimators_)


# --------------------------------------------------------------------- xgboost / lightgbm


def test_xgboost_counted():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    y = (rng.random(40) > 0.5).astype(int)
    model = xgb.XGBClassifier(n_estimators=3, max_depth=3, eval_metric="logloss").fit(X, y)
    det = _FakeDetector.named(model, "XGBoostDetector")
    n = count_parameters(det)
    assert n is not None and n > 0


def test_lightgbm_counted():
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (rng.random(50) > 0.5).astype(int)
    model = lgb.LGBMClassifier(n_estimators=3, num_leaves=7, verbose=-1).fit(X, y)
    det = _FakeDetector.named(model, "LightGBMDetector")
    n = count_parameters(det)
    # 3 棵树 × 至多 7 叶 = 至多 21
    assert n is not None and n > 0


# --------------------------------------------------------------------- OCSVM


def test_ocsvm_counted():
    from sklearn.svm import OneClassSVM

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    model = OneClassSVM(nu=0.1).fit(X)
    det = _FakeDetector.named(model, "OCSVMDetector")
    n = count_parameters(det)
    # support_vectors_.size = n_support × 4
    assert n is not None and n > 0
    assert n == model.support_vectors_.size


# --------------------------------------------------------------------- MiniRocket


def test_minirocket_counted_from_fit_params():
    """MiniRocket 没保存模型对象，参数量由 num_kernels 反推。"""
    det = _FakeDetector.named(object(), "MiniRocketDetector")
    n = count_parameters(det, fit_params={"num_kernels": 5000})
    # 5000 * 9 + 5000 = 50000
    assert n == 50000


# --------------------------------------------------------------------- parameterless


@pytest.mark.parametrize(
    "name",
    ["IQRDetector", "ECODDetector", "COPODDetector", "KNNDetector", "LOFDetector",
     "MatrixProfileDetector"],
)
def test_parameterless_algorithms_return_zero(name):
    det = _FakeDetector.named(object(), name)
    assert count_parameters(det) == 0


# --------------------------------------------------------------------- unknown


def test_unknown_returns_none():
    det = _FakeDetector.named(object(), "MysteryDetector")
    assert count_parameters(det) is None


def test_unprompt_returns_none():
    det = _FakeDetector.named(object(), "UNPromptDetector")
    assert count_parameters(det) is None
