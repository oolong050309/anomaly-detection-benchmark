"""时序 Exp-1 三个 bug 的探索属性测试。

未修复版本：3 个测试都应 FAIL（ImportError 或断言失败），证明 bug 存在。
修复完成：3 个测试全部 PASS（DADA 测试在权重缺失时自动 skip）。

对应 spec：.kiro/specs/ad-timeseries-exp1-fixes/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =====================================================================
# Bug 1: 时序训练集双类化
# =====================================================================


def test_bug1_train_has_two_classes_synthetic():
    """合成单类训练集 + 双类测试集，验证 _borrow_for_supervised 能借出双类训练。

    用合成数据避免依赖真实 TSB-AD（CI 友好）。
    """
    from experiments.exp1_baseline import _borrow_for_supervised

    rng = np.random.default_rng(0)
    # 训练全是正常窗口
    X_tr = rng.standard_normal((50, 100))
    y_tr = np.zeros(50, dtype=int)
    # 测试集双类（30 正常 + 20 异常）
    X_te = rng.standard_normal((50, 100))
    y_te = np.concatenate([np.zeros(30, int), np.ones(20, int)])

    X_tr_aug, y_tr_aug, X_te_kept, y_te_kept, n_borrowed, note = _borrow_for_supervised(
        X_tr, y_tr, X_te, y_te, rng_seed=42
    )

    assert n_borrowed > 0, "应至少借走 1 个窗口"
    assert y_tr_aug.sum() > 0, "训练集必须含异常"
    assert (y_tr_aug == 0).sum() > 0, "训练集必须含正常"
    assert "borrowed" in note, f"notes 应含 'borrowed'，got: {note!r}"
    assert len(y_te_kept) == len(y_te) - n_borrowed, "测试集必须同步移除被借窗口"


def test_bug1_no_borrow_when_already_balanced():
    """训练已经双类时 _borrow_for_supervised 不应改动。"""
    from experiments.exp1_baseline import _borrow_for_supervised

    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((40, 100))
    y_tr = np.concatenate([np.zeros(30, int), np.ones(10, int)])
    X_te = rng.standard_normal((20, 100))
    y_te = np.concatenate([np.zeros(15, int), np.ones(5, int)])

    X_tr_aug, y_tr_aug, X_te_kept, y_te_kept, n_borrowed, note = _borrow_for_supervised(
        X_tr, y_tr, X_te, y_te
    )
    assert n_borrowed == 0
    assert note == ""
    assert np.array_equal(y_tr_aug, y_tr)
    assert np.array_equal(y_te_kept, y_te)


def test_bug1_no_data_leakage():
    """被借走的窗口必须从 test 同步移除（防数据泄漏）。"""
    from experiments.exp1_baseline import _borrow_for_supervised

    rng = np.random.default_rng(0)
    # 用可识别的标记值（每个窗口第一个元素是它的原始 index）
    X_tr = np.zeros((20, 100))
    y_tr = np.zeros(20, dtype=int)
    X_te = np.tile(np.arange(100).reshape(1, 100), (40, 1)).astype(float)
    X_te[:, 0] = np.arange(40)  # 第 0 列 = 该窗口的 index
    y_te = np.concatenate([np.zeros(20, int), np.ones(20, int)])

    X_tr_aug, y_tr_aug, X_te_kept, y_te_kept, n_borrowed, _ = _borrow_for_supervised(
        X_tr, y_tr, X_te, y_te, rng_seed=42
    )
    assert n_borrowed > 0

    # 训练里新增的那部分 index 不应再出现在测试集
    new_train_indices = set(X_tr_aug[20:, 0].astype(int).tolist())
    kept_test_indices = set(X_te_kept[:, 0].astype(int).tolist())
    assert new_train_indices.isdisjoint(kept_test_indices), \
        f"数据泄漏：{new_train_indices & kept_test_indices} 同时出现在 train 和 test"


# =====================================================================
# Bug 2: 极端比例数据集汇总过滤
# =====================================================================


def test_bug2_extreme_imbalance_flagged():
    """annotate_extreme_imbalance 应正确标记极端比例数据集。"""
    from scripts.analyze_results import annotate_extreme_imbalance

    df = pd.DataFrame({
        "dataset_name": ["normal", "all_anom", "no_anom", "boundary_low", "boundary_high"],
        "train_anomaly_rate": [0.05, 0.999, 0.000, 0.01, 0.99],
        "test_anomaly_rate":  [0.04, 0.996, 0.000, 0.01, 0.99],
    })
    out = annotate_extreme_imbalance(df, lo=0.01, hi=0.99)

    assert out.loc[0, "extreme_imbalance"] == False  # 正常
    assert out.loc[1, "extreme_imbalance"] == True   # > 0.99
    assert out.loc[2, "extreme_imbalance"] == True   # < 0.01
    # 边界值（0.01 / 0.99）：用 < / > 严格不等号 → 边界 NOT extreme
    assert out.loc[3, "extreme_imbalance"] == False
    assert out.loc[4, "extreme_imbalance"] == False


def test_bug2_summarize_three_groups():
    """summarize_by_group 返回 all / normal_only / extreme_only 三组按算法的聚合。"""
    from scripts.analyze_results import annotate_extreme_imbalance, summarize_by_group

    df = pd.DataFrame({
        "dataset_name":       ["A", "A", "B", "B"],
        "algorithm":          ["LR", "RF", "LR", "RF"],
        "train_anomaly_rate": [0.05, 0.05, 0.999, 0.999],
        "test_anomaly_rate":  [0.04, 0.04, 0.996, 0.996],
        "auc_roc":            [0.8, 0.9, 0.02, 0.5],
        "auc_pr":             [0.6, 0.7, 0.99, 0.8],
        "f1_best":            [0.7, 0.8, 0.998, 0.9],
        "status":             ["success"] * 4,
    })
    df = annotate_extreme_imbalance(df)
    summaries = summarize_by_group(df)

    assert set(summaries.keys()) == {"all", "normal_only", "extreme_only"}
    # normal_only 只剩 A 数据集（2 行）
    assert len(summaries["normal_only"]) == 2
    # extreme_only 只剩 B 数据集（2 行）
    assert len(summaries["extreme_only"]) == 2


def test_bug2_does_not_mutate_input():
    """annotate_extreme_imbalance 不能修改输入 df（纯函数）。"""
    from scripts.analyze_results import annotate_extreme_imbalance

    df = pd.DataFrame({
        "train_anomaly_rate": [0.05, 0.999],
        "test_anomaly_rate":  [0.04, 0.996],
    })
    cols_before = list(df.columns)
    _ = annotate_extreme_imbalance(df)
    assert list(df.columns) == cols_before, "原 df 列被修改了"


# =====================================================================
# Bug 3: DADA 长序列推理性能
# =====================================================================


def _dada_available() -> bool:
    """DADA 权重 + transformers 都齐备时返回 True。"""
    try:
        import torch  # noqa: F401
        from transformers import AutoModel  # noqa: F401
    except ImportError:
        return False
    weights = ROOT / "models" / "timeseries" / "_vendor" / "dada" / "pytorch_model.bin"
    return weights.exists()


@pytest.mark.skipif(not _dada_available(), reason="DADA weights or transformers unavailable")
def test_bug3_dada_long_sequence_speed():
    """100k 点合成序列推理 wall time < 60s（CPU）/ < 30s（GPU）。"""
    import torch
    from models.timeseries.dada import DADADetector

    rng = np.random.default_rng(0)
    seq = rng.standard_normal(100_000).astype(np.float32)

    det = DADADetector(batch_size=32)
    det.fit(seq[:1000])  # zero-shot，触发权重加载

    t0 = time.perf_counter()
    scores = det.decision_function(seq)
    elapsed = time.perf_counter() - t0

    threshold = 30.0 if torch.cuda.is_available() else 60.0
    assert elapsed < threshold, f"DADA scoring took {elapsed:.1f}s, expected < {threshold}s"
    assert scores.shape == seq.shape, f"输出形状不匹配: {scores.shape} vs {seq.shape}"


@pytest.mark.skipif(not _dada_available(), reason="DADA weights unavailable")
def test_bug3_dada_batch_size_kwarg_exists():
    """DADADetector.__init__ 必须接受 batch_size kwarg。"""
    from models.timeseries.dada import DADADetector

    det = DADADetector(batch_size=8)
    assert det.batch_size == 8

    det2 = DADADetector()
    assert det2.batch_size >= 1, "batch_size 必须有合法默认值"



# =====================================================================
# 模型参数量统计（不属于 spec 三 bug，但便于后续画图）
# =====================================================================


def test_num_params_torch_module():
    """带 .parameters() 的 torch nn.Module 应被识别。"""
    pytest.importorskip("torch")
    import torch.nn as nn
    from experiments.exp1_baseline import _count_model_params

    class Det:
        def __init__(self):
            self.model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))

    n = _count_model_params(Det())
    # Linear(10,5)=55, Linear(5,1)=6 → 61
    assert n == 61


def test_num_params_returns_none_for_traditional_ml():
    """sklearn / xgboost / IForest 等没有 .parameters() 的算法返回 None。"""
    from experiments.exp1_baseline import _count_model_params

    class Det:
        def __init__(self):
            class DummyTreeModel:
                pass
            self._model = DummyTreeModel()

    assert _count_model_params(Det()) is None


def test_num_params_handles_pyod_nested_model():
    """PyOD 包装常见结构 detector._model.model 嵌套也能识别。"""
    pytest.importorskip("torch")
    import torch.nn as nn
    from experiments.exp1_baseline import _count_model_params

    class Inner:
        def __init__(self):
            self.model = nn.Linear(4, 2)

    class Det:
        def __init__(self):
            self._model = Inner()

    n = _count_model_params(Det())
    # Linear(4,2) = 4*2 + 2 = 10
    assert n == 10


def test_augment_kwargs_with_num_params_does_not_mutate():
    """注入参数量不能改变原 kwargs。"""
    from experiments.exp1_baseline import _augment_kwargs_with_num_params

    class Det:
        pass

    kwargs = {"epochs": 20, "hidden": 64}
    out = _augment_kwargs_with_num_params(kwargs, Det())
    assert "epochs" in out and "hidden" in out
    assert "_num_params" in out
    assert "_num_params" not in kwargs  # 原 dict 不变
