"""真实数据适配器集成测试。

这些测试需要原始 benchmark 数据，默认从 `AD_DATA_ROOT` 环境变量读取。
如果没有配置数据目录，pytest 会自动跳过本文件中的测试。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from adapters import load_dataset
from adapters.adbench_adapter import iter_selected_adbench_names
from adapters.timeseries_adapter import select_representative_tsb_files


DATA_ROOT_ENV = os.environ.get("AD_DATA_ROOT")
DATA_ROOT = Path(DATA_ROOT_ENV) if DATA_ROOT_ENV else None
pytestmark = pytest.mark.skipif(
    DATA_ROOT is None or not DATA_ROOT.exists(),
    reason="需要设置 AD_DATA_ROOT 指向服务器原始数据目录",
)


def _assert_bundle_shapes(bundle) -> None:
    """检查统一四元组的基本形状和标签合法性。"""

    X_train, X_test, y_train, y_test = bundle.as_tuple()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_all_selected_adbench_datasets_load() -> None:
    """ADBench 计划使用的 17 个数据集都应能加载。"""

    for name in iter_selected_adbench_names():
        bundle = load_dataset(name, data_root=DATA_ROOT)
        _assert_bundle_shapes(bundle)
        assert bundle.metadata["source"] == "ADBench"
        assert bundle.metadata["n_samples"] > 0


def test_selected_tsb_file_loads_with_windows() -> None:
    """至少一条选定 TSB-AD 时序应能完成训练段标准化和滑窗切分。"""

    selected = select_representative_tsb_files(data_root=DATA_ROOT, max_files=1)
    assert selected

    bundle = load_dataset(
        selected[0],
        modality="timeseries",
        data_root=DATA_ROOT,
        window_size=100,
        stride=10,
    )

    _assert_bundle_shapes(bundle)
    assert bundle.X_train.ndim == 2
    assert bundle.X_train.shape[1] == 100
    assert bundle.metadata["split"] == "filename_tr_cutoff"
    assert bundle.metadata["preprocessing"]["standardization"] == "zscore_train_segment"


@pytest.mark.parametrize(
    ("name", "expected_split"),
    [
        ("tfinance", "generated_stratified_node_masks"),
        ("reddit", "generated_stratified_node_masks"),
        ("amazon", "predefined_masks"),
        ("weibo", "generated_stratified_node_masks"),
    ],
)
def test_gadbench_graphs_load(name: str, expected_split: str) -> None:
    """GADBench 4 个计划图都应能被 DGL 读取并返回节点四元组。"""

    bundle = load_dataset(name, modality="graph", data_root=DATA_ROOT, standardize=False)

    _assert_bundle_shapes(bundle)
    assert bundle.metadata["source"] == "GADBench"
    assert bundle.metadata["split"] == expected_split
    assert "graph" in bundle.extras
    assert "masks" in bundle.extras
