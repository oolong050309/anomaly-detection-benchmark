"""统一数据加载入口。

算法侧建议只调用 `load_dataset()`。该函数会根据数据集名称或显式
`modality` 自动分发到 ADBench、TSB-AD 或 GADBench 适配器。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .adbench_adapter import ADBENCH_DATASETS, load_adbench_dataset, normalize_adbench_name
from .common import DatasetBundle
from .graph_adapter import GADBENCH_DATASETS, load_gadbench_dataset
from .timeseries_adapter import load_tsb_dataset


def infer_modality(name: str) -> str:
    """根据数据集名称推断数据形态。"""

    key = normalize_adbench_name(name)
    if key in ADBENCH_DATASETS:
        return ADBENCH_DATASETS[key]["modality"]
    graph_key = name.strip().lower().replace("_", "-")
    if graph_key in GADBENCH_DATASETS:
        return "graph"
    if str(name).lower().endswith(".csv") or "_tr_" in str(name):
        return "timeseries"
    raise KeyError(f"Cannot infer modality for dataset: {name}")


def load_dataset(
    name: str | Path,
    modality: Optional[str] = None,
    data_root: Optional[str | Path] = None,
    **kwargs,
) -> DatasetBundle:
    """按名称加载数据集并返回 `DatasetBundle`。

    示例：
        load_dataset("cardio")
        load_dataset("CIFAR10_0")
        load_dataset("003_NAB_id_3_WebService_tr_1362_1st_1462.csv", modality="timeseries")
        load_dataset("reddit", modality="graph")
    """

    modality = (modality or infer_modality(str(name))).lower()
    if modality in {"tabular", "cv", "nlp", "adbench"}:
        return load_adbench_dataset(str(name), data_root=data_root, **kwargs)
    if modality in {"timeseries", "ts", "tsb-ad", "tsb"}:
        return load_tsb_dataset(name, data_root=data_root, **kwargs)
    if modality in {"graph", "gadbench"}:
        return load_gadbench_dataset(str(name), data_root=data_root, **kwargs)
    raise ValueError(f"Unsupported modality: {modality}")
