"""ADBench 数据适配器。

负责读取已选的 ADBench `.npz` 文件，并统一完成类型转换、分层划分、
训练集标准化和元数据统计。ADBench 覆盖表格、CV 特征和 NLP embedding。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .common import (
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DatasetBundle,
    first_existing,
    get_data_root,
    standardize_train_test,
    stratified_train_test_split,
    summarize_array,
)


ADBENCH_DATASETS: Dict[str, Dict[str, str]] = {
    "cardio": {"modality": "tabular", "group": "Classical", "file": "6_cardio.npz"},
    "thyroid": {"modality": "tabular", "group": "Classical", "file": "38_thyroid.npz"},
    "satellite": {"modality": "tabular", "group": "Classical", "file": "30_satellite.npz"},
    "shuttle": {"modality": "tabular", "group": "Classical", "file": "32_shuttle.npz"},
    "credit_card": {"modality": "tabular", "group": "Classical", "file": "13_fraud.npz"},
    "fraud": {"modality": "tabular", "group": "Classical", "file": "13_fraud.npz"},
    "pima": {"modality": "tabular", "group": "Classical", "file": "29_Pima.npz"},
    "annthyroid": {"modality": "tabular", "group": "Classical", "file": "2_annthyroid.npz"},
    "mammography": {"modality": "tabular", "group": "Classical", "file": "23_mammography.npz"},
    "pendigits": {"modality": "tabular", "group": "Classical", "file": "28_pendigits.npz"},
    "cifar10_0": {"modality": "cv", "group": "CV_by_ResNet18", "file": "CIFAR10_0.npz"},
    "cifar10_1": {"modality": "cv", "group": "CV_by_ResNet18", "file": "CIFAR10_1.npz"},
    "fashionmnist_0": {
        "modality": "cv",
        "group": "CV_by_ResNet18",
        "file": "FashionMNIST_0.npz",
    },
    "fashionmnist_1": {
        "modality": "cv",
        "group": "CV_by_ResNet18",
        "file": "FashionMNIST_1.npz",
    },
    "20news_0": {"modality": "nlp", "group": "NLP_by_BERT", "file": "20news_0.npz"},
    "20news_1": {"modality": "nlp", "group": "NLP_by_BERT", "file": "20news_1.npz"},
    "agnews_0": {"modality": "nlp", "group": "NLP_by_BERT", "file": "agnews_0.npz"},
    "amazon": {"modality": "nlp", "group": "NLP_by_BERT", "file": "amazon.npz"},
}

# fraud/credit_card 在 ADBench 中通常已经做过归一化，默认跳过标准化。
ALREADY_NORMALIZED = {"credit_card", "fraud"}


def normalize_adbench_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def find_adbench_root(data_root: Optional[str | Path] = None) -> Path:
    root = get_data_root(data_root)
    candidates = [
        root / "raw" / "ADBench",
        root / "ADBench",
        root.parent / "repos" / "ADBench-main" / "adbench" / "datasets",
        root.parent / "ADBench-main" / "adbench" / "datasets",
    ]
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "ADBench data root not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )
    return found


def get_adbench_file(name: str, data_root: Optional[str | Path] = None) -> Path:
    key = normalize_adbench_name(name)
    if key not in ADBENCH_DATASETS:
        raise KeyError(f"Unknown ADBench dataset: {name}")
    spec = ADBENCH_DATASETS[key]
    root = find_adbench_root(data_root)
    candidates = [
        root / spec["group"] / spec["file"],
        root / spec["file"],
    ]
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            f"Missing ADBench file for {name}: expected {spec['group']}/{spec['file']}"
        )
    return found


def load_adbench_dataset(
    name: str,
    data_root: Optional[str | Path] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    standardize: Optional[bool] = None,
) -> DatasetBundle:
    """加载一个 ADBench 数据集，并返回统一的训练/测试四元组。"""

    key = normalize_adbench_name(name)
    path = get_adbench_file(key, data_root)
    spec = ADBENCH_DATASETS[key]
    with np.load(path, allow_pickle=False) as data:
        if "X" not in data or "y" not in data:
            raise KeyError(f"{path} must contain X and y arrays")
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"]).astype(int).reshape(-1)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=test_size, seed=seed
    )
    if standardize is None:
        standardize = key not in ALREADY_NORMALIZED

    preprocessing = {"standardization": "skipped"}
    if standardize:
        X_train, X_test, preprocessing = standardize_train_test(X_train, X_test)

    metadata = {
        "source": "ADBench",
        "path": str(path),
        "file": spec["file"],
        "group": spec["group"],
        "split": "stratified_train_test_split",
        "test_size": test_size,
        "seed": seed,
        "preprocessing": preprocessing,
        **summarize_array(X, y),
    }
    return DatasetBundle(
        name=key,
        modality=spec["modality"],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
    )


def iter_selected_adbench_names():
    """按固定顺序产出本项目选定的 ADBench 数据集名称。"""

    names = [
        "cardio",
        "thyroid",
        "satellite",
        "shuttle",
        "credit_card",
        "pima",
        "annthyroid",
        "mammography",
        "pendigits",
        "cifar10_0",
        "cifar10_1",
        "fashionmnist_0",
        "fashionmnist_1",
        "20news_0",
        "20news_1",
        "agnews_0",
        "amazon",
    ]
    yield from names
