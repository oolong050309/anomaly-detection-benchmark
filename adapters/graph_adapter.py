"""GADBench 图数据适配器。

负责用 DGL 读取 GADBench 图文件，提取节点特征、标签和 train/test mask。
若部分图文件没有预设 mask，则用固定随机种子生成分层节点划分，并在
metadata 中记录划分来源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .common import DEFAULT_SEED, DatasetBundle, first_existing, get_data_root, standardize_train_test


GADBENCH_DATASETS = {
    "t-finance": "t-finance",
    "t_finance": "t-finance",
    "tfinance": "tfinance",
    "reddit": "reddit",
    "amazon": "amazon",
    "weibo": "weibo",
}


def find_gadbench_root(data_root: Optional[str | Path] = None) -> Path:
    """查找 GADBench 图数据根目录。"""

    root = get_data_root(data_root)
    candidates = [
        root / "raw" / "GADBench",
        root / "graph",
        root / "raw" / "GADBench" / "datasets",
        root.parent / "repos" / "GADBench-master" / "datasets",
    ]
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "GADBench root not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )
    return found


def _to_numpy(value):
    """把 torch / numpy / 类数组对象统一转成 numpy 数组。"""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _get_node_data(graph, candidates):
    """按候选字段名从 DGL graph.ndata 中提取节点数据。"""

    for key in candidates:
        if key in graph.ndata:
            return graph.ndata[key], key
    raise KeyError(f"None of node data keys exist: {candidates}. Available: {list(graph.ndata.keys())}")


def _make_stratified_masks(
    y: np.ndarray,
    test_size: float = 0.3,
    val_size: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """当图文件没有预设 mask 时，生成可复现的分层节点 mask。"""

    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_val = int(round(len(idx) * val_size))
        if len(idx) > 1:
            n_test = min(max(n_test, 1), len(idx) - 1)
        if len(idx) - n_test > 1:
            n_val = min(max(n_val, 1), len(idx) - n_test - 1)
        else:
            n_val = 0
        test_mask[idx[:n_test]] = True
        val_mask[idx[n_test:n_test + n_val]] = True
        train_mask[idx[n_test + n_val:]] = True
    return {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}


def resolve_graph_file(name: str, data_root: Optional[str | Path] = None) -> Path:
    """根据数据集名称定位 GADBench 图文件，兼容无扩展名原始文件。"""

    key = name.strip().lower().replace("_", "-")
    canonical = GADBENCH_DATASETS.get(key, key)
    root = find_gadbench_root(data_root)
    direct_candidates = [
        root / canonical,
        root / canonical.replace("-", ""),
        root / "datasets" / canonical,
        root / "datasets" / canonical.replace("-", ""),
        root / "extracted" / "datasets" / canonical,
        root / "extracted" / "datasets" / canonical.replace("-", ""),
    ]
    found = first_existing(direct_candidates)
    if found is not None and found.is_file():
        return found

    patterns = [
        f"*{canonical}*.bin",
        f"*{canonical}*.pt",
        f"*{canonical}*.pkl",
        f"*{canonical}*.dgl",
        f"*{canonical.replace('-', '')}*.bin",
        f"*{canonical.replace('-', '')}*.pt",
        f"*{canonical.replace('-', '')}*.pkl",
        f"*{canonical.replace('-', '')}*.dgl",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"GADBench file for {name} not found under {root}. "
            "The graph data package may still need manual download."
        )
    return sorted(matches)[0]


def _load_from_npz(name: str, data_root: Optional[str | Path] = None) -> Optional[DatasetBundle]:
    """无 DGL 时的回退：从 export_graph_npz.py 预导出的 .npz 直接读特征矩阵。

    只用于通用算法（Exp-2 / Exp-3）跑图模态 —— 它们只需要节点特征矩阵四元组，
    不需要 DGL 图对象。图专用算法（DOMINANT/CoLA/GCN…）仍需真正的图对象，
    npz 回退不覆盖它们（extras 里没有 'graph'，专用算法会自然报错跳过）。

    查找路径：<data_root>/graph_npz/<name>.npz 以及若干常见位置。
    找不到返回 None，由调用方决定是否继续走 DGL 路径。
    """
    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root) / "graph_npz" / f"{name}.npz")
    try:
        root = get_data_root(data_root)
        candidates.append(root / "graph_npz" / f"{name}.npz")
        candidates.append(root / "graph" / "npz" / f"{name}.npz")
    except Exception:
        pass
    # 项目根 data/graph_npz/（export 脚本的默认输出位置）
    candidates.append(Path(__file__).resolve().parent.parent / "data" / "graph_npz" / f"{name}.npz")

    npz_path = first_existing(candidates)
    if npz_path is None or not npz_path.is_file():
        return None

    data = np.load(npz_path)
    X_train = data["X_train"].astype(np.float64)
    X_test = data["X_test"].astype(np.float64)
    y_train = data["y_train"].astype(int).reshape(-1)
    y_test = data["y_test"].astype(int).reshape(-1)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    metadata = {
        "source": "GADBench_npz_export",
        "path": str(npz_path),
        "n_nodes": int(X_all.shape[0]),
        "n_features": int(X_all.shape[1]) if X_all.ndim > 1 else 1,
        "n_anomalies": int(np.sum(y_all == 1)),
        "anomaly_rate": float(np.mean(y_all == 1)) if len(y_all) else 0.0,
        "split": "from_npz_export",
        "note": "loaded from pre-exported npz (no DGL); graph object unavailable",
    }
    return DatasetBundle(
        name=name,
        modality="graph",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        extras={"X_all": X_all, "y_all": y_all},
        metadata=metadata,
    )


def load_gadbench_dataset(
    name: str,
    data_root: Optional[str | Path] = None,
    standardize: bool = True,
    seed: int = DEFAULT_SEED,
) -> DatasetBundle:
    """使用 DGL 加载一个 GADBench 图数据集。

    返回按 mask 划分后的节点特征四元组，同时在 `extras` 中保留原始图对象、
    mask、全量特征和标签，方便图模型直接使用。

    若 DGL 不可用（如机器 A 的 torch 2.7 装不了 DGL），自动回退到读取
    scripts/export_graph_npz.py 预导出的 .npz（仅含特征矩阵，供通用算法使用）。
    """

    try:
        from dgl import load_graphs
    except ImportError as exc:
        # DGL 不可用 → 尝试 npz 回退（通用算法够用）
        bundle = _load_from_npz(name, data_root)
        if bundle is not None:
            return bundle
        raise ImportError(
            "DGL is required for GADBench loading, and no pre-exported npz was "
            f"found for '{name}'. 在有 DGL 的机器上运行 "
            "`python -m scripts.export_graph_npz` 生成 npz 并拷到本机 "
            "data/graph_npz/ 后即可在无 DGL 环境跑图模态通用算法。"
        ) from exc

    path = resolve_graph_file(name, data_root)
    graphs, aux = load_graphs(str(path))
    if not graphs:
        raise ValueError(f"No graph found in {path}")
    graph = graphs[0]

    feat, feat_key = _get_node_data(graph, ["feature", "feat", "features", "x"])
    label, label_key = _get_node_data(graph, ["label", "labels", "y"])
    X = _to_numpy(feat).astype(np.float64)
    y = _to_numpy(label).astype(int).reshape(-1)

    masks: Dict[str, np.ndarray] = {}
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if mask_name in graph.ndata:
            masks[mask_name] = _to_numpy(graph.ndata[mask_name]).astype(bool)
        elif mask_name in aux:
            masks[mask_name] = _to_numpy(aux[mask_name]).astype(bool)
    split = "predefined_masks"
    if "train_mask" not in masks or "test_mask" not in masks:
        masks.update(_make_stratified_masks(y, seed=seed))
        split = "generated_stratified_node_masks"

    X_train = X[masks["train_mask"]]
    X_test = X[masks["test_mask"]]
    y_train = y[masks["train_mask"]]
    y_test = y[masks["test_mask"]]
    preprocessing = {"standardization": "skipped"}
    if standardize:
        X_train, X_test, preprocessing = standardize_train_test(X_train, X_test)
        X_scaled = X.copy()
        X_scaled[masks["train_mask"]] = X_train
        X_scaled[masks["test_mask"]] = X_test
        if "val_mask" in masks and np.any(masks["val_mask"]):
            mean = np.mean(X[masks["train_mask"]], axis=0)
            scale = np.std(X[masks["train_mask"]], axis=0)
            scale = np.where(scale < 1e-12, 1.0, scale)
            X_scaled[masks["val_mask"]] = (X[masks["val_mask"]] - mean) / scale
    else:
        X_scaled = X

    # Keep the raw graph object consistent with the matrix view so graph models
    # use the same train-only standardization and generated masks.
    try:
        import torch

        feat_dtype = feat.dtype if isinstance(getattr(feat, "dtype", None), torch.dtype) else torch.float32
        feat_device = getattr(feat, "device", None)
        graph.ndata[feat_key] = torch.as_tensor(X_scaled, dtype=feat_dtype, device=feat_device)
        if feat_key != "feature":
            graph.ndata["feature"] = graph.ndata[feat_key]
        for mask_name, mask in masks.items():
            graph.ndata[mask_name] = torch.as_tensor(mask, dtype=torch.bool, device=feat_device)
    except Exception:
        pass

    metadata = {
        "source": "GADBench",
        "path": str(path),
        "feature_key": feat_key,
        "label_key": label_key,
        "n_nodes": int(X.shape[0]),
        "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
        "n_anomalies": int(np.sum(y == 1)),
        "anomaly_rate": float(np.mean(y == 1)) if len(y) else 0.0,
        "split": split,
        "seed": int(seed) if split.startswith("generated") else None,
        "preprocessing": preprocessing,
    }
    return DatasetBundle(
        name=name,
        modality="graph",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        extras={"graph": graph, "masks": masks, "X_all": X_scaled, "y_all": y},
        metadata=metadata,
    )
