"""TSB-AD 单变量时序数据适配器。

负责读取 TSB-AD 的 CSV 文件，解析文件名中的 `tr_` 训练截止点，只用训练段
统计量做 z-score，并将原始序列切成可供算法训练的滑窗样本。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .common import DEFAULT_SEED, DatasetBundle, first_existing, get_data_root, summarize_array


TR_PATTERN = re.compile(r"_tr_(\d+)_")


def find_tsb_root(data_root: Optional[str | Path] = None) -> Path:
    """查找 TSB-AD 数据根目录，兼容服务器和本地多种目录布局。"""

    root = get_data_root(data_root)
    candidates = [
        root / "raw" / "TSB-AD-U" / "TSB-AD-U",
        root / "raw" / "TSB-AD-U",
        root / "TSB-AD-U",
        root.parent / "repos" / "TSB-AD-main" / "Datasets",
    ]
    for candidate in candidates:
        if candidate.exists():
            csvs = list(candidate.rglob("*.csv"))
            if csvs:
                return candidate
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "TSB-AD root not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )
    return found


def parse_train_end(path: str | Path) -> int:
    """从文件名中解析 `tr_数字` 形式的训练截止索引。"""

    match = TR_PATTERN.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse train cutoff from filename: {path}")
    return int(match.group(1))


def list_tsb_files(data_root: Optional[str | Path] = None) -> List[Path]:
    """列出可用的 TSB-AD CSV 文件。"""

    return sorted(find_tsb_root(data_root).rglob("*.csv"))


def resolve_tsb_file(name_or_path: str | Path, data_root: Optional[str | Path] = None) -> Path:
    """根据完整路径、文件名或文件名片段定位单条时序 CSV。"""

    path = Path(name_or_path)
    if path.exists():
        return path
    root = find_tsb_root(data_root)
    matches = [p for p in root.rglob("*.csv") if path.name == p.name or str(name_or_path) in p.name]
    if not matches:
        raise FileNotFoundError(f"No TSB-AD CSV matched: {name_or_path}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous TSB-AD name {name_or_path}: {[p.name for p in matches[:5]]}")
    return matches[0]


def read_tsb_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取 TSB-AD CSV，返回数值序列和 0/1 标签。"""

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        value_col = "Data" if "Data" in reader.fieldnames else reader.fieldnames[0]
        label_col = "Label" if "Label" in reader.fieldnames else reader.fieldnames[-1]
        values = []
        labels = []
        for row in reader:
            values.append(float(row[value_col]))
            labels.append(int(float(row[label_col])))
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=int)


def make_windows(
    values: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    label_rule: str = "any",
) -> Tuple[np.ndarray, np.ndarray]:
    """把一维序列切成滑窗样本，并按指定规则生成窗口标签。"""

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    X = []
    y = []
    for start in range(0, len(values) - window_size + 1, stride):
        end = start + window_size
        window_labels = labels[start:end]
        X.append(values[start:end])
        if label_rule == "any":
            y.append(int(np.any(window_labels == 1)))
        elif label_rule == "last":
            y.append(int(window_labels[-1]))
        elif label_rule == "majority":
            y.append(int(np.mean(window_labels == 1) >= 0.5))
        else:
            raise ValueError(f"Unknown label_rule: {label_rule}")
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)


def zscore_by_train(values: np.ndarray, train_end: int, eps: float = 1e-12):
    """使用训练段均值和标准差标准化整条序列。"""

    train_values = values[:train_end]
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if std < eps:
        std = 1.0
    return (values - mean) / std, {"mean": mean, "std": std, "standardization": "zscore_train_segment"}


def load_tsb_dataset(
    name_or_path: str | Path,
    data_root: Optional[str | Path] = None,
    window_size: int = 100,
    stride: int = 10,
    label_rule: str = "any",
    standardize: bool = True,
    seed: int = DEFAULT_SEED,
) -> DatasetBundle:
    """加载单条 TSB-AD 序列，并按文件名中的 `tr_` 截止点划分。"""

    del seed  # 时序划分由 benchmark 文件名决定，不使用随机切分。
    path = resolve_tsb_file(name_or_path, data_root)
    values, labels = read_tsb_csv(path)
    train_end = parse_train_end(path)
    train_end = min(max(train_end, 1), len(values) - 1)

    preprocessing = {"standardization": "skipped"}
    if standardize:
        values, preprocessing = zscore_by_train(values, train_end)

    X_train, y_train = make_windows(values[:train_end], labels[:train_end], window_size, stride, label_rule)
    X_test, y_test = make_windows(values[train_end:], labels[train_end:], window_size, stride, label_rule)

    metadata = {
        "source": "TSB-AD-U",
        "path": str(path),
        "train_end": int(train_end),
        "window_size": int(window_size),
        "stride": int(stride),
        "label_rule": label_rule,
        "split": "filename_tr_cutoff",
        "preprocessing": preprocessing,
        **summarize_array(values.reshape(-1, 1), labels),
    }
    return DatasetBundle(
        name=path.stem,
        modality="timeseries",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        extras={"raw_values": values, "raw_labels": labels, "train_end": train_end},
        metadata=metadata,
    )


def select_representative_tsb_files(
    data_root: Optional[str | Path] = None,
    domains: Optional[Iterable[str]] = None,
    max_files: int = 8,
) -> List[Path]:
    """按领域关键词确定性选择代表性时序文件。"""

    domain_list = list(domains or ["Facility", "Medical", "Sensor", "Web", "Finance", "Traffic"])
    files = list_tsb_files(data_root)
    selected: List[Path] = []
    for domain in domain_list:
        matches = [p for p in files if f"_{domain}_" in p.name or domain.lower() in p.name.lower()]
        for match in matches[:2]:
            if match not in selected:
                selected.append(match)
            if len(selected) >= max_files:
                return selected
    for path in files:
        if path not in selected:
            selected.append(path)
        if len(selected) >= max_files:
            break
    return selected
