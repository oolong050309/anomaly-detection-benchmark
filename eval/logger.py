"""实验日志写入工具。

每次算法跑完后调用 ``log_experiment`` 把结果以一行追加到
``results/experiment_log.csv``。表头由本模块控制，保证下游
``pandas.read_csv`` 后可直接消费。
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime, timezone
from typing import Any

# 表头顺序固定；新字段只能追加到末尾
HEADER = [
    "dataset_name",
    "algorithm_name",
    "auc_roc",
    "auc_pr",
    "f1_best",
    "fit_time_sec",
    "predict_time_sec",
    "seed",
    "timestamp",
    "notes",
]

DEFAULT_LOG_PATH = os.path.join("results", "experiment_log.csv")


def _fmt_number(v: Any) -> str:
    """把数值字段格式化成字符串。

    None / NaN -> "nan"，保留语义以便下游 ``pd.to_numeric(errors='coerce')``。
    """
    if v is None:
        return "nan"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(f) or math.isinf(f):
        return "nan"
    return f"{f:.6f}"


def _utc_iso8601() -> str:
    """ISO 8601 UTC 时间戳，精确到秒。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_experiment(
    dataset_name: str,
    algorithm_name: str,
    auc_roc: float | None,
    auc_pr: float | None,
    f1_best: float | None,
    fit_time_sec: float | None,
    predict_time_sec: float | None,
    seed: int | None,
    notes: str = "",
    *,
    log_path: str = DEFAULT_LOG_PATH,
) -> None:
    """把一次实验结果追加到 CSV。

    - 父目录不存在时自动创建。
    - 文件不存在时先写表头再写数据；存在则只追加数据。
    - 数值字段为 None / NaN 时写 ``"nan"``。
    - ``notes`` 内的换行会被替换成空格，避免破坏 CSV 行结构。
    """
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    notes_clean = (notes or "").replace("\n", " ").replace("\r", " ").strip()
    seed_str = "" if seed is None else str(seed)

    row = [
        str(dataset_name),
        str(algorithm_name),
        _fmt_number(auc_roc),
        _fmt_number(auc_pr),
        _fmt_number(f1_best),
        _fmt_number(fit_time_sec),
        _fmt_number(predict_time_sec),
        seed_str,
        _utc_iso8601(),
        notes_clean,
    ]

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADER)
        writer.writerow(row)
        f.flush()
