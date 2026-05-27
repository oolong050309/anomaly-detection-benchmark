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

# 表头顺序固定；新字段只能追加到末尾，保证旧结果和旧测试仍可读取。
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
    "experiment_name",
    "modality",
    "status",
    "best_threshold",
    "contamination_mode",
    "contamination_rate",
    "label_flip_rate",
    "n_train",
    "n_test",
    "train_anomaly_rate",
    "test_anomaly_rate",
    "fit_params",
    "artifact_path",
    "error_type",
    "error_message",
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


def _ensure_current_header(log_path: str) -> bool:
    """Return whether the file already has data, upgrading old prefix headers."""

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return False
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False
    if rows[0] == HEADER:
        return True
    if rows[0] == HEADER[: len(rows[0])]:
        width = len(HEADER)
        upgraded = [HEADER]
        for row in rows[1:]:
            upgraded.append(row + [""] * max(0, width - len(row)))
        with open(log_path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(upgraded)
        return len(upgraded) > 1
    raise RuntimeError(
        f"CSV header at {log_path} is incompatible with eval.logger.HEADER"
    )


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
    experiment_name: str = "",
    modality: str = "",
    status: str = "success",
    best_threshold: float | None = None,
    contamination_mode: str = "",
    contamination_rate: float | None = None,
    label_flip_rate: float | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    train_anomaly_rate: float | None = None,
    test_anomaly_rate: float | None = None,
    fit_params: str = "",
    artifact_path: str = "",
    error_type: str = "",
    error_message: str = "",
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

    file_exists = _ensure_current_header(log_path)

    notes_clean = (notes or "").replace("\n", " ").replace("\r", " ").strip()
    error_clean = (error_message or "").replace("\n", " ").replace("\r", " ").strip()
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
        str(experiment_name),
        str(modality),
        str(status),
        _fmt_number(best_threshold),
        str(contamination_mode),
        _fmt_number(contamination_rate),
        _fmt_number(label_flip_rate),
        "" if n_train is None else str(int(n_train)),
        "" if n_test is None else str(int(n_test)),
        _fmt_number(train_anomaly_rate),
        _fmt_number(test_anomaly_rate),
        str(fit_params),
        str(artifact_path),
        str(error_type),
        error_clean,
    ]

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADER)
        writer.writerow(row)
        f.flush()
