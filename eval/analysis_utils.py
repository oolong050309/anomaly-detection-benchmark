"""Utilities shared by result analysis and plotting code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "auc_roc",
    "auc_pr",
    "f1_best",
    "fit_time_sec",
    "predict_time_sec",
    "best_threshold",
    "contamination_rate",
    "label_flip_rate",
    "train_anomaly_rate",
    "test_anomaly_rate",
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read one or more experiment CSV files and normalize common columns."""

    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            continue
        frame = pd.read_csv(p)
        frame["source_csv"] = str(p)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" not in df.columns:
        df["status"] = "success"
    if "experiment_name" not in df.columns:
        df["experiment_name"] = ""
    if "modality" not in df.columns:
        df["modality"] = ""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["timestamp"] = pd.NaT
    return df


def read_experiment(results_dir: str | Path, exp_name: str) -> pd.DataFrame:
    results = Path(results_dir)
    candidates = [
        results / f"{exp_name}_results.csv",
        results / "experiment_log.csv",
    ]
    df = read_results(candidates)
    if df.empty:
        return df
    named = df[df["experiment_name"].fillna("").eq(exp_name)]
    return named.copy() if not named.empty else df.copy()


def successful_runs(df: pd.DataFrame, metric: str = "auc_roc") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[df["status"].fillna("success").eq("success")].copy()
    if metric in out.columns:
        out = out[np.isfinite(out[metric])]
    return out


def latest_per_run_key(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Deduplicate repeated reruns by keeping the latest timestamp per key."""

    if df.empty:
        return df.copy()
    cols = [k for k in keys if k in df.columns]
    if not cols:
        return df.copy()
    order_col = "timestamp" if "timestamp" in df.columns else None
    if order_col:
        ordered = df.sort_values(order_col)
    else:
        ordered = df.copy()
    return ordered.drop_duplicates(cols, keep="last")


def load_artifact_npz(row: pd.Series) -> dict[str, np.ndarray] | None:
    path = row.get("artifact_path", "")
    if not isinstance(path, str) or not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def parse_fit_params(value) -> dict:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def parameter_proxy(row: pd.Series) -> float:
    """A coarse size proxy for efficiency plots when exact parameters are absent."""

    params = parse_fit_params(row.get("fit_params", ""))
    algo = str(row.get("algorithm_name", "")).lower()
    if "n_estimators" in params:
        return float(params.get("n_estimators") or 1)
    if "num_kernels" in params:
        return float(params.get("num_kernels") or 1)
    if "hidden_size" in params:
        return float(params.get("hidden_size") or 1) * float(params.get("epochs", 1) or 1)
    if "epoch" in params:
        return float(params.get("epoch") or 1)
    if any(token in algo for token in ["autoencoder", "svdd", "lstm", "dominant", "cola", "gcn", "bwg", "unprompt"]):
        return 100.0
    if any(token in algo for token in ["rf", "forest", "xgboost", "lightgbm", "xgb"]):
        return 50.0
    return 1.0


def write_table_csv(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    return out
