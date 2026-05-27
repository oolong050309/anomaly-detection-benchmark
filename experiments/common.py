"""Shared experiment utilities.

The experiment scripts intentionally keep CSV rows compact while saving the
heavier per-run evidence (scores, labels, metadata) as compressed artifacts.
Those artifacts make later threshold analysis, error analysis, and plotting
possible without rerunning model training.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from eval.logger import log_experiment
from eval.metrics import evaluate_all


def as_json(value: Any) -> str:
    """Serialize small metadata values with stable key ordering."""

    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=default)


def shape0(x: Any) -> int | None:
    try:
        return int(np.asarray(x).shape[0])
    except Exception:
        return None


def anomaly_rate(y: Any) -> float | None:
    if y is None:
        return None
    arr = np.asarray(y).reshape(-1)
    if arr.size == 0:
        return None
    return float(np.mean(arr.astype(int) == 1))


def make_run_id(experiment_name: str, dataset_name: str, algorithm_name: str) -> str:
    safe = "_".join(
        part.replace("/", "-").replace(" ", "_")
        for part in (experiment_name, dataset_name, algorithm_name)
    )
    return f"{safe}_{uuid.uuid4().hex[:10]}"


def save_run_artifact(
    *,
    output_dir: str | Path,
    experiment_name: str,
    run_id: str,
    y_true: Any | None = None,
    scores: Any | None = None,
    y_train: Any | None = None,
    test_index: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save labels/scores and JSON metadata for a completed run."""

    artifact_dir = Path(output_dir) / "artifacts" / experiment_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz_path = artifact_dir / f"{run_id}.npz"
    arrays: dict[str, Any] = {}
    if y_true is not None:
        arrays["y_true"] = np.asarray(y_true)
    if scores is not None:
        arrays["scores"] = np.asarray(scores, dtype=np.float64)
    if y_train is not None:
        arrays["y_train"] = np.asarray(y_train)
    if test_index is not None:
        arrays["test_index"] = np.asarray(test_index)
    if arrays:
        np.savez_compressed(npz_path, **arrays)
    else:
        np.savez_compressed(npz_path, empty=np.asarray([], dtype=np.float64))

    meta_path = artifact_dir / f"{run_id}.json"
    meta = dict(metadata or {})
    meta["npz_path"] = str(npz_path)
    meta_path.write_text(as_json(meta) + "\n", encoding="utf-8")
    return str(npz_path)


def record_success(
    *,
    log_path: str | Path,
    output_dir: str | Path,
    experiment_name: str,
    modality: str,
    dataset_name: str,
    algorithm_name: str,
    y_true: Any,
    scores: Any,
    fit_time_sec: float,
    predict_time_sec: float,
    seed: int,
    fit_params: dict[str, Any] | None = None,
    notes: str = "",
    contamination_mode: str = "",
    contamination_rate: float | None = None,
    label_flip_rate: float | None = None,
    X_train: Any | None = None,
    y_train: Any | None = None,
    test_index: Any | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    metrics = evaluate_all(y_true, scores)
    run_id = make_run_id(experiment_name, dataset_name, algorithm_name)
    artifact_path = save_run_artifact(
        output_dir=output_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        y_true=y_true,
        scores=scores,
        y_train=y_train,
        test_index=test_index,
        metadata={
            "run_id": run_id,
            "experiment_name": experiment_name,
            "modality": modality,
            "dataset_name": dataset_name,
            "algorithm_name": algorithm_name,
            "seed": seed,
            "fit_params": fit_params or {},
            "metrics": metrics,
            "notes": notes,
            "contamination_mode": contamination_mode,
            "contamination_rate": contamination_rate,
            "label_flip_rate": label_flip_rate,
            "extra": extra_metadata or {},
        },
    )
    log_experiment(
        dataset_name,
        algorithm_name,
        metrics["auc_roc"],
        metrics["auc_pr"],
        metrics["f1_best"],
        fit_time_sec,
        predict_time_sec,
        seed,
        notes=notes,
        log_path=str(log_path),
        experiment_name=experiment_name,
        modality=modality,
        status="success",
        best_threshold=metrics["best_threshold"],
        contamination_mode=contamination_mode,
        contamination_rate=contamination_rate,
        label_flip_rate=label_flip_rate,
        n_train=shape0(X_train),
        n_test=shape0(y_true),
        train_anomaly_rate=anomaly_rate(y_train),
        test_anomaly_rate=anomaly_rate(y_true),
        fit_params=as_json(fit_params or {}),
        artifact_path=artifact_path,
    )
    return metrics


def record_failure(
    *,
    log_path: str | Path,
    experiment_name: str,
    modality: str,
    dataset_name: str,
    algorithm_name: str,
    seed: int,
    error: BaseException,
    fit_params: dict[str, Any] | None = None,
    notes: str = "",
    contamination_mode: str = "",
    contamination_rate: float | None = None,
    label_flip_rate: float | None = None,
    X_train: Any | None = None,
    y_train: Any | None = None,
    y_test: Any | None = None,
) -> None:
    log_experiment(
        dataset_name,
        algorithm_name,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        seed,
        notes=notes,
        log_path=str(log_path),
        experiment_name=experiment_name,
        modality=modality,
        status="failed",
        contamination_mode=contamination_mode,
        contamination_rate=contamination_rate,
        label_flip_rate=label_flip_rate,
        n_train=shape0(X_train),
        n_test=shape0(y_test),
        train_anomaly_rate=anomaly_rate(y_train),
        test_anomaly_rate=anomaly_rate(y_test),
        fit_params=as_json(fit_params or {}),
        error_type=type(error).__name__,
        error_message=repr(error)[:500],
    )


def fit_and_score(detector: Any, fit_args: tuple[Any, ...], score_input: Any) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    detector.fit(*fit_args)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = np.asarray(detector.decision_function(score_input), dtype=np.float64).ravel()
    predict_time = time.perf_counter() - t0
    return scores, fit_time, predict_time
