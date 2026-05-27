"""Low-resource smoke test for Exp-1/Exp-2/Exp-3 pipelines.

This script is designed for small CPU-only servers. It does not try to validate
all 26 algorithms. Instead, it samples small slices from real datasets and runs
two cheap detectors (IQR + Logistic Regression) through the same logging and
artifact path used by the full experiments.

Example:
    python -m scripts.smoke_test_experiments \
      --data-root /root/autodl-tmp/final_project/data \
      --output-dir results/smoke_cpu \
      --run-analysis
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Avoid oversubscribing a 0.5-core CPU server during smoke tests.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset
from data.contaminate import contaminate_supervised, contaminate_unsupervised
from experiments.common import fit_and_score, record_failure, record_success
from models import IQRDetector, LogisticRegressionDetector


DEFAULT_DATASETS = {
    "tabular": "cardio",
    "cv": "cifar10_0",
    "nlp": "20news_0",
    "timeseries": "276_IOPS_id_17_WebService",
    "graph": "reddit",
}

ALGOS = [
    ("IQR", IQRDetector, {"aggregation": "max"}, False),
    ("LR", LogisticRegressionDetector, {"max_iter": 200, "solver": "liblinear"}, True),
]


def _take_stratified(X: Any, y: Any, max_samples: int, seed: int):
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).astype(int).reshape(-1)
    if len(y_arr) <= max_samples:
        return X_arr, y_arr
    rng = np.random.default_rng(seed)
    parts = []
    for label in np.unique(y_arr):
        idx = np.flatnonzero(y_arr == label)
        n_label = max(1, int(round(max_samples * len(idx) / len(y_arr))))
        n_label = min(n_label, len(idx))
        parts.append(rng.choice(idx, size=n_label, replace=False))
    chosen = np.concatenate(parts)
    if len(chosen) > max_samples:
        chosen = rng.choice(chosen, size=max_samples, replace=False)
    rng.shuffle(chosen)
    return X_arr[chosen], y_arr[chosen]


def _load_small(modality: str, dataset: str, data_root: str | None, max_train: int, max_test: int, seed: int):
    if modality == "timeseries":
        bundle = load_dataset(
            dataset,
            modality="timeseries",
            data_root=data_root,
            window_size=64,
            stride=64,
        )
    elif modality == "graph":
        bundle = load_dataset(dataset, modality="graph", data_root=data_root)
    else:
        bundle = load_dataset(dataset, data_root=data_root)
    X_train, X_test, y_train, y_test = bundle.as_tuple()
    X_train, y_train = _take_stratified(X_train, y_train, max_train, seed)
    X_test, y_test = _take_stratified(X_test, y_test, max_test, seed + 1)
    return bundle, X_train, X_test, y_train, y_test


def _run_detector(
    *,
    experiment_name: str,
    modality: str,
    dataset_name: str,
    algo_name: str,
    cls,
    kwargs: dict,
    needs_y: bool,
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir: Path,
    seed: int,
    notes: str = "",
    contamination_mode: str = "",
    contamination_rate: float | None = None,
    label_flip_rate: float | None = None,
) -> bool:
    log_path = output_dir / f"{experiment_name}_results.csv"
    try:
        if needs_y:
            if len(np.unique(y_train)) < 2:
                raise RuntimeError("training labels contain a single class")
            fit_args = (X_train, y_train)
        else:
            fit_args = (X_train,)
        det = cls(contamination=0.1, random_state=seed, **kwargs)
        scores, fit_t, pred_t = fit_and_score(det, fit_args, X_test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = record_success(
                log_path=log_path,
                output_dir=output_dir,
                experiment_name=experiment_name,
                modality=modality,
                dataset_name=dataset_name,
                algorithm_name=algo_name,
                y_true=y_test,
                scores=scores,
                fit_time_sec=fit_t,
                predict_time_sec=pred_t,
                seed=seed,
                fit_params=kwargs,
                notes=notes,
                contamination_mode=contamination_mode,
                contamination_rate=contamination_rate,
                label_flip_rate=label_flip_rate,
                X_train=X_train,
                y_train=y_train,
            )
        print(f"    [{experiment_name}/{modality}/{algo_name}] AUC={metrics['auc_roc']:.4f} fit={fit_t:.2f}s")
        return True
    except Exception as exc:
        record_failure(
            log_path=log_path,
            experiment_name=experiment_name,
            modality=modality,
            dataset_name=dataset_name,
            algorithm_name=algo_name,
            seed=seed,
            error=exc,
            fit_params=kwargs,
            notes=notes,
            contamination_mode=contamination_mode,
            contamination_rate=contamination_rate,
            label_flip_rate=label_flip_rate,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test,
        )
        print(f"    [{experiment_name}/{modality}/{algo_name}] FAILED: {exc!r}"[:140])
        return False


def smoke_exp1(args) -> tuple[int, int]:
    print("\n=== Smoke Exp-1: baseline path ===")
    ok = total = 0
    for modality in args.modalities:
        dataset = args.datasets.get(modality, DEFAULT_DATASETS[modality])
        try:
            _, X_train, X_test, y_train, y_test = _load_small(
                modality, dataset, args.data_root, args.max_train, args.max_test, args.seed
            )
            print(f"  Loaded {modality}/{dataset}: train={X_train.shape}, test={X_test.shape}")
        except Exception as exc:
            print(f"  [SKIP] {modality}/{dataset}: {exc!r}")
            continue
        for name, cls, kwargs, needs_y in ALGOS:
            total += 1
            ok += int(_run_detector(
                experiment_name="exp1",
                modality=modality,
                dataset_name=f"{modality}/{dataset}",
                algo_name=name,
                cls=cls,
                kwargs=kwargs,
                needs_y=needs_y,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                output_dir=args.output_dir,
                seed=args.seed,
                notes="low_resource_smoke",
            ))
    return ok, total


def smoke_exp2(args) -> tuple[int, int]:
    print("\n=== Smoke Exp-2: contamination path ===")
    ok = total = 0
    for modality in [m for m in args.modalities if m in {"tabular", "timeseries", "graph"}]:
        dataset = args.datasets.get(modality, DEFAULT_DATASETS[modality])
        try:
            _, X_train, X_test, y_train, y_test = _load_small(
                modality, dataset, args.data_root, args.max_train, args.max_test, args.seed
            )
            print(f"  Loaded {modality}/{dataset}: train={X_train.shape}, test={X_test.shape}")
        except Exception as exc:
            print(f"  [SKIP] {modality}/{dataset}: {exc!r}")
            continue
        for rate in args.rates:
            for name, cls, kwargs, needs_y in ALGOS:
                total += 1
                if needs_y:
                    X_fit, y_fit, _ = contaminate_supervised(X_train, y_train, rate, seed=args.seed)
                    mode = "supervised_label_flip"
                    c_rate = None
                    flip_rate = rate
                else:
                    X_fit, y_fit, _ = contaminate_unsupervised(X_train, y_train, rate, seed=args.seed)
                    mode = "unsupervised_anomaly_keep_ratio"
                    c_rate = rate
                    flip_rate = None
                ok += int(_run_detector(
                    experiment_name="exp2",
                    modality=modality,
                    dataset_name=f"{modality}/{dataset}",
                    algo_name=name,
                    cls=cls,
                    kwargs=kwargs,
                    needs_y=needs_y,
                    X_train=X_fit,
                    X_test=X_test,
                    y_train=y_fit,
                    y_test=y_test,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    notes=f"low_resource_smoke_rate={rate}",
                    contamination_mode=mode,
                    contamination_rate=c_rate,
                    label_flip_rate=flip_rate,
                ))
    return ok, total


def smoke_exp3(args) -> tuple[int, int]:
    print("\n=== Smoke Exp-3: cross-modal path ===")
    ok = total = 0
    for modality in args.modalities:
        dataset = args.datasets.get(modality, DEFAULT_DATASETS[modality])
        try:
            _, X_train, X_test, y_train, y_test = _load_small(
                modality, dataset, args.data_root, args.max_train, args.max_test, args.seed
            )
            print(f"  Loaded {modality}/{dataset}: train={X_train.shape}, test={X_test.shape}")
        except Exception as exc:
            print(f"  [SKIP] {modality}/{dataset}: {exc!r}")
            continue
        for name, cls, kwargs, needs_y in ALGOS:
            total += 1
            ok += int(_run_detector(
                experiment_name="exp3",
                modality=modality,
                dataset_name=f"{modality}/{dataset}",
                algo_name=name,
                cls=cls,
                kwargs=kwargs,
                needs_y=needs_y,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                output_dir=args.output_dir,
                seed=args.seed,
                notes="low_resource_smoke",
            ))
    return ok, total


def _parse_dataset_overrides(values: list[str] | None) -> dict[str, str]:
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Dataset override must be modality=name, got {value!r}")
        modality, dataset = value.split("=", 1)
        overrides[modality.strip()] = dataset.strip()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-resource smoke test for all experiments.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "smoke_cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=160)
    parser.add_argument("--max-test", type=int, default=120)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--modalities", nargs="+",
                        default=["tabular", "timeseries", "graph"],
                        choices=["tabular", "cv", "nlp", "timeseries", "graph"])
    parser.add_argument("--dataset", action="append",
                        help="Override dataset as modality=name, e.g. --dataset tabular=thyroid")
    parser.add_argument("--run-analysis", action="store_true",
                        help="Also run plotting smoke test into <output-dir>/figures.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any attempted smoke run fails.")
    ns = parser.parse_args()
    ns.output_dir = Path(ns.output_dir)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    ns.datasets = _parse_dataset_overrides(ns.dataset)

    t0 = time.perf_counter()
    ok1, total1 = smoke_exp1(ns)
    ok2, total2 = smoke_exp2(ns)
    ok3, total3 = smoke_exp3(ns)
    total_ok = ok1 + ok2 + ok3
    total = total1 + total2 + total3

    if ns.run_analysis:
        from eval.visualize import generate_all

        figure_dir = ns.output_dir / "figures"
        outputs = generate_all(ns.output_dir, figure_dir, metric="auc_roc")
        print(f"\nAnalysis smoke generated {len(outputs)} artifacts under {figure_dir}")

    elapsed = time.perf_counter() - t0
    print("\n=== Smoke summary ===")
    print(f"Exp-1: {ok1}/{total1} succeeded")
    print(f"Exp-2: {ok2}/{total2} succeeded")
    print(f"Exp-3: {ok3}/{total3} succeeded")
    print(f"Total: {total_ok}/{total} succeeded in {elapsed:.1f}s")
    print(f"Output directory: {ns.output_dir}")

    if total == 0:
        raise SystemExit("No smoke runs were attempted; check --data-root and dataset names.")
    if ns.strict and total_ok != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
