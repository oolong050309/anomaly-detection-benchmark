"""Exp-2: training-set contamination robustness.

The experiment uses the same contamination grid for three data shapes:
tabular feature matrices, time-series windows, and graph node features.

For unsupervised detectors, the training matrix is rebuilt so the retained
anomaly ratio is approximately 0% / 1% / 5% / 10% / 20%.
For supervised detectors, features are kept fixed and labels are flipped
symmetrically at the same rates. Each completed run saves CSV metrics plus the
raw score/label artifact needed for later plotting and error analysis.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset
from data.contaminate import contaminate_supervised, contaminate_unsupervised
from experiments.common import fit_and_score, record_failure, record_success

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp2_results.csv")
OUTPUT_DIR = str(ROOT / "results")

CONTAMINATION_RATES = [0.0, 0.01, 0.05, 0.10, 0.20]

DEFAULT_DATASETS = {
    "tabular": ["cardio", "thyroid", "satellite", "pima", "annthyroid"],
    "timeseries": ["276_IOPS_id_17_WebService", "550_SWaT_id_1_Sensor"],
    "graph": ["reddit", "amazon"],
}


def _get_common_algos():
    """Common detectors that accept a 2D feature matrix."""
    from models import (
        AutoEncoderDetector,
        COPODDetector,
        DeepSVDDDetector,
        ECODDetector,
        IForestDetector,
        IQRDetector,
        KNNDetector,
        LightGBMDetector,
        LOFDetector,
        LogisticRegressionDetector,
        MLPDetector,
        OCSVMDetector,
        RandomForestDetector,
        TabPFNDetector,
        XGBoostDetector,
    )

    return [
        ("IQR", IQRDetector, {}, False),
        ("LOF", LOFDetector, {}, False),
        ("KNN", KNNDetector, {}, False),
        ("IForest", IForestDetector, {}, False),
        ("ECOD", ECODDetector, {}, False),
        ("COPOD", COPODDetector, {}, False),
        ("OCSVM", OCSVMDetector, {}, False),
        ("AutoEncoder", AutoEncoderDetector, {"epoch_num": 10}, False),
        ("DeepSVDD", DeepSVDDDetector, {"epochs": 10}, False),
        ("LR", LogisticRegressionDetector, {}, True),
        ("RF", RandomForestDetector, {"n_estimators": 100}, True),
        ("MLP", MLPDetector, {"max_iter": 200}, True),
        ("XGBoost", XGBoostDetector, {"n_estimators": 50}, True),
        ("LightGBM", LightGBMDetector, {"n_estimators": 50}, True),
        ("TabPFN", TabPFNDetector, {}, True),
    ]


def _load_as_2d(modality: str, ds_name: str, data_root=None):
    if modality == "tabular":
        bundle = load_dataset(ds_name, data_root=data_root)
    elif modality == "timeseries":
        bundle = load_dataset(
            ds_name,
            modality="timeseries",
            data_root=data_root,
            window_size=100,
            stride=10,
        )
    elif modality == "graph":
        bundle = load_dataset(ds_name, modality="graph", data_root=data_root)
    else:
        raise ValueError(f"Unsupported modality for Exp-2: {modality}")
    return bundle, bundle.as_tuple()


def _run_one(
    *,
    modality: str,
    dataset_name: str,
    algo_name: str,
    cls,
    kwargs: dict,
    needs_y: bool,
    rate: float,
    X_train,
    y_train,
    X_test,
    y_test,
) -> bool:
    mode = "supervised_label_flip" if needs_y else "unsupervised_anomaly_keep_ratio"
    try:
        if needs_y:
            X_fit, y_fit, meta = contaminate_supervised(X_train, y_train, rate, seed=SEED)
            if len(np.unique(y_fit)) < 2:
                raise RuntimeError("contaminated training labels contain a single class")
            fit_args = (X_fit, y_fit)
            label_flip_rate = rate
            train_for_log = y_fit
        else:
            X_fit, y_fit, meta = contaminate_unsupervised(X_train, y_train, rate, seed=SEED)
            fit_args = (X_fit,)
            label_flip_rate = None
            train_for_log = y_fit

        det = cls(contamination=0.1, random_state=SEED, **kwargs)
        scores, fit_t, pred_t = fit_and_score(det, fit_args, X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = record_success(
                log_path=LOG_PATH,
                output_dir=OUTPUT_DIR,
                experiment_name="exp2",
                modality=modality,
                dataset_name=dataset_name,
                algorithm_name=algo_name,
                y_true=y_test,
                scores=scores,
                fit_time_sec=fit_t,
                predict_time_sec=pred_t,
                seed=SEED,
                fit_params=kwargs,
                notes=f"target_rate={rate:.2f}; meta={meta}",
                contamination_mode=mode,
                contamination_rate=rate if not needs_y else None,
                label_flip_rate=label_flip_rate,
                X_train=X_fit,
                y_train=train_for_log,
                extra_metadata={"contamination_meta": meta},
            )
        print(f"    [{algo_name:>12s}] rate={rate:.2f} AUC-ROC={m['auc_roc']:.4f}")
        return True
    except Exception as e:
        record_failure(
            log_path=LOG_PATH,
            experiment_name="exp2",
            modality=modality,
            dataset_name=dataset_name,
            algorithm_name=algo_name,
            seed=SEED,
            error=e,
            fit_params=kwargs,
            notes=f"target_rate={rate:.2f}",
            contamination_mode=mode,
            contamination_rate=rate if not needs_y else None,
            label_flip_rate=rate if needs_y else None,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test,
        )
        print(f"    [{algo_name:>12s}] rate={rate:.2f} FAILED: {e!r}"[:100])
        return False


def main():
    global SEED, LOG_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Exp-2: contamination robustness")
    parser.add_argument("--modalities", nargs="+", default=["tabular", "timeseries", "graph"],
                        choices=["tabular", "timeseries", "graph"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Override datasets for every selected modality.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_DIR = str(Path(args.output_dir))
    LOG_PATH = str(Path(args.log_path) if args.log_path else Path(OUTPUT_DIR) / "exp2_results.csv")

    algos = _get_common_algos()
    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    for modality in args.modalities:
        datasets = args.datasets or DEFAULT_DATASETS[modality]
        print(f"\n{'=' * 70}\nExp-2 {modality}: {len(datasets)} datasets\n{'=' * 70}")
        for ds in datasets:
            print(f"\n  --- {modality}/{ds} ---")
            try:
                bundle, (X_tr, X_te, y_tr, y_te) = _load_as_2d(modality, ds, args.data_root)
                print(
                    f"  Loaded: train={np.asarray(X_tr).shape}, test={np.asarray(X_te).shape}, "
                    f"train_anomaly_rate={np.mean(y_tr):.4f}, test_anomaly_rate={np.mean(y_te):.4f}"
                )
            except Exception as e:
                print(f"  [SKIP] Cannot load {modality}/{ds}: {e!r}")
                continue

            for rate in CONTAMINATION_RATES:
                print(f"  --- target contamination/flip rate={rate:.2f} ---")
                for name, cls, kwargs, needs_y in algos:
                    n_total += 1
                    ok = _run_one(
                        modality=bundle.modality if modality != "graph" else "graph",
                        dataset_name=ds,
                        algo_name=name,
                        cls=cls,
                        kwargs=kwargs,
                        needs_y=needs_y,
                        rate=rate,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_test=X_te,
                        y_test=y_te,
                    )
                    n_ok += int(ok)

    total_t = time.perf_counter() - total_t0
    print(f"\n{'=' * 70}")
    print(f"Exp-2 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"Artifacts: {Path(OUTPUT_DIR) / 'artifacts' / 'exp2'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
