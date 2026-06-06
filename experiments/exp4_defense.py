"""Exp-4: Robust anomaly defense evaluation.

This experiment evaluates the "RobustDefenseWrapper" against label contamination (label flips).
It compares standard supervised detectors (LightGBM, XGBoost) against their defended
counterparts under 0%, 5%, 10%, and 20% training label flip rates.
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
from data.contaminate import contaminate_supervised
from experiments.common import (
    anomaly_rate,
    as_json,
    fit_and_score,
    record_failure,
    record_success,
    shape0,
)
from experiments.exp1_baseline import _borrow_for_supervised
from eval.logger import log_experiment
from models import (
    LightGBMDetector,
    XGBoostDetector,
    IForestDetector,
    RobustDefenseWrapper,
)

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp4_results.csv")
OUTPUT_DIR = str(ROOT / "results")

# We run on 0%, 5%, 10%, 20% training label flip rates
FLIP_RATES = [0.0, 0.05, 0.10, 0.20]

# Use a representative subset of datasets for fast evaluation
DEFAULT_DATASETS = {
    "tabular": [
        "cardio", "thyroid", "satellite", "shuttle", "credit_card",
        "pima", "annthyroid", "mammography", "pendigits",
    ],
    "timeseries": [
        "006_NAB_id_6_Traffic",
        "149_Stock_id_1_Finance",
        "171_MITDB_id_2_Medical",
        "225_MGAB_id_1_Synthetic",
        "276_IOPS_id_17_WebService",
        "331_UCR_id_29_Facility",
        "337_UCR_id_35_HumanActivity",
        "550_SWaT_id_1_Sensor",
    ],
    "graph": ["tfinance", "reddit", "amazon", "weibo"],
    "cv": ["cifar10_0", "cifar10_1", "fashionmnist_0", "fashionmnist_1"],
    "nlp": ["20news_0", "20news_1", "agnews_0", "amazon"],
}


def _get_defense_algos():
    """Generates a list of (Name, Class_Constructor, Kwargs) for all universal supervised detectors,
    both in undefended and defended (trim / flip) configurations.
    Total: 6 base models * 3 configurations = 18 setups.
    """
    from models import (
        LightGBMDetector,
        XGBoostDetector,
        RandomForestDetector,
        LogisticRegressionDetector,
        MLPDetector,
        TabPFNDetector,
    )
    
    base_models = [
        ("LightGBM", LightGBMDetector, {"n_estimators": 100}),
        ("XGBoost", XGBoostDetector, {"n_estimators": 100}),
        ("RF", RandomForestDetector, {"n_estimators": 100}),
        ("LR", LogisticRegressionDetector, {}),
        ("MLP", MLPDetector, {"max_iter": 200}),
        ("TabPFN", TabPFNDetector, {}),
    ]
    
    configs = []
    
    # Define our 9 universal unsupervised cleaners
    from models import (
        IQRDetector,
        LOFDetector,
        KNNDetector,
        IForestDetector,
        ECODDetector,
        COPODDetector,
        OCSVMDetector,
        AutoEncoderDetector,
        DeepSVDDDetector,
    )
    
    cleaners_spec = [
        ("IQR", IQRDetector, {}),
        ("LOF", LOFDetector, {}),
        ("KNN", KNNDetector, {}),
        ("IForest", IForestDetector, {}),
        ("ECOD", ECODDetector, {}),
        ("COPOD", COPODDetector, {}),
        ("OCSVM", OCSVMDetector, {}),
        ("AutoEncoder", AutoEncoderDetector, {"epoch_num": 100}),
        ("DeepSVDD", DeepSVDDDetector, {"epochs": 100}),
    ]
    
    for name, detector_cls, default_kwargs in base_models:
        # 1. Standard (Undefended)
        configs.append((f"Standard_{name}", detector_cls, default_kwargs))
        
        # 2 & 3. Defended configurations using ALL 9 unsupervised cleaners
        for cl_name, cl_cls, cl_kwargs in cleaners_spec:
            # TRIM Strategy
            configs.append((
                f"Defended_{name}_{cl_name}_Trim",
                # Freeze all variables using default arguments to prevent lambda closure late binding
                lambda detector_cls=detector_cls, default_kwargs=default_kwargs,
                       cl_cls=cl_cls, cl_kwargs=cl_kwargs, **kw: RobustDefenseWrapper(
                    base_detector=detector_cls(random_state=SEED, **default_kwargs),
                    unsupervised_cleaner=cl_cls(random_state=SEED, **cl_kwargs),
                    trim_rate=kw.get("trim_rate", 0.1),
                    strategy="trim",
                    random_state=SEED
                ),
                {}
            ))
            
            # FLIP Strategy
            configs.append((
                f"Defended_{name}_{cl_name}_Flip",
                # Freeze all variables using default arguments to prevent lambda closure late binding
                lambda detector_cls=detector_cls, default_kwargs=default_kwargs,
                       cl_cls=cl_cls, cl_kwargs=cl_kwargs, **kw: RobustDefenseWrapper(
                    base_detector=detector_cls(random_state=SEED, **default_kwargs),
                    unsupervised_cleaner=cl_cls(random_state=SEED, **cl_kwargs),
                    trim_rate=kw.get("trim_rate", 0.1),
                    strategy="flip",
                    random_state=SEED
                ),
                {}
            ))
        
    return configs


def _load_as_2d(modality: str, ds_name: str, data_root=None):
    if modality in ("tabular", "cv", "nlp"):
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
        raise ValueError(f"Unsupported modality for Exp-4: {modality}")
    return bundle, bundle.as_tuple()


def _run_one(
    *,
    modality: str,
    dataset_name: str,
    algo_name: str,
    cls,
    kwargs: dict,
    rate: float,
    X_train,
    y_train,
    X_test,
    y_test,
) -> bool:
    mode = "supervised_label_flip"
    try:
        # Inject the training label flip noise
        X_fit, y_fit, meta = contaminate_supervised(X_train, y_train, rate, seed=SEED)
        
        # Symmetrical Label Flip safety: borrow class samples if one class collapses
        if len(np.unique(y_fit)) < 2:
            X_fit_aug, y_fit_aug, _, _, _, borrow_note = _borrow_for_supervised(
                X_fit, y_fit, X_test, y_test, rng_seed=SEED
            )
            X_fit, y_fit = X_fit_aug, y_fit_aug
            meta["borrow_note"] = borrow_note

        # Instantiate model. For defended wrappers, we pass the current flip rate as trim_rate parameter
        if "Defended_" in algo_name:
            # We set trim_rate in defense wrapper to match the training contamination rate
            # In real-world scenarios, this could also be estimated, but in controlled benchmark we use the actual rate
            model_kwargs = {"trim_rate": rate if rate > 0 else 0.05} # fallback to small rate if rate is 0
            det = cls(**model_kwargs)
        else:
            det = cls(random_state=SEED, **kwargs)

        fit_args = (X_fit, y_fit)
        scores, fit_t, pred_t = fit_and_score(det, fit_args, X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = record_success(
                log_path=LOG_PATH,
                output_dir=OUTPUT_DIR,
                experiment_name="exp4",
                modality=modality,
                dataset_name=dataset_name,
                algorithm_name=algo_name,
                y_true=y_test,
                scores=scores,
                fit_time_sec=fit_t,
                predict_time_sec=pred_t,
                seed=SEED,
                fit_params=kwargs,
                notes=f"flip_rate={rate:.2f}; meta={meta}",
                contamination_mode=mode,
                contamination_rate=None,
                label_flip_rate=rate,
                X_train=X_fit,
                y_train=y_fit,
                extra_metadata={"contamination_meta": meta},
            )
        print(f"    [{algo_name:>23s}] rate={rate:.2f} AUC-ROC={m['auc_roc']:.4f}")
        return True
    except Exception as e:
        record_failure(
            log_path=LOG_PATH,
            experiment_name="exp4",
            modality=modality,
            dataset_name=dataset_name,
            algorithm_name=algo_name,
            seed=SEED,
            error=e,
            fit_params=kwargs,
            notes=f"flip_rate={rate:.2f}",
            contamination_mode=mode,
            contamination_rate=None,
            label_flip_rate=rate,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test,
        )
        print(f"    [{algo_name:>23s}] rate={rate:.2f} FAILED: {e!r}"[:110])
        return False


def main():
    global SEED, LOG_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Exp-4: robust anomaly defense evaluation")
    parser.add_argument("--modalities", nargs="+", default=["tabular", "timeseries", "graph", "cv", "nlp"],
                        choices=["tabular", "timeseries", "graph", "cv", "nlp"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Override datasets for selected modalities.")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestamped", action="store_true")
    args = parser.parse_args()

    SEED = args.seed
    OUTPUT_DIR = str(Path(args.output_dir))
    
    if args.timestamped:
        LOG_PATH = str(Path(OUTPUT_DIR) / f"exp4_results_seed{SEED}_{int(time.time())}.csv")
    else:
        LOG_PATH = str(Path(OUTPUT_DIR) / "exp4_results.csv")

    print("=" * 80)
    print(f"Executing Exp-4: Robust Anomaly Defense Evaluation (seed={SEED})")
    print(f"Logs will be written to: {LOG_PATH}")
    print("=" * 80)

    # Prepare directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    algos = _get_defense_algos()
    modalities = args.modalities

    for modality in modalities:
        print(f"\n>>> Modality: {modality.upper()} <<<")
        ds_names = args.datasets or DEFAULT_DATASETS.get(modality, [])
        for ds_name in ds_names:
            print(f"  Dataset: {ds_name}")
            try:
                bundle, (X_train, X_test, y_train, y_test) = _load_as_2d(modality, ds_name)
            except Exception as e:
                print(f"    Failed to load {ds_name}: {e!r}. Skipping dataset.")
                continue

            for rate in FLIP_RATES:
                for algo_name, cls, kwargs in algos:
                    _run_one(
                        modality=modality,
                        dataset_name=ds_name,
                        algo_name=algo_name,
                        cls=cls,
                        kwargs=kwargs,
                        rate=rate,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )


if __name__ == "__main__":
    main()
