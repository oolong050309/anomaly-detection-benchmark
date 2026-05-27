"""Exp-3：跨模态对比实验。

通用算法（接受二维 (n, d) numpy 数组的算法）在 5 类形态上运行，
对比同一算法在不同形态上的排名差异。

对于时序和图数据：
- 时序：使用 adapter 切窗后的二维窗口数组 ``(n_windows, window_size)``
- 图：使用图节点特征矩阵 ``(n_nodes, n_features)`` + 节点级 mask 划分

结果写入 ``results/exp3_results.csv``。

用法：
    python -m experiments.exp3_cross_modal
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
from experiments.common import fit_and_score, record_failure, record_success

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp3_results.csv")
OUTPUT_DIR = str(ROOT / "results")


def _get_universal_algos():
    """11 个通用算法。"""
    from models import (
        COPODDetector, ECODDetector, IForestDetector, IQRDetector,
        KNNDetector, LightGBMDetector, LOFDetector,
        LogisticRegressionDetector, OCSVMDetector,
        RandomForestDetector, XGBoostDetector,
    )
    return [
        # (name, class, kwargs, needs_y)
        ("IQR", IQRDetector, {}, False),
        ("LOF", LOFDetector, {}, False),
        ("KNN", KNNDetector, {}, False),
        ("IForest", IForestDetector, {}, False),
        ("ECOD", ECODDetector, {}, False),
        ("COPOD", COPODDetector, {}, False),
        ("OCSVM", OCSVMDetector, {}, False),
        ("LR", LogisticRegressionDetector, {}, True),
        ("RF", RandomForestDetector, {"n_estimators": 100}, True),
        ("XGBoost", XGBoostDetector, {"n_estimators": 50}, True),
        ("LightGBM", LightGBMDetector, {"n_estimators": 50}, True),
    ]


# 每种形态选 1~2 个代表性数据集
MODALITY_DATASETS = {
    "tabular": ["cardio", "satellite"],
    "cv": ["cifar10_0", "fashionmnist_0"],
    "nlp": ["20news_0", "amazon"],
    "timeseries": ["276_IOPS_id_17_WebService"],
    "graph": ["reddit"],
}


def _load_as_2d(modality: str, ds_name: str, data_root=None):
    """统一返回 (X_train, X_test, y_train, y_test) 二维数组。"""
    if modality in ("tabular", "cv", "nlp"):
        bundle = load_dataset(ds_name, data_root=data_root)
        return bundle, bundle.as_tuple()

    if modality == "timeseries":
        bundle = load_dataset(ds_name, modality="timeseries",
                              data_root=data_root,
                              window_size=64, stride=32)
        return bundle, bundle.as_tuple()

    if modality == "graph":
        bundle = load_dataset(ds_name, modality="graph", data_root=data_root)
        # 把节点特征当表格数据用
        return bundle, bundle.as_tuple()

    raise ValueError(f"Unknown modality: {modality}")


def main():
    global SEED, LOG_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Exp-3: Cross-modal comparison")
    parser.add_argument("--modalities", nargs="+",
                        default=["tabular", "cv", "nlp", "timeseries", "graph"],
                        choices=["tabular", "cv", "nlp", "timeseries", "graph"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_DIR = str(Path(args.output_dir))
    LOG_PATH = str(Path(args.log_path) if args.log_path else Path(OUTPUT_DIR) / "exp3_results.csv")

    algos = _get_universal_algos()
    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    for modality, datasets in MODALITY_DATASETS.items():
        if modality not in args.modalities:
            continue
        print(f"\n{'='*70}\nModality: {modality} ({len(datasets)} datasets)\n{'='*70}")
        for ds in datasets:
            print(f"\n  --- {modality}/{ds} ---")
            try:
                bundle, (X_tr, X_te, y_tr, y_te) = _load_as_2d(modality, ds, args.data_root)
                print(f"  Loaded: train={X_tr.shape}, test={X_te.shape}, "
                      f"anomaly_rate={y_te.mean():.4f}")
            except Exception as e:
                print(f"  [SKIP] {e}")
                continue

            for name, Cls, kwargs, needs_y in algos:
                n_total += 1
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    if needs_y:
                        if len(np.unique(y_tr)) < 2:
                            raise RuntimeError("y_train has single class")
                        fit_args = (X_tr, y_tr)
                    else:
                        fit_args = (X_tr,)
                    scores, fit_t, pred_t = fit_and_score(det, fit_args, X_te)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = record_success(
                            log_path=LOG_PATH,
                            output_dir=OUTPUT_DIR,
                            experiment_name="exp3",
                            modality=modality,
                            dataset_name=f"{modality}/{ds}",
                            algorithm_name=name,
                            y_true=y_te,
                            scores=scores,
                            fit_time_sec=fit_t,
                            predict_time_sec=pred_t,
                            seed=SEED,
                            fit_params=kwargs,
                            notes=f"exp3_modality={modality}",
                            X_train=X_tr,
                            y_train=y_tr,
                        )
                    print(f"    [{name:>10s}] AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    record_failure(
                        log_path=LOG_PATH,
                        experiment_name="exp3",
                        modality=modality,
                        dataset_name=f"{modality}/{ds}",
                        algorithm_name=name,
                        seed=SEED,
                        error=e,
                        fit_params=kwargs,
                        notes=f"exp3_modality={modality}",
                        X_train=X_tr if "X_tr" in locals() else None,
                        y_train=y_tr if "y_tr" in locals() else None,
                        y_test=y_te if "y_te" in locals() else None,
                    )
                    print(f"    [{name:>10s}] FAILED: {e!r}"[:80])

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-3 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"Artifacts: {Path(OUTPUT_DIR) / 'artifacts' / 'exp3'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
