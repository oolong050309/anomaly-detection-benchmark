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
from eval.logger import log_experiment
from eval.metrics import evaluate_all

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp3_results.csv")


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


def _load_as_2d(modality: str, ds_name: str):
    """统一返回 (X_train, X_test, y_train, y_test) 二维数组。"""
    if modality in ("tabular", "cv", "nlp"):
        bundle = load_dataset(ds_name)
        return bundle.as_tuple()

    if modality == "timeseries":
        bundle = load_dataset(ds_name, modality="timeseries",
                              window_size=64, stride=32)
        return bundle.as_tuple()

    if modality == "graph":
        bundle = load_dataset(ds_name, modality="graph")
        # 把节点特征当表格数据用
        return bundle.as_tuple()

    raise ValueError(f"Unknown modality: {modality}")


def main():
    parser = argparse.ArgumentParser(description="Exp-3: Cross-modal comparison")
    args = parser.parse_args()

    algos = _get_universal_algos()
    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    for modality, datasets in MODALITY_DATASETS.items():
        print(f"\n{'='*70}\nModality: {modality} ({len(datasets)} datasets)\n{'='*70}")
        for ds in datasets:
            print(f"\n  --- {modality}/{ds} ---")
            try:
                X_tr, X_te, y_tr, y_te = _load_as_2d(modality, ds)
                print(f"  Loaded: train={X_tr.shape}, test={X_te.shape}, "
                      f"anomaly_rate={y_te.mean():.4f}")
            except Exception as e:
                print(f"  [SKIP] {e}")
                continue

            for name, Cls, kwargs, needs_y in algos:
                n_total += 1
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    t0 = time.perf_counter()
                    if needs_y:
                        if len(np.unique(y_tr)) < 2:
                            raise RuntimeError("y_train has single class")
                        det.fit(X_tr, y_tr)
                    else:
                        det.fit(X_tr)
                    fit_t = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    scores = det.decision_function(X_te)
                    pred_t = time.perf_counter() - t0

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = evaluate_all(y_te, scores)

                    log_experiment(
                        f"{modality}/{ds}", name,
                        m["auc_roc"], m["auc_pr"], m["f1_best"],
                        fit_t, pred_t, SEED,
                        notes=f"exp3_modality={modality}",
                        log_path=LOG_PATH,
                    )
                    print(f"    [{name:>10s}] AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    log_experiment(
                        f"{modality}/{ds}", name,
                        float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED,
                        notes=f"exp3_modality={modality} FAILED: {e!r}"[:200],
                        log_path=LOG_PATH,
                    )
                    print(f"    [{name:>10s}] FAILED: {e!r}"[:80])

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-3 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
