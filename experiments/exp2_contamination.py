"""Exp-2：训练集污染率鲁棒性实验。

通过 ``adapters.load_dataset`` 加载表格数据，在训练集上注入两类污染：
- 无监督污染：控制训练集中异常样本的比例 0% / 1% / 5% / 10% / 20%
- 有监督污染：标签双向对称翻转 0% / 5% / 10% / 20%

记录每个 (算法, 数据集, 污染率) 组合的指标，用于绘制退化曲线。
结果写入 ``results/exp2_results.csv``。

用法：
    python -m experiments.exp2_contamination
    python -m experiments.exp2_contamination --datasets cardio thyroid
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
LOG_PATH = str(ROOT / "results" / "exp2_results.csv")

CONTAMINATION_RATES_UNSUP = [0.0, 0.01, 0.05, 0.10, 0.20]
CONTAMINATION_RATES_SUP = [0.0, 0.05, 0.10, 0.20]

DEFAULT_DATASETS = ["cardio", "thyroid", "satellite", "pima", "annthyroid"]


# ---------------------------------------------------------------------------
# 污染注入（与 data/contaminate.py 兼容；这里就近实现避免循环引用）
# ---------------------------------------------------------------------------


def inject_unsupervised_contamination(X_train, y_train, rate, seed):
    """让训练集中异常占比 ≈ rate。"""
    rng = np.random.RandomState(seed)
    normal_idx = np.where(y_train == 0)[0]
    anomaly_idx = np.where(y_train == 1)[0]

    if rate == 0.0 or len(anomaly_idx) == 0:
        return X_train[normal_idx], y_train[normal_idx]

    n_normal = len(normal_idx)
    n_anomaly_target = int(n_normal * rate / max(1 - rate, 1e-6))
    n_anomaly_target = min(n_anomaly_target, len(anomaly_idx))
    if n_anomaly_target == 0:
        return X_train[normal_idx], y_train[normal_idx]

    chosen = rng.choice(anomaly_idx, size=n_anomaly_target, replace=False)
    keep = np.concatenate([normal_idx, chosen])
    rng.shuffle(keep)
    return X_train[keep], y_train[keep]


def inject_supervised_contamination(y_train, rate, seed):
    """标签双向对称翻转。"""
    if rate == 0.0:
        return y_train.copy()
    rng = np.random.RandomState(seed)
    y_noisy = y_train.copy()
    n = len(y_noisy)
    n_flip = int(n * rate)
    flip = rng.choice(n, size=n_flip, replace=False)
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy


# ---------------------------------------------------------------------------
# 算法
# ---------------------------------------------------------------------------


def _get_unsupervised_algos():
    from models import (
        AutoEncoderDetector, COPODDetector, DeepSVDDDetector,
        ECODDetector, IForestDetector, IQRDetector, KNNDetector,
        LOFDetector, OCSVMDetector,
    )
    return [
        ("IQR", IQRDetector, {}),
        ("LOF", LOFDetector, {}),
        ("KNN", KNNDetector, {}),
        ("IForest", IForestDetector, {}),
        ("ECOD", ECODDetector, {}),
        ("COPOD", COPODDetector, {}),
        ("OCSVM", OCSVMDetector, {}),
        ("AutoEncoder", AutoEncoderDetector, {"epoch_num": 10}),
        ("DeepSVDD", DeepSVDDDetector, {"epochs": 10}),
    ]


def _get_supervised_algos():
    from models import (
        LightGBMDetector, LogisticRegressionDetector,
        RandomForestDetector, XGBoostDetector,
    )
    return [
        ("LR", LogisticRegressionDetector, {}),
        ("RF", RandomForestDetector, {"n_estimators": 100}),
        ("XGBoost", XGBoostDetector, {"n_estimators": 50}),
        ("LightGBM", LightGBMDetector, {"n_estimators": 50}),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Exp-2: Contamination robustness")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0
    unsup_algos = _get_unsupervised_algos()
    sup_algos = _get_supervised_algos()

    for ds in args.datasets:
        print(f"\n{'='*70}\nDataset: {ds}\n{'='*70}")
        try:
            bundle = load_dataset(ds)
            X_tr, X_te, y_tr, y_te = bundle.as_tuple()
            print(f"  Loaded: train={X_tr.shape}, test={X_te.shape}, "
                  f"train_anomaly_rate={y_tr.mean():.4f}")
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        # ---- 无监督算法 × 无监督污染率 ----
        print("\n  --- Unsupervised × contamination rates ---")
        for rate in CONTAMINATION_RATES_UNSUP:
            X_tr_c, y_tr_c = inject_unsupervised_contamination(X_tr, y_tr, rate, SEED)
            for name, Cls, kwargs in unsup_algos:
                n_total += 1
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    det.fit(X_tr_c)
                    scores = det.decision_function(X_te)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = evaluate_all(y_te, scores)
                    log_experiment(
                        ds, name, m["auc_roc"], m["auc_pr"], m["f1_best"],
                        0.0, 0.0, SEED,
                        notes=f"exp2_unsup_rate={rate:.2f}",
                        log_path=LOG_PATH,
                    )
                    print(f"    [{name:>12s}] rate={rate:.2f} "
                          f"AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    log_experiment(
                        ds, name, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED,
                        notes=f"exp2_unsup_rate={rate:.2f} FAILED: {e!r}"[:200],
                        log_path=LOG_PATH,
                    )
                    print(f"    [{name:>12s}] rate={rate:.2f} FAILED")

        # ---- 有监督算法 × 标签翻转率 ----
        print("\n  --- Supervised × label flip rates ---")
        for rate in CONTAMINATION_RATES_SUP:
            y_tr_noisy = inject_supervised_contamination(y_tr, rate, SEED)
            if len(np.unique(y_tr_noisy)) < 2:
                continue
            for name, Cls, kwargs in sup_algos:
                n_total += 1
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    det.fit(X_tr, y_tr_noisy)
                    scores = det.decision_function(X_te)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = evaluate_all(y_te, scores)
                    log_experiment(
                        ds, name, m["auc_roc"], m["auc_pr"], m["f1_best"],
                        0.0, 0.0, SEED,
                        notes=f"exp2_sup_flip={rate:.2f}",
                        log_path=LOG_PATH,
                    )
                    print(f"    [{name:>12s}] flip={rate:.2f} "
                          f"AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    log_experiment(
                        ds, name, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED,
                        notes=f"exp2_sup_flip={rate:.2f} FAILED: {e!r}"[:200],
                        log_path=LOG_PATH,
                    )
                    print(f"    [{name:>12s}] flip={rate:.2f} FAILED")

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-2 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
