"""Exp-2：训练集污染率鲁棒性实验。

对通用算法（表格无监督 + 有监督），在表格数据集上控制训练集中异常样本的保留比例：
  - 无监督污染率：0% / 1% / 5% / 10% / 20%（控制训练集中异常样本的比例）
  - 有监督污染率：标签双向对称翻转 0% / 5% / 10% / 20%

记录每个 (算法, 数据集, 污染率) 组合的 AUC-ROC，用于绘制退化曲线。
结果写入 results/exp2_results.csv。

用法：
    python -m experiments.exp2_contamination
    python -m experiments.exp2_contamination --datasets 6_cardio 38_thyroid
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.logger import log_experiment
from eval.metrics import evaluate_all

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp2_results.csv")

# 污染率档位
CONTAMINATION_RATES_UNSUP = [0.0, 0.01, 0.05, 0.10, 0.20]
CONTAMINATION_RATES_SUP = [0.0, 0.05, 0.10, 0.20]

# 默认数据集（表格子集，避免太慢）
DEFAULT_DATASETS = ["6_cardio", "38_thyroid", "30_satellite", "29_Pima", "2_annthyroid"]


# ---------------------------------------------------------------------------
# 污染注入
# ---------------------------------------------------------------------------


def inject_unsupervised_contamination(
    X_train: np.ndarray, y_train: np.ndarray, rate: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """无监督污染：控制训练集中异常样本的保留比例。

    rate=0 表示纯净训练集（移除所有异常）；
    rate=0.2 表示保留 20% 的异常样本在训练集中。
    """
    rng = np.random.RandomState(seed)
    normal_idx = np.where(y_train == 0)[0]
    anomaly_idx = np.where(y_train == 1)[0]

    if rate == 0.0 or len(anomaly_idx) == 0:
        # 纯净：只保留正常样本
        return X_train[normal_idx], y_train[normal_idx]

    # 保留 rate 比例的异常
    n_keep = max(1, int(len(anomaly_idx) * rate / max(y_train.mean(), 1e-6) * rate))
    # 更简单的做法：目标是让最终训练集中异常占比 = rate
    n_normal = len(normal_idx)
    n_anomaly_target = int(n_normal * rate / (1 - rate))
    n_anomaly_target = min(n_anomaly_target, len(anomaly_idx))
    if n_anomaly_target == 0:
        return X_train[normal_idx], y_train[normal_idx]

    chosen_anomaly = rng.choice(anomaly_idx, size=n_anomaly_target, replace=False)
    keep_idx = np.concatenate([normal_idx, chosen_anomaly])
    rng.shuffle(keep_idx)
    return X_train[keep_idx], y_train[keep_idx]


def inject_supervised_contamination(
    y_train: np.ndarray, rate: float, seed: int
) -> np.ndarray:
    """有监督污染：标签双向对称翻转。

    随机选 rate 比例的样本，把 0↔1 互换。
    """
    if rate == 0.0:
        return y_train.copy()
    rng = np.random.RandomState(seed)
    y_noisy = y_train.copy()
    n = len(y_noisy)
    n_flip = int(n * rate)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy


# ---------------------------------------------------------------------------
# 算法
# ---------------------------------------------------------------------------


def _get_unsupervised_algos():
    from models import (
        IQRDetector, LOFDetector, KNNDetector, IForestDetector,
        ECODDetector, COPODDetector, OCSVMDetector,
        AutoEncoderDetector, DeepSVDDDetector,
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
        LogisticRegressionDetector, RandomForestDetector,
        XGBoostDetector, LightGBMDetector,
    )
    return [
        ("LR", LogisticRegressionDetector, {}),
        ("RF", RandomForestDetector, {"n_estimators": 100}),
        ("XGBoost", XGBoostDetector, {"n_estimators": 50}),
        ("LightGBM", LightGBMDetector, {"n_estimators": 50}),
    ]


# ---------------------------------------------------------------------------
# 数据加载（复用 exp1 的逻辑）
# ---------------------------------------------------------------------------


def load_tabular(name: str):
    data_dir = ROOT / "data" / "tabular"
    candidates = list(data_dir.glob(f"*{name}*"))
    if not candidates:
        raise FileNotFoundError(f"Dataset {name} not found")
    data = np.load(candidates[0])
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    mu, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    X = (X - mu) / std
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


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
        print(f"\n{'='*70}")
        print(f"Dataset: {ds}")
        print(f"{'='*70}")
        try:
            X_tr, X_te, y_tr, y_te = load_tabular(ds)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        # ---- 无监督算法 × 无监督污染率 ----
        print(f"\n  --- Unsupervised algorithms × contamination rates ---")
        for rate in CONTAMINATION_RATES_UNSUP:
            X_tr_c, y_tr_c = inject_unsupervised_contamination(X_tr, y_tr, rate, SEED)
            for algo_name, Cls, kwargs in unsup_algos:
                n_total += 1
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    det.fit(X_tr_c)
                    scores = det.decision_function(X_te)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = evaluate_all(y_te, scores)
                    log_experiment(
                        ds, algo_name, m["auc_roc"], m["auc_pr"], m["f1_best"],
                        0.0, 0.0, SEED,
                        notes=f"exp2_unsup_rate={rate:.2f}",
                        log_path=LOG_PATH,
                    )
                    print(f"    [{algo_name:>12s}] rate={rate:.2f} AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    log_experiment(
                        ds, algo_name, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED,
                        notes=f"exp2_unsup_rate={rate:.2f} FAILED: {e!r}"[:200],
                        log_path=LOG_PATH,
                    )
                    print(f"    [{algo_name:>12s}] rate={rate:.2f} FAILED")

        # ---- 有监督算法 × 标签翻转率 ----
        print(f"\n  --- Supervised algorithms × label flip rates ---")
        for rate in CONTAMINATION_RATES_SUP:
            y_tr_noisy = inject_supervised_contamination(y_tr, rate, SEED)
            # 确保翻转后仍有两个类
            if len(np.unique(y_tr_noisy)) < 2:
                continue
            for algo_name, Cls, kwargs in sup_algos:
                n_total += 1
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    det.fit(X_tr, y_tr_noisy)
                    scores = det.decision_function(X_te)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = evaluate_all(y_te, scores)
                    log_experiment(
                        ds, algo_name, m["auc_roc"], m["auc_pr"], m["f1_best"],
                        0.0, 0.0, SEED,
                        notes=f"exp2_sup_flip={rate:.2f}",
                        log_path=LOG_PATH,
                    )
                    print(f"    [{algo_name:>12s}] flip={rate:.2f} AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    log_experiment(
                        ds, algo_name, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED,
                        notes=f"exp2_sup_flip={rate:.2f} FAILED: {e!r}"[:200],
                        log_path=LOG_PATH,
                    )
                    print(f"    [{algo_name:>12s}] flip={rate:.2f} FAILED")

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-2 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
