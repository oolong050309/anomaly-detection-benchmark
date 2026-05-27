"""任务 13：表格烟雾测试。

加载 ADBench 的 Cardio 数据集，对全部 15 个表格算法（9 个无监督 + 6 个有监督）
做端到端 fit -> decision_function -> evaluate_all -> log_experiment。
单算法异常被捕获，notes 字段记录截断后的错误信息（最多 200 字符）。

用法：
    python -m scripts.smoke_test_tabular
"""

from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.logger import log_experiment  # noqa: E402
from eval.metrics import evaluate_all  # noqa: E402
from models import (  # noqa: E402
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
    SupervisedDetector,
    TabPFNDetector,
    XGBoostDetector,
)


DATASET_NAME = "Cardio"
DATASET_PATH = ROOT / "data" / "tabular" / "6_cardio.npz"
LOG_PATH = ROOT / "results" / "experiment_log.csv"
SEED = 42


# 9 个无监督 + 6 个有监督 = 15 个表格算法
ALGOS = [
    # 无监督（不需要 y）
    ("IQR", IQRDetector, {}, False),
    ("LOF", LOFDetector, {}, False),
    ("KNN", KNNDetector, {}, False),
    ("IForest", IForestDetector, {}, False),
    ("ECOD", ECODDetector, {}, False),
    ("COPOD", COPODDetector, {}, False),
    ("OCSVM", OCSVMDetector, {}, False),
    ("AutoEncoder", AutoEncoderDetector, {"epoch_num": 5, "batch_size": 64}, False),
    ("DeepSVDD", DeepSVDDDetector, {"epochs": 5, "batch_size": 64}, False),
    # 有监督（需要 y）
    ("LR", LogisticRegressionDetector, {}, True),
    ("RF", RandomForestDetector, {"n_estimators": 100}, True),
    ("MLP", MLPDetector, {"max_iter": 100}, True),
    ("XGBoost", XGBoostDetector, {"n_estimators": 100}, True),
    ("LightGBM", LightGBMDetector, {"n_estimators": 100}, True),
    ("TabPFN", TabPFNDetector, {}, True),
]


def load_cardio() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Cardio dataset not found at {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    print(
        f"Loaded {DATASET_NAME}: X.shape={X.shape}, y.shape={y.shape}, "
        f"anomaly_rate={y.mean():.4f}"
    )

    # StandardScaler 归一化（adapter 之外，本脚本兜底处理）
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    print(
        f"Split: train={X_train.shape[0]}, test={X_test.shape[0]}, "
        f"train_anomaly_rate={y_train.mean():.4f}"
    )
    return X_train, X_test, y_train, y_test


def run_one(
    name: str,
    Cls: type,
    kwargs: dict,
    needs_y: bool,
    X_train, X_test, y_train, y_test,
) -> dict:
    result = {
        "name": name,
        "auc_roc": float("nan"),
        "auc_pr": float("nan"),
        "f1_best": float("nan"),
        "fit_time": float("nan"),
        "predict_time": float("nan"),
        "notes": "",
    }
    try:
        det = Cls(contamination=0.1, random_state=SEED, **kwargs)

        t0 = time.perf_counter()
        if needs_y:
            det.fit(X_train, y_train)
        else:
            det.fit(X_train)
        result["fit_time"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = det.decision_function(X_test)
        result["predict_time"] = time.perf_counter() - t0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = evaluate_all(y_test, scores)

        result["auc_roc"] = metrics["auc_roc"]
        result["auc_pr"] = metrics["auc_pr"]
        result["f1_best"] = metrics["f1_best"]
        print(
            f"  [{name:>12s}]  AUC-ROC={result['auc_roc']:.4f}  "
            f"AUC-PR={result['auc_pr']:.4f}  F1={result['f1_best']:.4f}  "
            f"fit={result['fit_time']:.2f}s  pred={result['predict_time']:.3f}s"
        )
    except Exception as e:
        result["notes"] = (f"FAILED: {e!r}")[:200]
        print(f"  [{name:>12s}]  FAILED: {e!r}"[:120])
        # 输出简短堆栈帮助诊断
        tb = traceback.format_exc().splitlines()[-3:]
        for line in tb:
            print(f"    {line}")
    return result


def print_summary_table(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"Summary: {DATASET_NAME}")
    print("=" * 78)
    print(
        f"{'Algorithm':<14} | {'AUC-ROC':>8} | {'AUC-PR':>8} | "
        f"{'F1@best':>8} | {'fit(s)':>7} | {'pred(s)':>7} | notes"
    )
    print("-" * 78)
    for r in results:
        notes = r["notes"][:30] + "..." if len(r["notes"]) > 33 else r["notes"]
        print(
            f"{r['name']:<14} | {r['auc_roc']:>8.4f} | {r['auc_pr']:>8.4f} | "
            f"{r['f1_best']:>8.4f} | {r['fit_time']:>7.2f} | "
            f"{r['predict_time']:>7.3f} | {notes}"
        )


def main() -> None:
    X_train, X_test, y_train, y_test = load_cardio()
    print()

    results = []
    total_t0 = time.perf_counter()
    for name, Cls, kwargs, needs_y in ALGOS:
        # SupervisedDetector 子类必须传 y
        is_supervised = issubclass(Cls, SupervisedDetector)
        if is_supervised != needs_y:
            print(f"  [WARN] {name} supervised flag mismatch")

        r = run_one(
            name, Cls, kwargs, needs_y,
            X_train, X_test, y_train, y_test
        )
        results.append(r)

        log_experiment(
            dataset_name=DATASET_NAME,
            algorithm_name=name,
            auc_roc=r["auc_roc"],
            auc_pr=r["auc_pr"],
            f1_best=r["f1_best"],
            fit_time_sec=r["fit_time"],
            predict_time_sec=r["predict_time"],
            seed=SEED,
            notes=r["notes"],
            log_path=str(LOG_PATH),
        )

    total = time.perf_counter() - total_t0
    print_summary_table(results)
    n_ok = sum(1 for r in results if not np.isnan(r["auc_roc"]))
    print(f"\n{n_ok}/{len(results)} algorithms succeeded in {total:.1f}s.")
    print(f"Log written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
