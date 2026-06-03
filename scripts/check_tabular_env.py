"""测试 tabular 模态全部 15 个算法是否能跑通。

每个算法用 Cardio（最小数据集，1831 样本 × 21 维）跑一次。
减少 epoch 加快速度。捕获所有错误，最后打印总结表。

用法：
    python -m scripts.check_tabular_env
"""

from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset
from eval.metrics import evaluate_all


def _build_algos():
    from models import (
        AutoEncoderDetector, COPODDetector, DeepSVDDDetector,
        ECODDetector, IForestDetector, IQRDetector, KNNDetector,
        LightGBMDetector, LOFDetector, LogisticRegressionDetector,
        MLPDetector, OCSVMDetector, RandomForestDetector,
        TabPFNDetector, XGBoostDetector,
    )
    # 全部用最小 epoch / 默认参数，目标只是验证 fit + decision_function 能跑
    return [
        ("IQR", IQRDetector, {}, False),
        ("LOF", LOFDetector, {}, False),
        ("KNN", KNNDetector, {}, False),
        ("IForest", IForestDetector, {"n_estimators": 50}, False),
        ("ECOD", ECODDetector, {}, False),
        ("COPOD", COPODDetector, {}, False),
        ("OCSVM", OCSVMDetector, {}, False),
        ("AutoEncoder", AutoEncoderDetector, {"epoch_num": 3}, False),
        ("DeepSVDD", DeepSVDDDetector, {"epochs": 3}, False),
        ("LR", LogisticRegressionDetector, {}, True),
        ("RF", RandomForestDetector, {"n_estimators": 30}, True),
        ("MLP", MLPDetector, {"max_iter": 30}, True),
        ("XGBoost", XGBoostDetector, {"n_estimators": 30}, True),
        ("LightGBM", LightGBMDetector, {"n_estimators": 30}, True),
        ("TabPFN", TabPFNDetector, {}, True),
    ]


def main() -> None:
    print(f"\n{'='*70}\nTabular environment check ({len(_build_algos())} algorithms)\n{'='*70}\n")

    print("Loading Cardio dataset...")
    bundle = load_dataset("cardio")
    X_tr, X_te, y_tr, y_te = bundle.as_tuple()
    print(f"  train={X_tr.shape}, test={X_te.shape}, anomaly_rate={y_te.mean():.4f}\n")

    results = []
    for name, Cls, kwargs, needs_y in _build_algos():
        t0 = time.perf_counter()
        try:
            det = Cls(contamination=0.1, random_state=42, **kwargs)
            if needs_y:
                det.fit(X_tr, y_tr)
            else:
                det.fit(X_tr)
            scores = det.decision_function(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = evaluate_all(y_te, scores)
            elapsed = time.perf_counter() - t0
            print(f"  ✓ [{name:>14s}]  AUC-ROC={m['auc_roc']:.4f}  "
                  f"AUC-PR={m['auc_pr']:.4f}  time={elapsed:.2f}s")
            results.append({"name": name, "ok": True, "auc": m["auc_roc"],
                           "time": elapsed, "err": ""})
        except Exception as e:
            elapsed = time.perf_counter() - t0
            err_msg = f"{type(e).__name__}: {e}"[:120]
            print(f"  ✗ [{name:>14s}]  FAILED  time={elapsed:.2f}s")
            print(f"      {err_msg}")
            results.append({"name": name, "ok": False, "auc": float("nan"),
                           "time": elapsed, "err": err_msg})

    # 总结
    print(f"\n{'='*70}\nSummary\n{'='*70}")
    print(f"{'Algorithm':<15} {'Status':<10} {'AUC-ROC':<10} {'Time(s)':<10}")
    print("-" * 50)
    for r in results:
        status = "✓ OK" if r["ok"] else "✗ FAILED"
        auc = f"{r['auc']:.4f}" if r["ok"] else "—"
        print(f"{r['name']:<15} {status:<10} {auc:<10} {r['time']:<10.2f}")

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} algorithms passed.")
    if n_ok < len(results):
        print("\nFailed algorithms:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['name']}: {r['err']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
