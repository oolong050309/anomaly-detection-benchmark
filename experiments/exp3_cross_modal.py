"""Exp-3：跨模态对比实验。

通用算法（能同时跑在表格/CV/NLP/时序/图上的算法）在 5 类形态上运行，
对比同一算法在不同形态上的排名差异。

"通用算法"定义：接受二维 numpy 数组 (n, d) 的无监督/有监督算法。
对于时序和图数据，我们把它们的特征矩阵（窗口化后的二维数组 / 节点特征矩阵）
直接当作表格数据喂给通用算法。

结果写入 results/exp3_results.csv。

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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.logger import log_experiment
from eval.metrics import evaluate_all

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp3_results.csv")


# ---------------------------------------------------------------------------
# 通用算法（能跑在任何二维 numpy 数组上的）
# ---------------------------------------------------------------------------


def _get_universal_algos():
    """通用算法：无监督 + 有监督，都接受 (n, d) 二维数组。"""
    from models import (
        IQRDetector, LOFDetector, KNNDetector, IForestDetector,
        ECODDetector, COPODDetector, OCSVMDetector,
        LogisticRegressionDetector, RandomForestDetector,
        XGBoostDetector, LightGBMDetector,
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


# ---------------------------------------------------------------------------
# 每种形态选 1~2 个代表性数据集
# ---------------------------------------------------------------------------

MODALITY_DATASETS = {
    "tabular": ["6_cardio", "30_satellite"],
    "cv": ["CIFAR10_0", "FashionMNIST_0"],
    "nlp": ["20news_0", "amazon"],
    "timeseries": ["276_IOPS_id_17_WebService_tr_19197_1st_19297"],
    "graph": ["reddit"],
}


# ---------------------------------------------------------------------------
# 数据加载：统一返回 (X_train, X_test, y_train, y_test) 二维 numpy
# ---------------------------------------------------------------------------


def load_as_tabular(modality: str, name: str):
    """把任何形态的数据加载为 (X_train, X_test, y_train, y_test) 二维数组。"""
    import re
    import pandas as pd

    if modality in ("tabular", "cv", "nlp"):
        for subdir in ["tabular", "cv", "nlp"]:
            data_dir = ROOT / "data" / subdir
            candidates = list(data_dir.glob(f"*{name}*"))
            if candidates:
                data = np.load(candidates[0])
                X = data["X"].astype(np.float64)
                y = data["y"].astype(np.int64)
                mu, std = X.mean(0), X.std(0)
                std[std == 0] = 1.0
                X = (X - mu) / std
                return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        raise FileNotFoundError(f"{name} not found in tabular/cv/nlp")

    elif modality == "timeseries":
        data_dir = ROOT / "data" / "timeseries"
        candidates = list(data_dir.glob(f"*{name}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Timeseries {name} not found")
        df = pd.read_csv(candidates[0])
        seq = df["Data"].to_numpy(dtype=np.float64)
        lbl = df["Label"].to_numpy(dtype=np.int64)
        m = re.search(r"_tr_(\d+)_", candidates[0].name)
        tr = int(m.group(1)) if m else len(seq) // 2
        mu, sigma = seq[:tr].mean(), max(seq[:tr].std(), 1e-8)
        seq = (seq - mu) / sigma
        # 切窗
        w, stride = 64, 32
        def slide(s, l, start, end):
            wins, labs = [], []
            for i in range(start, end - w + 1, stride):
                wins.append(s[i:i+w])
                labs.append(int(l[i:i+w].max()))
            return np.array(wins) if wins else np.empty((0, w)), np.array(labs, dtype=np.int64)
        X_tr, y_tr = slide(seq, lbl, 0, tr)
        X_te, y_te = slide(seq, lbl, tr, len(seq))
        # 借异常窗口
        if y_tr.sum() == 0 and y_te.sum() > 0:
            rng = np.random.RandomState(SEED)
            ai = np.where(y_te == 1)[0]
            nb = min(len(ai) // 3, 50)
            if nb > 0:
                b = rng.choice(ai, nb, replace=False)
                X_tr = np.concatenate([X_tr, X_te[b]])
                y_tr = np.concatenate([y_tr, y_te[b]])
                keep = np.setdiff1d(np.arange(len(y_te)), b)
                X_te, y_te = X_te[keep], y_te[keep]
        return X_tr, X_te, y_tr, y_te

    elif modality == "graph":
        # 把图的节点特征当表格数据
        try:
            import dgl
            from dgl.data.utils import load_graphs
        except ImportError:
            raise RuntimeError("DGL not installed")
        path = ROOT / "data" / "graph" / name
        if not path.exists():
            raise FileNotFoundError(f"Graph {name} not found")
        g = load_graphs(str(path))[0][0]
        X = g.ndata["feature"].cpu().numpy().astype(np.float64)
        y = g.ndata["label"].cpu().numpy().astype(np.int64)
        # 用预设 mask 划分
        if "train_masks" in g.ndata:
            tr_mask = g.ndata["train_masks"][:, 0].bool().cpu().numpy()
            te_mask = g.ndata["test_masks"][:, 0].bool().cpu().numpy()
            X_tr, y_tr = X[tr_mask], y[tr_mask]
            X_te, y_te = X[te_mask], y[te_mask]
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=SEED, stratify=y
            )
        # StandardScaler
        mu, std = X_tr.mean(0), X_tr.std(0)
        std[std == 0] = 1.0
        X_tr = (X_tr - mu) / std
        X_te = (X_te - mu) / std
        return X_tr, X_te, y_tr, y_te

    raise ValueError(f"Unknown modality: {modality}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Exp-3: Cross-modal comparison")
    args = parser.parse_args()

    algos = _get_universal_algos()
    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    for modality, datasets in MODALITY_DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Modality: {modality} ({len(datasets)} datasets)")
        print(f"{'='*70}")

        for ds in datasets:
            print(f"\n  --- {modality}/{ds} ---")
            try:
                X_tr, X_te, y_tr, y_te = load_as_tabular(modality, ds)
                print(f"  Loaded: train={X_tr.shape}, test={X_te.shape}, "
                      f"anomaly_rate={y_te.mean():.4f}")
            except Exception as e:
                print(f"  [SKIP] {e}")
                continue

            for algo_name, Cls, kwargs, needs_y in algos:
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
                        f"{modality}/{ds}", algo_name,
                        m["auc_roc"], m["auc_pr"], m["f1_best"],
                        fit_t, pred_t, SEED,
                        notes=f"exp3_modality={modality}",
                        log_path=LOG_PATH,
                    )
                    print(f"    [{algo_name:>10s}] AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    log_experiment(
                        f"{modality}/{ds}", algo_name,
                        float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED,
                        notes=f"exp3_modality={modality} FAILED: {e!r}"[:200],
                        log_path=LOG_PATH,
                    )
                    print(f"    [{algo_name:>10s}] FAILED: {e!r}"[:80])

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-3 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
