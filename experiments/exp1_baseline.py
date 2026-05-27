"""Exp-1：基准对比实验。

26 个算法 × 29 个数据集，默认超参，记录 AUC-ROC / AUC-PR / F1@best / 耗时。
结果写入 results/exp1_results.csv。

用法：
    python -m experiments.exp1_baseline
    python -m experiments.exp1_baseline --modality tabular   # 只跑表格
    python -m experiments.exp1_baseline --modality timeseries
    python -m experiments.exp1_baseline --modality graph
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.logger import log_experiment
from eval.metrics import evaluate_all

# ---------------------------------------------------------------------------
# 算法注册表
# ---------------------------------------------------------------------------

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp1_results.csv")


def _get_tabular_algos():
    """15 个表格算法（9 无监督 + 6 有监督）。"""
    from models import (
        AutoEncoderDetector, COPODDetector, DeepSVDDDetector,
        ECODDetector, IForestDetector, IQRDetector, KNNDetector,
        LightGBMDetector, LOFDetector, LogisticRegressionDetector,
        MLPDetector, OCSVMDetector, RandomForestDetector,
        TabPFNDetector, XGBoostDetector,
    )
    return [
        # (name, class, extra_kwargs, needs_y)
        ("IQR", IQRDetector, {}, False),
        ("LOF", LOFDetector, {}, False),
        ("KNN", KNNDetector, {}, False),
        ("IForest", IForestDetector, {}, False),
        ("ECOD", ECODDetector, {}, False),
        ("COPOD", COPODDetector, {}, False),
        ("OCSVM", OCSVMDetector, {}, False),
        ("AutoEncoder", AutoEncoderDetector, {"epoch_num": 20}, False),
        ("DeepSVDD", DeepSVDDDetector, {"epochs": 20}, False),
        ("LR", LogisticRegressionDetector, {}, True),
        ("RF", RandomForestDetector, {}, True),
        ("MLP", MLPDetector, {"max_iter": 200}, True),
        ("XGBoost", XGBoostDetector, {}, True),
        ("LightGBM", LightGBMDetector, {}, True),
        ("TabPFN", TabPFNDetector, {}, True),
    ]


def _get_timeseries_algos():
    """4 个时序算法。"""
    from models.timeseries import (
        LSTMAutoEncoderDetector, LSTMSupervisedDetector,
        MatrixProfileDetector, MiniRocketDetector,
    )
    return [
        ("MatrixProfile", MatrixProfileDetector, {"window_size": 64}, False),
        ("MiniRocket", MiniRocketDetector, {"num_kernels": 5000}, True),
        ("LSTM-AE", LSTMAutoEncoderDetector, {"epochs": 20, "hidden_size": 64}, False),
        ("LSTM-Sup", LSTMSupervisedDetector, {"epochs": 20, "hidden_size": 64}, True),
    ]


def _get_graph_algos():
    """2~7 个图算法（取决于 DGL/pyg-lib 是否可用）。"""
    algos = []
    try:
        from models.graph import CoLADetector, DOMINANTDetector
        algos.append(("DOMINANT", DOMINANTDetector, {"epoch": 50, "hid_dim": 64}, False))
        algos.append(("CoLA", CoLADetector, {"epoch": 50, "hid_dim": 64}, False))
    except ImportError:
        pass
    try:
        from models.graph.gnn_supervised import BWGNNDetector, GCNDetector, XGBGraphDetector
        algos.append(("GCN", GCNDetector, {"epochs": 100}, True))
        algos.append(("BWGNN", BWGNNDetector, {"epochs": 100}, True))
        algos.append(("XGBGraph", XGBGraphDetector, {}, True))
    except ImportError:
        pass
    try:
        from models.graph.unprompt import UNPromptDetector
        algos.append(("UNPrompt", UNPromptDetector, {"pretrain_epochs": 50, "prompt_epochs": 50}, False))
    except ImportError:
        pass
    try:
        from models.timeseries.dada import DADADetector
        # DADA 放在时序里，但如果只跑 graph 模态就不加
    except ImportError:
        pass
    return algos


# ---------------------------------------------------------------------------
# 数据加载（兜底逻辑：直接读 npz/csv，等成员 A 的 adapter 就绪后替换）
# ---------------------------------------------------------------------------

TABULAR_DATASETS = [
    "6_cardio", "38_thyroid", "30_satellite", "32_shuttle", "13_fraud",
    "29_Pima", "2_annthyroid", "23_mammography", "28_pendigits",
]
CV_DATASETS = ["CIFAR10_0", "CIFAR10_1", "FashionMNIST_0", "FashionMNIST_1"]
NLP_DATASETS = ["20news_0", "20news_1", "agnews_0", "amazon"]

TIMESERIES_DATASETS = [
    "006_NAB_id_6_Traffic_tr_2579_1st_5839",
    "149_Stock_id_1_Finance_tr_500_1st_7",
    "171_MITDB_id_2_Medical_tr_50000_1st_88864",
    "225_MGAB_id_1_Synthetic_tr_25000_1st_38478",
    "276_IOPS_id_17_WebService_tr_19197_1st_19297",
    "331_UCR_id_29_Facility_tr_50000_1st_837400",
    "337_UCR_id_35_HumanActivity_tr_50000_1st_110260",
    "550_SWaT_id_1_Sensor_tr_43700_1st_43800",
]

GRAPH_DATASETS = ["tfinance", "reddit", "amazon", "weibo"]


def load_tabular(name: str):
    """加载表格/CV/NLP 数据集（npz 格式）。"""
    import re
    for subdir in ["tabular", "cv", "nlp"]:
        data_dir = ROOT / "data" / subdir
        candidates = list(data_dir.glob(f"*{name}*"))
        if candidates:
            data = np.load(candidates[0])
            X = data["X"].astype(np.float64)
            y = data["y"].astype(np.int64)
            # StandardScaler
            mu, std = X.mean(0), X.std(0)
            std[std == 0] = 1.0
            X = (X - mu) / std
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=SEED, stratify=y
            )
            return X_tr, X_te, y_tr, y_te
    raise FileNotFoundError(f"Dataset {name} not found in data/tabular|cv|nlp")


def load_timeseries(name: str, window_size: int = 64, stride: int = 32):
    """加载时序数据集（csv 格式）。"""
    import re
    import pandas as pd

    data_dir = ROOT / "data" / "timeseries"
    candidates = list(data_dir.glob(f"*{name}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Timeseries {name} not found")

    df = pd.read_csv(candidates[0])
    seq = df["Data"].to_numpy(dtype=np.float64)
    lbl = df["Label"].to_numpy(dtype=np.int64)

    # 解析 tr_ 索引
    m = re.search(r"_tr_(\d+)_", candidates[0].name)
    tr = int(m.group(1)) if m else len(seq) // 2

    # z-score
    mu, sigma = seq[:tr].mean(), seq[:tr].std()
    if sigma == 0:
        sigma = 1.0
    seq = (seq - mu) / sigma

    def slide(arr_seq, arr_lbl, start, end):
        windows, labels = [], []
        for i in range(start, end - window_size + 1, stride):
            windows.append(arr_seq[i:i + window_size])
            labels.append(int(arr_lbl[i:i + window_size].max()))
        if not windows:
            return np.empty((0, window_size)), np.empty(0, dtype=np.int64)
        return np.array(windows), np.array(labels, dtype=np.int64)

    X_tr, y_tr = slide(seq, lbl, 0, tr)
    X_te, y_te = slide(seq, lbl, tr, len(seq))

    # 如果训练集没有异常窗口，从测试集借一些
    if y_tr.sum() == 0 and y_te.sum() > 0:
        rng = np.random.RandomState(SEED)
        anom_idx = np.where(y_te == 1)[0]
        n_borrow = min(len(anom_idx) // 3, 50)
        if n_borrow > 0:
            borrow = rng.choice(anom_idx, size=n_borrow, replace=False)
            X_tr = np.concatenate([X_tr, X_te[borrow]])
            y_tr = np.concatenate([y_tr, y_te[borrow]])
            keep = np.setdiff1d(np.arange(len(y_te)), borrow)
            X_te, y_te = X_te[keep], y_te[keep]

    return X_tr, X_te, y_tr, y_te, seq[:tr], seq[tr:]


def load_graph(name: str):
    """加载图数据集（DGL 二进制）。"""
    try:
        import dgl
        from dgl.data.utils import load_graphs
    except ImportError:
        raise RuntimeError("DGL not installed; graph datasets require DGL")

    data_dir = ROOT / "data" / "graph"
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Graph {name} not found at {path}")

    graphs, _ = load_graphs(str(path))
    g = graphs[0]

    # GADBench 格式：ndata 里有 feature / label / train_masks / val_masks / test_masks
    if "train_masks" in g.ndata:
        g.ndata["train_mask"] = g.ndata["train_masks"][:, 0]
        g.ndata["val_mask"] = g.ndata["val_masks"][:, 0]
        g.ndata["test_mask"] = g.ndata["test_masks"][:, 0]

    return g


# ---------------------------------------------------------------------------
# 运行逻辑
# ---------------------------------------------------------------------------


def run_one_tabular(algo_name, Cls, kwargs, needs_y, dataset_name, X_tr, X_te, y_tr, y_te):
    """跑一个表格/CV/NLP 算法。"""
    try:
        det = Cls(contamination=0.1, random_state=SEED, **kwargs)
        t0 = time.perf_counter()
        if needs_y:
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
            dataset_name, algo_name, m["auc_roc"], m["auc_pr"], m["f1_best"],
            fit_t, pred_t, SEED, log_path=LOG_PATH,
        )
        print(f"  [{algo_name:>14s}] AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f} "
              f"F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
        return True
    except Exception as e:
        msg = f"FAILED: {e!r}"[:200]
        log_experiment(
            dataset_name, algo_name, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), SEED, notes=msg, log_path=LOG_PATH,
        )
        print(f"  [{algo_name:>14s}] {msg[:80]}")
        return False


def run_one_timeseries(algo_name, Cls, kwargs, needs_y, dataset_name,
                       X_tr, X_te, y_tr, y_te, train_seq, test_seq):
    """跑一个时序算法。"""
    try:
        det = Cls(contamination=0.1, random_state=SEED, **kwargs)

        # MatrixProfile 用原始序列
        if "MatrixProfile" in algo_name:
            t0 = time.perf_counter()
            det.fit(train_seq)
            fit_t = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores_raw = det.decision_function(test_seq)
            pred_t = time.perf_counter() - t0
            # 映射到窗口级
            stride = 32
            n_w = X_te.shape[0]
            starts = np.clip(np.arange(n_w) * stride, 0, scores_raw.size - 1)
            scores = scores_raw[starts]
        else:
            t0 = time.perf_counter()
            if needs_y:
                if y_tr.sum() == 0 or (y_tr == 1).all():
                    raise RuntimeError(f"训练集只有单一类别 y={np.unique(y_tr)}")
                det.fit(X_tr, y_tr)
            else:
                # 无监督：只用正常窗口训练
                normal_mask = y_tr == 0
                train_data = X_tr[normal_mask] if normal_mask.sum() >= 5 else X_tr
                det.fit(train_data)
            fit_t = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = det.decision_function(X_te)
            pred_t = time.perf_counter() - t0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = evaluate_all(y_te, scores)

        log_experiment(
            dataset_name, algo_name, m["auc_roc"], m["auc_pr"], m["f1_best"],
            fit_t, pred_t, SEED, log_path=LOG_PATH,
        )
        print(f"  [{algo_name:>14s}] AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f} "
              f"F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
        return True
    except Exception as e:
        msg = f"FAILED: {e!r}"[:200]
        log_experiment(
            dataset_name, algo_name, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), SEED, notes=msg, log_path=LOG_PATH,
        )
        print(f"  [{algo_name:>14s}] {msg[:80]}")
        return False


def run_one_graph(algo_name, Cls, kwargs, needs_y, dataset_name, graph):
    """跑一个图算法。"""
    try:
        det = Cls(contamination=0.1, random_state=SEED, **kwargs)
        t0 = time.perf_counter()
        det.fit(graph)
        fit_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = det.decision_function(graph)
        pred_t = time.perf_counter() - t0

        # 只评估测试节点
        import torch
        test_mask = graph.ndata["test_mask"].bool()
        y_test = graph.ndata["label"][test_mask].cpu().numpy().astype(np.int64)
        scores_test = scores[test_mask.cpu().numpy()]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = evaluate_all(y_test, scores_test)

        log_experiment(
            dataset_name, algo_name, m["auc_roc"], m["auc_pr"], m["f1_best"],
            fit_t, pred_t, SEED, log_path=LOG_PATH,
        )
        print(f"  [{algo_name:>14s}] AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f} "
              f"F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
        return True
    except Exception as e:
        msg = f"FAILED: {e!r}"[:200]
        log_experiment(
            dataset_name, algo_name, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), SEED, notes=msg, log_path=LOG_PATH,
        )
        print(f"  [{algo_name:>14s}] {msg[:80]}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Exp-1: Baseline comparison")
    parser.add_argument("--modality", choices=["tabular", "timeseries", "graph", "all"],
                        default="all", help="Which modality to run")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-algorithm timeout in seconds (not enforced, just logged)")
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    # ---- 表格 / CV / NLP ----
    if args.modality in ("tabular", "all"):
        algos = _get_tabular_algos()
        datasets = TABULAR_DATASETS + CV_DATASETS + NLP_DATASETS
        print(f"\n{'='*70}")
        print(f"Exp-1 Tabular/CV/NLP: {len(algos)} algos × {len(datasets)} datasets")
        print(f"{'='*70}")
        for ds in datasets:
            print(f"\n--- Dataset: {ds} ---")
            try:
                X_tr, X_te, y_tr, y_te = load_tabular(ds)
            except Exception as e:
                print(f"  [SKIP] Cannot load {ds}: {e}")
                continue
            for name, Cls, kwargs, needs_y in algos:
                n_total += 1
                if run_one_tabular(name, Cls, kwargs, needs_y, ds, X_tr, X_te, y_tr, y_te):
                    n_ok += 1

    # ---- 时序 ----
    if args.modality in ("timeseries", "all"):
        algos = _get_timeseries_algos()
        # DADA 也加进来
        try:
            from models.timeseries.dada import DADADetector
            algos.append(("DADA", DADADetector, {}, False))
        except ImportError:
            pass
        print(f"\n{'='*70}")
        print(f"Exp-1 Timeseries: {len(algos)} algos × {len(TIMESERIES_DATASETS)} datasets")
        print(f"{'='*70}")
        for ds in TIMESERIES_DATASETS:
            print(f"\n--- Dataset: {ds} ---")
            try:
                X_tr, X_te, y_tr, y_te, train_seq, test_seq = load_timeseries(ds)
            except Exception as e:
                print(f"  [SKIP] Cannot load {ds}: {e}")
                continue
            for name, Cls, kwargs, needs_y in algos:
                n_total += 1
                if run_one_timeseries(name, Cls, kwargs, needs_y, ds,
                                      X_tr, X_te, y_tr, y_te, train_seq, test_seq):
                    n_ok += 1

    # ---- 图 ----
    if args.modality in ("graph", "all"):
        algos = _get_graph_algos()
        if algos:
            print(f"\n{'='*70}")
            print(f"Exp-1 Graph: {len(algos)} algos × {len(GRAPH_DATASETS)} datasets")
            print(f"{'='*70}")
            for ds in GRAPH_DATASETS:
                print(f"\n--- Dataset: {ds} ---")
                try:
                    g = load_graph(ds)
                except Exception as e:
                    print(f"  [SKIP] Cannot load {ds}: {e}")
                    continue
                for name, Cls, kwargs, needs_y in algos:
                    n_total += 1
                    if run_one_graph(name, Cls, kwargs, needs_y, ds, g):
                        n_ok += 1
        else:
            print("\n[INFO] No graph algorithms available (DGL/pyg-lib not installed)")

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-1 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
