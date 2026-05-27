"""Exp-1：基准对比实验。

26 个算法 × 29 个数据集，默认超参，记录 AUC-ROC / AUC-PR / F1@best / 耗时。
通过 ``adapters.load_dataset`` 加载数据，由成员 A 的统一适配器负责所有预处理。

结果写入 ``results/exp1_results.csv``。

用法：
    python -m experiments.exp1_baseline
    python -m experiments.exp1_baseline --modality tabular
    python -m experiments.exp1_baseline --modality timeseries
    python -m experiments.exp1_baseline --modality graph
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

from adapters import DatasetBundle, load_dataset
from eval.logger import log_experiment
from eval.metrics import evaluate_all

# ---------------------------------------------------------------------------
# 数据集清单（与 adapters/adbench_adapter.py 的命名一致）
# ---------------------------------------------------------------------------

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp1_results.csv")

TABULAR_DATASETS = [
    "cardio", "thyroid", "satellite", "shuttle", "credit_card",
    "pima", "annthyroid", "mammography", "pendigits",
]
CV_DATASETS = ["cifar10_0", "cifar10_1", "fashionmnist_0", "fashionmnist_1"]
NLP_DATASETS = ["20news_0", "20news_1", "agnews_0", "amazon"]

# 时序数据集：传 CSV 文件名片段，adapter 会自动定位
TIMESERIES_DATASETS = [
    "006_NAB_id_6_Traffic",
    "149_Stock_id_1_Finance",
    "171_MITDB_id_2_Medical",
    "225_MGAB_id_1_Synthetic",
    "276_IOPS_id_17_WebService",
    "331_UCR_id_29_Facility",
    "337_UCR_id_35_HumanActivity",
    "550_SWaT_id_1_Sensor",
]

GRAPH_DATASETS = ["tfinance", "reddit", "amazon", "weibo"]


# ---------------------------------------------------------------------------
# 算法注册表
# ---------------------------------------------------------------------------


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
    """4 个时序算法 + 可选 DADA。"""
    from models.timeseries import (
        LSTMAutoEncoderDetector, LSTMSupervisedDetector,
        MatrixProfileDetector, MiniRocketDetector,
    )
    algos = [
        ("MatrixProfile", MatrixProfileDetector, {"window_size": 100}, False),
        ("MiniRocket", MiniRocketDetector, {"num_kernels": 5000}, True),
        ("LSTM-AE", LSTMAutoEncoderDetector, {"epochs": 20, "hidden_size": 64}, False),
        ("LSTM-Sup", LSTMSupervisedDetector, {"epochs": 20, "hidden_size": 64}, True),
    ]
    try:
        from models.timeseries.dada import DADADetector
        algos.append(("DADA", DADADetector, {}, False))
    except ImportError:
        pass
    return algos


def _get_graph_algos():
    """图算法（取决于 DGL/pyg-lib/PyGOD 是否可用）。"""
    algos = []
    try:
        from models.graph import CoLADetector, DOMINANTDetector
        algos.append(("DOMINANT", DOMINANTDetector, {"epoch": 50, "hid_dim": 64}, False))
        algos.append(("CoLA", CoLADetector, {"epoch": 50, "hid_dim": 64}, False))
    except ImportError:
        pass
    try:
        from models.graph.gnn_supervised import (
            BWGNNDetector, GCNDetector, XGBGraphDetector,
        )
        algos.append(("GCN", GCNDetector, {"epochs": 100}, True))
        algos.append(("BWGNN", BWGNNDetector, {"epochs": 100}, True))
        algos.append(("XGBGraph", XGBGraphDetector, {}, True))
    except ImportError:
        pass
    try:
        from models.graph.unprompt import UNPromptDetector
        algos.append(
            ("UNPrompt", UNPromptDetector,
             {"pretrain_epochs": 50, "prompt_epochs": 50}, False)
        )
    except ImportError:
        pass
    return algos


# ---------------------------------------------------------------------------
# 通用 runner
# ---------------------------------------------------------------------------


def _run_one(name, Cls, kwargs, needs_y, ds_name, fit_input, predict_input,
             y_test, fit_kwargs=None) -> bool:
    """通用算法运行器：fit -> decision_function -> evaluate -> log。"""
    fit_kwargs = fit_kwargs or {}
    try:
        det = Cls(contamination=0.1, random_state=SEED, **kwargs)

        t0 = time.perf_counter()
        if needs_y:
            det.fit(*fit_input)
        else:
            det.fit(fit_input if not isinstance(fit_input, tuple) else fit_input[0])
        fit_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = det.decision_function(predict_input)
        pred_t = time.perf_counter() - t0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = evaluate_all(y_test, scores)

        log_experiment(
            ds_name, name, m["auc_roc"], m["auc_pr"], m["f1_best"],
            fit_t, pred_t, SEED, log_path=LOG_PATH,
        )
        print(f"  [{name:>14s}] AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f} "
              f"F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
        return True
    except Exception as e:
        msg = f"FAILED: {e!r}"[:200]
        log_experiment(
            ds_name, name, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), SEED, notes=msg, log_path=LOG_PATH,
        )
        print(f"  [{name:>14s}] {msg[:80]}")
        return False


# ---------------------------------------------------------------------------
# 各模态实验主逻辑
# ---------------------------------------------------------------------------


def run_tabular(datasets, algos):
    print(f"\n{'='*70}")
    print(f"Exp-1 Tabular/CV/NLP: {len(algos)} algos × {len(datasets)} datasets")
    print(f"{'='*70}")
    n_ok, n_total = 0, 0
    for ds in datasets:
        print(f"\n--- Dataset: {ds} ---")
        try:
            bundle = load_dataset(ds)
            X_tr, X_te, y_tr, y_te = bundle.as_tuple()
            print(f"  Loaded: train={X_tr.shape}, test={X_te.shape}, "
                  f"anomaly_rate={y_te.mean():.4f}")
        except Exception as e:
            print(f"  [SKIP] Cannot load {ds}: {e}")
            continue
        for name, Cls, kwargs, needs_y in algos:
            n_total += 1
            fit_input = (X_tr, y_tr) if needs_y else X_tr
            if _run_one(name, Cls, kwargs, needs_y, ds, fit_input, X_te, y_te):
                n_ok += 1
    return n_ok, n_total


def run_timeseries(datasets, algos):
    print(f"\n{'='*70}")
    print(f"Exp-1 Timeseries: {len(algos)} algos × {len(datasets)} datasets")
    print(f"{'='*70}")
    n_ok, n_total = 0, 0
    for ds in datasets:
        print(f"\n--- Dataset: {ds} ---")
        try:
            bundle = load_dataset(ds, modality="timeseries",
                                  window_size=100, stride=10)
            X_tr, X_te, y_tr, y_te = bundle.as_tuple()
            raw_values = bundle.extras["raw_values"]
            train_end = bundle.extras["train_end"]
            train_seq = raw_values[:train_end]
            test_seq = raw_values[train_end:]
            print(f"  Loaded: train_windows={X_tr.shape}, test_windows={X_te.shape}, "
                  f"test_anomaly_rate={y_te.mean():.4f}")
        except Exception as e:
            print(f"  [SKIP] Cannot load {ds}: {e}")
            continue

        for name, Cls, kwargs, needs_y in algos:
            n_total += 1

            # MatrixProfile: 用原始一维序列，再映射回窗口
            if name == "MatrixProfile":
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    t0 = time.perf_counter()
                    det.fit(train_seq)
                    fit_t = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    scores_seq = det.decision_function(test_seq)
                    pred_t = time.perf_counter() - t0
                    n_w = X_te.shape[0]
                    starts = np.clip(np.arange(n_w) * 10, 0, scores_seq.size - 1)
                    scores = scores_seq[starts]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = evaluate_all(y_te, scores)
                    log_experiment(
                        ds, name, m["auc_roc"], m["auc_pr"], m["f1_best"],
                        fit_t, pred_t, SEED, log_path=LOG_PATH,
                    )
                    print(f"  [{name:>14s}] AUC-ROC={m['auc_roc']:.4f} "
                          f"AUC-PR={m['auc_pr']:.4f} F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
                    n_ok += 1
                except Exception as e:
                    msg = f"FAILED: {e!r}"[:200]
                    log_experiment(
                        ds, name, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED, notes=msg, log_path=LOG_PATH,
                    )
                    print(f"  [{name:>14s}] {msg[:80]}")
                continue

            # 无监督算法只用正常窗口训练
            if not needs_y:
                normal_mask = y_tr == 0
                X_tr_use = X_tr[normal_mask] if normal_mask.sum() >= 5 else X_tr
                fit_input = X_tr_use
            else:
                if y_tr.sum() == 0 or (y_tr == 1).all():
                    log_experiment(
                        ds, name, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), SEED,
                        notes="train set has single class", log_path=LOG_PATH,
                    )
                    print(f"  [{name:>14s}] SKIP: train set has single class")
                    continue
                fit_input = (X_tr, y_tr)

            if _run_one(name, Cls, kwargs, needs_y, ds, fit_input, X_te, y_te):
                n_ok += 1
    return n_ok, n_total


def run_graph(datasets, algos):
    print(f"\n{'='*70}")
    print(f"Exp-1 Graph: {len(algos)} algos × {len(datasets)} datasets")
    print(f"{'='*70}")
    n_ok, n_total = 0, 0
    for ds in datasets:
        print(f"\n--- Dataset: {ds} ---")
        try:
            bundle = load_dataset(ds, modality="graph")
            graph = bundle.extras["graph"]
            masks = bundle.extras["masks"]
            y_all = bundle.extras["y_all"]
            test_idx = np.flatnonzero(masks["test_mask"])
            y_te = y_all[test_idx]
            print(f"  Loaded: nodes={graph.num_nodes()}, "
                  f"edges={graph.num_edges()}, test_nodes={len(test_idx)}, "
                  f"test_anomaly_rate={y_te.mean():.4f}")
        except Exception as e:
            print(f"  [SKIP] Cannot load {ds}: {e}")
            continue

        for name, Cls, kwargs, needs_y in algos:
            n_total += 1
            try:
                det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                t0 = time.perf_counter()
                det.fit(graph)
                fit_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                scores_all = det.decision_function(graph)
                pred_t = time.perf_counter() - t0

                # 只在测试节点上评估
                scores_test = np.asarray(scores_all)[test_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = evaluate_all(y_te, scores_test)

                log_experiment(
                    ds, name, m["auc_roc"], m["auc_pr"], m["f1_best"],
                    fit_t, pred_t, SEED, log_path=LOG_PATH,
                )
                print(f"  [{name:>14s}] AUC-ROC={m['auc_roc']:.4f} "
                      f"AUC-PR={m['auc_pr']:.4f} F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
                n_ok += 1
            except Exception as e:
                msg = f"FAILED: {e!r}"[:200]
                log_experiment(
                    ds, name, float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), SEED,
                    notes=msg, log_path=LOG_PATH,
                )
                print(f"  [{name:>14s}] {msg[:80]}")
    return n_ok, n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Exp-1: Baseline comparison")
    parser.add_argument("--modality",
                        choices=["tabular", "timeseries", "graph", "all"],
                        default="all", help="Which modality to run")
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    if args.modality in ("tabular", "all"):
        algos = _get_tabular_algos()
        datasets = TABULAR_DATASETS + CV_DATASETS + NLP_DATASETS
        ok, total = run_tabular(datasets, algos)
        n_ok += ok
        n_total += total

    if args.modality in ("timeseries", "all"):
        algos = _get_timeseries_algos()
        ok, total = run_timeseries(TIMESERIES_DATASETS, algos)
        n_ok += ok
        n_total += total

    if args.modality in ("graph", "all"):
        algos = _get_graph_algos()
        if algos:
            ok, total = run_graph(GRAPH_DATASETS, algos)
            n_ok += ok
            n_total += total
        else:
            print("\n[INFO] No graph algorithms available "
                  "(DGL/pyg-lib/PyGOD not installed)")

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-1 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
