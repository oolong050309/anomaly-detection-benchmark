"""测试时序 + 图模态全部算法是否能跑通。

时序：MatrixProfile / MiniRocket / LSTM-AE / LSTM-Sup / DADA（5 个）
图：DOMINANT / CoLA / GCN / BWGNN / XGBGraph / UNPrompt（6 个）

用最小数据集 + 最小 epoch，只验证环境配置、import、fit/predict 能跑。

用法：
    python -m scripts.check_ts_graph_env
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


# ---------------------------------------------------------------------------
# 时序部分
# ---------------------------------------------------------------------------


def _build_ts_algos():
    algos = []
    try:
        from models.timeseries import (
            LSTMAutoEncoderDetector, LSTMSupervisedDetector,
            MatrixProfileDetector, MiniRocketDetector,
        )
        algos.extend([
            ("MatrixProfile", MatrixProfileDetector, {"window_size": 50}, False),
            ("MiniRocket", MiniRocketDetector, {"num_kernels": 500}, True),
            ("LSTM-AE", LSTMAutoEncoderDetector,
             {"epochs": 2, "hidden_size": 16, "batch_size": 16}, False),
            ("LSTM-Sup", LSTMSupervisedDetector,
             {"epochs": 2, "hidden_size": 16, "batch_size": 16}, True),
        ])
    except ImportError as e:
        print(f"  [SKIP TS] cannot import timeseries module: {e}")
    try:
        from models.timeseries.dada import DADADetector
        algos.append(("DADA", DADADetector, {"copies": 3}, False))
    except ImportError as e:
        print(f"  [SKIP DADA] {e}")
    return algos


def check_timeseries() -> list[dict]:
    print(f"\n{'='*70}\nTimeseries environment check\n{'='*70}\n")
    algos = _build_ts_algos()
    if not algos:
        return []

    print("Loading IOPS time-series (window=64, stride=32)...")
    try:
        bundle = load_dataset(
            "276_IOPS_id_17_WebService",
            modality="timeseries",
            window_size=64,
            stride=32,
        )
        X_tr, X_te, y_tr, y_te = bundle.as_tuple()
        train_seq = bundle.extras["raw_values"][:bundle.extras["train_end"]]
        test_seq = bundle.extras["raw_values"][bundle.extras["train_end"]:]
        print(f"  train_windows={X_tr.shape}, test_windows={X_te.shape}, "
              f"test_anomaly_rate={y_te.mean():.4f}")
        # 借异常窗口给训练集（避免 supervised 单类 fail）
        if y_tr.sum() == 0 and y_te.sum() > 0:
            rng = np.random.RandomState(42)
            ai = np.where(y_te == 1)[0]
            n_borrow = min(len(ai) // 3, 30)
            if n_borrow > 0:
                b = rng.choice(ai, n_borrow, replace=False)
                X_tr = np.concatenate([X_tr, X_te[b]])
                y_tr = np.concatenate([y_tr, y_te[b]])
                keep = np.setdiff1d(np.arange(len(y_te)), b)
                X_te, y_te = X_te[keep], y_te[keep]
            print(f"  After borrow: train_anomaly_rate={y_tr.mean():.4f}")
    except Exception as e:
        print(f"  [FAIL] cannot load timeseries: {e}")
        return [{"name": "load_timeseries", "ok": False, "err": str(e)}]

    print()
    results = []
    for name, Cls, kwargs, needs_y in algos:
        t0 = time.perf_counter()
        try:
            det = Cls(contamination=0.1, random_state=42, **kwargs)

            if name == "MatrixProfile":
                det.fit(train_seq)
                scores_seq = det.decision_function(test_seq)
                # 映射到窗口起点
                stride = 32
                n_w = X_te.shape[0]
                starts = np.clip(np.arange(n_w) * stride, 0, scores_seq.size - 1)
                scores = scores_seq[starts]
            else:
                if needs_y:
                    det.fit(X_tr, y_tr)
                else:
                    # 无监督只用正常窗口
                    normal = X_tr[y_tr == 0] if (y_tr == 0).sum() >= 5 else X_tr
                    det.fit(normal)
                scores = det.decision_function(X_te)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = evaluate_all(y_te, scores)
            elapsed = time.perf_counter() - t0
            print(f"  ✓ [{name:>14s}]  AUC-ROC={m['auc_roc']:.4f}  time={elapsed:.2f}s")
            results.append({"name": f"TS/{name}", "ok": True,
                           "auc": m["auc_roc"], "time": elapsed, "err": ""})
        except Exception as e:
            elapsed = time.perf_counter() - t0
            err_msg = f"{type(e).__name__}: {e}"[:150]
            print(f"  ✗ [{name:>14s}]  FAILED  time={elapsed:.2f}s")
            print(f"      {err_msg}")
            results.append({"name": f"TS/{name}", "ok": False,
                           "auc": float("nan"), "time": elapsed, "err": err_msg})

    return results


# ---------------------------------------------------------------------------
# 图部分
# ---------------------------------------------------------------------------


def _build_graph_algos():
    algos = []
    try:
        from models.graph import CoLADetector, DOMINANTDetector
        algos.append(("DOMINANT", DOMINANTDetector,
                     {"epoch": 3, "hid_dim": 16, "num_layers": 2}, False))
        algos.append(("CoLA", CoLADetector,
                     {"epoch": 3, "hid_dim": 16, "num_layers": 2}, False))
    except ImportError as e:
        print(f"  [SKIP PyGOD] {e}")
    try:
        from models.graph.gnn_supervised import (
            BWGNNDetector, GCNDetector, XGBGraphDetector,
        )
        algos.append(("GCN", GCNDetector,
                     {"epochs": 5, "h_feats": 16, "patience": 5}, True))
        algos.append(("BWGNN", BWGNNDetector,
                     {"epochs": 5, "h_feats": 16, "patience": 5}, True))
        algos.append(("XGBGraph", XGBGraphDetector,
                     {"n_estimators": 20, "num_layers": 2}, True))
    except ImportError as e:
        print(f"  [SKIP GADBench] {e}")
    try:
        from models.graph.unprompt import UNPromptDetector
        algos.append(("UNPrompt", UNPromptDetector,
                     {"pretrain_epochs": 5, "prompt_epochs": 5,
                      "embedding_dim": 32}, False))
    except ImportError as e:
        print(f"  [SKIP UNPrompt] {e}")
    return algos


def check_graph() -> list[dict]:
    print(f"\n{'='*70}\nGraph environment check\n{'='*70}\n")
    algos = _build_graph_algos()
    if not algos:
        print("  No graph algorithms available.")
        return []

    print("Loading Reddit graph (smallest, 10k nodes)...")
    try:
        bundle = load_dataset("reddit", modality="graph")
        graph = bundle.extras["graph"]
        masks = bundle.extras["masks"]
        y_all = bundle.extras["y_all"]
        test_idx = np.flatnonzero(masks["test_mask"])
        y_te = y_all[test_idx]
        print(f"  nodes={graph.num_nodes()}, edges={graph.num_edges()}, "
              f"test_anomaly_rate={y_te.mean():.4f}")
    except Exception as e:
        print(f"  [FAIL] cannot load graph: {e}")
        traceback.print_exc()
        return [{"name": "load_graph", "ok": False, "err": str(e)}]

    print()
    results = []
    for name, Cls, kwargs, needs_y in algos:
        t0 = time.perf_counter()
        try:
            det = Cls(contamination=0.1, random_state=42, **kwargs)
            det.fit(graph)
            scores_all = det.decision_function(graph)
            scores_test = np.asarray(scores_all)[test_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = evaluate_all(y_te, scores_test)
            elapsed = time.perf_counter() - t0
            print(f"  ✓ [{name:>10s}]  AUC-ROC={m['auc_roc']:.4f}  time={elapsed:.2f}s")
            results.append({"name": f"GRAPH/{name}", "ok": True,
                           "auc": m["auc_roc"], "time": elapsed, "err": ""})
        except Exception as e:
            elapsed = time.perf_counter() - t0
            err_msg = f"{type(e).__name__}: {e}"[:150]
            print(f"  ✗ [{name:>10s}]  FAILED  time={elapsed:.2f}s")
            print(f"      {err_msg}")
            results.append({"name": f"GRAPH/{name}", "ok": False,
                           "auc": float("nan"), "time": elapsed, "err": err_msg})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ts_results = check_timeseries()
    graph_results = check_graph()
    all_results = ts_results + graph_results

    # 总结
    print(f"\n{'='*70}\nFinal Summary\n{'='*70}")
    print(f"{'Algorithm':<25} {'Status':<10} {'AUC-ROC':<10} {'Time(s)':<10}")
    print("-" * 60)
    for r in all_results:
        status = "✓ OK" if r["ok"] else "✗ FAILED"
        auc = f"{r['auc']:.4f}" if r["ok"] and not np.isnan(r["auc"]) else "—"
        print(f"{r['name']:<25} {status:<10} {auc:<10} {r['time']:<10.2f}")

    n_ok = sum(1 for r in all_results if r["ok"])
    print(f"\n{n_ok}/{len(all_results)} algorithms passed.")
    if n_ok < len(all_results):
        print("\nFailed algorithms:")
        for r in all_results:
            if not r["ok"]:
                print(f"  - {r['name']}: {r['err']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
