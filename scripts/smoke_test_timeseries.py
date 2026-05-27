"""任务 14：时序烟雾测试。

使用 TSB-AD 真实时序数据（首选）或合成 sin 波（兜底）跑通：
    MatrixProfileDetector / MiniRocketDetector /
    LSTMAutoEncoderDetector / LSTMSupervisedDetector

文件名约定：``..._tr_<N>_..._.csv``，``tr_<N>`` 是训练段截止点。

用法：
    python -m scripts.smoke_test_timeseries
"""

from __future__ import annotations

import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.logger import log_experiment  # noqa: E402
from eval.metrics import evaluate_all  # noqa: E402
from models.timeseries import (  # noqa: E402
    LSTMAutoEncoderDetector,
    LSTMSupervisedDetector,
    MatrixProfileDetector,
    MiniRocketDetector,
)


SEED = 42
WINDOW_SIZE = 64
STRIDE = 32  # 滑窗步长
LOG_PATH = ROOT / "results" / "experiment_log.csv"
TIMESERIES_DIR = ROOT / "data" / "timeseries"


# ---------------------------------------------------------------------------
# 数据加载（真实 TSB-AD CSV 优先，sin 波合成数据兜底）
# ---------------------------------------------------------------------------


_TR_PAT = re.compile(r"_tr_(\d+)_")


def _parse_tr_index(filename: str) -> int | None:
    m = _TR_PAT.search(filename)
    return int(m.group(1)) if m else None


def load_real_tsbad(filename: str = "276_IOPS_id_17_WebService_tr_19197_1st_19297.csv"):
    csv = TIMESERIES_DIR / filename
    if not csv.exists():
        return None
    tr = _parse_tr_index(filename)
    if tr is None:
        return None
    df = pd.read_csv(csv)
    if not {"Data", "Label"}.issubset(df.columns):
        return None

    seq = df["Data"].to_numpy(dtype=np.float64)
    lbl = df["Label"].to_numpy(dtype=np.int64)

    # 训练段统计量做 z-score
    mu = float(seq[:tr].mean())
    sigma = float(seq[:tr].std()) or 1.0
    seq = (seq - mu) / sigma

    # 切窗：(n_windows, WINDOW_SIZE)；窗口标签 = 该窗口内是否包含 anomaly
    def slide(arr_seq, arr_lbl, start, end):
        windows = []
        labels = []
        for i in range(start, end - WINDOW_SIZE + 1, STRIDE):
            windows.append(arr_seq[i : i + WINDOW_SIZE])
            labels.append(int(arr_lbl[i : i + WINDOW_SIZE].max()))
        return np.array(windows), np.array(labels, dtype=np.int64)

    X_train_full, y_train_full = slide(seq, lbl, 0, tr)
    X_test, y_test = slide(seq, lbl, tr, len(seq))

    # 如果训练段没有异常窗口，从测试段借一些异常窗口给训练（保持 IID 假设的轻微违反，
    # 但烟雾测试以"跑通"为目标）
    if y_train_full.sum() == 0 and y_test.sum() > 0:
        rng = np.random.RandomState(SEED)
        anomaly_idx = np.where(y_test == 1)[0]
        n_borrow = min(len(anomaly_idx) // 3, 50)  # 借不超过 1/3 的异常 / 50 个
        borrow = rng.choice(anomaly_idx, size=n_borrow, replace=False)
        X_train = np.concatenate([X_train_full, X_test[borrow]], axis=0)
        y_train = np.concatenate([y_train_full, y_test[borrow]], axis=0)
        keep = np.setdiff1d(np.arange(X_test.shape[0]), borrow)
        X_test = X_test[keep]
        y_test = y_test[keep]
    else:
        X_train = X_train_full
        y_train = y_train_full

    print(
        f"Loaded {csv.name}\n"
        f"  train_windows={X_train.shape}, test_windows={X_test.shape}\n"
        f"  train anomaly rate={y_train.mean():.4f}, "
        f"test anomaly rate={y_test.mean():.4f}"
    )
    return X_train, X_test, y_train, y_test, seq[:tr], seq[tr:]


def load_synthetic():
    """合成 sin 波 + 异常点。"""
    rng = np.random.RandomState(SEED)
    T = 4000
    t = np.linspace(0, 80 * np.pi, T)
    seq = np.sin(t) + 0.05 * rng.randn(T)
    lbl = np.zeros(T, dtype=np.int64)
    # 注入异常块
    for start in (500, 1500, 2500, 3500):
        end = start + 30
        seq[start:end] += rng.choice([-1, 1]) * 4.0
        lbl[start:end] = 1

    tr = T // 2

    def slide(arr_seq, arr_lbl, start, end):
        windows, labels = [], []
        for i in range(start, end - WINDOW_SIZE + 1, STRIDE):
            windows.append(arr_seq[i : i + WINDOW_SIZE])
            labels.append(int(arr_lbl[i : i + WINDOW_SIZE].max()))
        return np.array(windows), np.array(labels, dtype=np.int64)

    X_train, y_train = slide(seq, lbl, 0, tr)
    X_test, y_test = slide(seq, lbl, tr, T)
    print(
        f"[synthetic] train_windows={X_train.shape}, test_windows={X_test.shape}\n"
        f"  test anomaly rate={y_test.mean():.4f}"
    )
    return X_train, X_test, y_train, y_test, seq[:tr], seq[tr:]


def load_dataset():
    real = load_real_tsbad()
    if real is not None:
        return "TSB-AD-IOPS", real
    print("[WARN] 真实 TSB-AD 数据加载失败，使用合成 sin 波")
    return "Synthetic-Sine", load_synthetic()


# ---------------------------------------------------------------------------
# 算法定义
# ---------------------------------------------------------------------------


def make_algorithms():
    """返回 (name, callable_run) 列表。每个 callable 接收 train/test/raw_seq 并返回结果。"""
    return [
        ("MatrixProfile", _run_matrix_profile),
        ("MiniRocket", _run_minirocket),
        ("LSTM-AE", _run_lstm_ae),
        ("LSTM-Sup", _run_lstm_sup),
    ]


def _run_matrix_profile(X_train, X_test, y_train, y_test, train_seq, test_seq):
    det = MatrixProfileDetector(
        window_size=WINDOW_SIZE, contamination=0.1, random_state=SEED
    )
    t0 = time.perf_counter()
    det.fit(train_seq)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 对测试段直接评分；返回与 test_seq 等长的分数
    scores_seq = det.decision_function(test_seq)
    pred_t = time.perf_counter() - t0

    # 把分数从点级映射到窗口级：取窗口起点位置的分数
    n_windows = X_test.shape[0]
    starts = np.arange(n_windows) * STRIDE
    starts = np.clip(starts, 0, scores_seq.size - 1)
    scores = scores_seq[starts]

    return _evaluate(scores, y_test, fit_t, pred_t)


def _run_minirocket(X_train, X_test, y_train, y_test, train_seq, test_seq):
    det = MiniRocketDetector(
        num_kernels=2000, contamination=0.1, random_state=SEED
    )
    t0 = time.perf_counter()
    det.fit(X_train, y_train)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = det.decision_function(X_test)
    pred_t = time.perf_counter() - t0
    return _evaluate(scores, y_test, fit_t, pred_t)


def _run_lstm_ae(X_train, X_test, y_train, y_test, train_seq, test_seq):
    det = LSTMAutoEncoderDetector(
        hidden_size=32, num_layers=1, epochs=10,
        batch_size=32, lr=1e-3, contamination=0.1, random_state=SEED,
    )
    # LSTM-AE 是无监督，仅用正常窗口（y==0）训练效果更好
    train_normal = X_train[y_train == 0]
    if train_normal.shape[0] < 5:
        train_normal = X_train  # 退化到全部窗口

    t0 = time.perf_counter()
    det.fit(train_normal)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = det.decision_function(X_test)
    pred_t = time.perf_counter() - t0
    return _evaluate(scores, y_test, fit_t, pred_t)


def _run_lstm_sup(X_train, X_test, y_train, y_test, train_seq, test_seq):
    if y_train.sum() == 0 or y_train.sum() == y_train.shape[0]:
        raise RuntimeError("LSTM-Sup 需要训练集同时包含正常和异常窗口")

    det = LSTMSupervisedDetector(
        hidden_size=32, num_layers=1, epochs=10,
        batch_size=32, lr=1e-3, contamination=0.1, random_state=SEED,
    )
    t0 = time.perf_counter()
    det.fit(X_train, y_train)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = det.decision_function(X_test)
    pred_t = time.perf_counter() - t0
    return _evaluate(scores, y_test, fit_t, pred_t)


def _evaluate(scores, y_test, fit_t, pred_t):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_all(y_test, scores)
    return {
        "auc_roc": m["auc_roc"],
        "auc_pr": m["auc_pr"],
        "f1_best": m["f1_best"],
        "fit_time": fit_t,
        "predict_time": pred_t,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dataset_name, (X_train, X_test, y_train, y_test, train_seq, test_seq) = load_dataset()
    print()

    results = []
    total_t0 = time.perf_counter()

    for name, fn in make_algorithms():
        try:
            r = fn(X_train, X_test, y_train, y_test, train_seq, test_seq)
            r["name"] = name
            print(
                f"  [{name:>14s}]  AUC-ROC={r['auc_roc']:.4f}  "
                f"AUC-PR={r['auc_pr']:.4f}  F1={r['f1_best']:.4f}  "
                f"fit={r['fit_time']:.2f}s  pred={r['predict_time']:.3f}s"
            )
        except Exception as e:
            r = {
                "name": name,
                "auc_roc": float("nan"),
                "auc_pr": float("nan"),
                "f1_best": float("nan"),
                "fit_time": float("nan"),
                "predict_time": float("nan"),
                "notes": (f"FAILED: {e!r}")[:200],
            }
            print(f"  [{name:>14s}]  FAILED: {e!r}"[:120])
            tb = traceback.format_exc().splitlines()[-3:]
            for line in tb:
                print(f"    {line}")

        results.append(r)
        log_experiment(
            dataset_name=dataset_name,
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

    print("\n" + "=" * 78)
    print(f"Summary: {dataset_name}")
    print("=" * 78)
    for r in results:
        print(
            f"{r['name']:<14} | AUC-ROC={r['auc_roc']:.4f}  AUC-PR={r['auc_pr']:.4f}  "
            f"F1={r['f1_best']:.4f}  | {r['notes'][:40]}"
        )
    n_ok = sum(1 for r in results if not np.isnan(r["auc_roc"]))
    print(f"\n{n_ok}/{len(results)} algorithms succeeded in {total:.1f}s.")


if __name__ == "__main__":
    main()
