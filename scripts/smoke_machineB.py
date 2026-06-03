"""机器 B 轻量冒烟测试：验证 pyod 2.x 升级后，图 / 时序专用算法不崩。

只测「能否 fit + 出分数」，不看精度：
- epoch 类参数强制压到 2，num_kernels 压到 200，加速到秒级
- 每类只用一个最小数据集
- 每个算法独立 try/except，打印 OK / ERR，不中断

用法（机器 B）：
    export CUDA_VISIBLE_DEVICES=0
    python -m scripts.smoke_machineB                 # 图 + 时序都测
    python -m scripts.smoke_machineB --only graph
    python -m scripts.smoke_machineB --only timeseries
    python -m scripts.smoke_machineB --graph-ds amazon --ts-ds 276_IOPS_id_17_WebService
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset
from experiments.exp1_baseline import (
    _get_graph_algos,
    _get_timeseries_algos,
    _borrow_for_supervised,
)

SEED = 42
# 把所有 epoch 类参数压到这么小，纯粹验证不崩
EPOCH_KEYS = {"epoch", "epochs", "pretrain_epochs", "prompt_epochs", "epoch_num"}
SMOKE_EPOCH = 2


def _shrink(kwargs: dict) -> dict:
    """复制 kwargs，把 epoch 类压到 SMOKE_EPOCH，num_kernels 压小，加速冒烟。"""
    out = dict(kwargs)
    for k in list(out.keys()):
        if k in EPOCH_KEYS:
            out[k] = SMOKE_EPOCH
        if k == "num_kernels":
            out[k] = 200
    return out


def smoke_graph(ds: str) -> None:
    print(f"\n{'='*60}\n[GRAPH] dataset={ds}\n{'='*60}")
    try:
        bundle = load_dataset(ds, modality="graph")
        graph = bundle.extras["graph"]
        masks = bundle.extras["masks"]
        y_all = bundle.extras["y_all"]
        test_idx = np.flatnonzero(masks["test_mask"])
        print(f"  loaded: nodes={graph.num_nodes()} test={len(test_idx)}")
    except Exception as e:
        print(f"  [FATAL] cannot load graph {ds}: {e!r}")
        return

    for name, Cls, kwargs, needs_y in _get_graph_algos():
        kw = _shrink(kwargs)
        t0 = time.perf_counter()
        try:
            det = Cls(contamination=0.1, random_state=SEED, **kw)
            det.fit(graph)
            scores = np.asarray(det.decision_function(graph))
            s_test = scores[test_idx]
            dt = time.perf_counter() - t0
            assert s_test.shape[0] == len(test_idx)
            print(f"  OK  {name:12s} scores={s_test.shape} t={dt:.1f}s kw={kw}")
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  ERR {name:12s} t={dt:.1f}s {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)


def smoke_timeseries(ds: str) -> None:
    print(f"\n{'='*60}\n[TIMESERIES] dataset={ds}\n{'='*60}")
    try:
        bundle = load_dataset(ds, modality="timeseries",
                              window_size=100, stride=10)
        X_tr, X_te, y_tr, y_te = bundle.as_tuple()
        train_seq = bundle.extras["raw_values"][: bundle.extras["train_end"]]
        test_seq = bundle.extras["raw_values"][bundle.extras["train_end"]:]
        print(f"  loaded: train_win={X_tr.shape} test_win={X_te.shape}")
    except Exception as e:
        print(f"  [FATAL] cannot load timeseries {ds}: {e!r}")
        return

    for name, Cls, kwargs, needs_y in _get_timeseries_algos():
        kw = _shrink(kwargs)
        t0 = time.perf_counter()
        try:
            det = Cls(contamination=0.1, random_state=SEED, **kw)
            if name == "MatrixProfile":
                # 用原始一维序列
                det.fit(train_seq)
                scores = np.asarray(det.decision_function(test_seq))
            elif needs_y:
                X_fit, y_fit = X_tr, y_tr
                if len(np.unique(y_fit)) < 2:
                    X_fit, y_fit, *_ = _borrow_for_supervised(
                        X_tr, y_tr, X_te, y_te, rng_seed=SEED
                    )
                det.fit(X_fit, y_fit)
                scores = np.asarray(det.decision_function(X_te))
            else:
                # 无监督只用正常窗口
                mask = y_tr == 0
                det.fit(X_tr[mask] if mask.sum() >= 5 else X_tr)
                scores = np.asarray(det.decision_function(X_te))
            dt = time.perf_counter() - t0
            print(f"  OK  {name:12s} scores={scores.shape} t={dt:.1f}s kw={kw}")
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  ERR {name:12s} t={dt:.1f}s {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="机器B 轻量冒烟（图+时序专用算法）")
    parser.add_argument("--only", choices=["graph", "timeseries"], default=None)
    parser.add_argument("--graph-ds", default="amazon")
    parser.add_argument("--ts-ds", default="276_IOPS_id_17_WebService")
    args = parser.parse_args()

    if args.only in (None, "graph"):
        smoke_graph(args.graph_ds)
    if args.only in (None, "timeseries"):
        smoke_timeseries(args.ts_ds)

    print(f"\n{'='*60}\n冒烟结束。看上面每行 OK / ERR。\n{'='*60}")


if __name__ == "__main__":
    main()
