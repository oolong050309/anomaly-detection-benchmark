"""任务 15：图烟雾测试。

优先尝试加载 GADBench 真实图（DGL 二进制）；本地受 DGL Windows 安装限制
时，使用 PyG 内置 ``KarateClub`` + 随机注入异常节点作为兜底。

依赖 ``pyg-lib`` / ``torch-sparse`` 才能让 PyGOD 跑通；本地缺这两个包时，
脚本会捕获错误并写入 notes，不影响其他任务。

用法：
    python -m scripts.smoke_test_graph
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


SEED = 42
LOG_PATH = ROOT / "results" / "experiment_log.csv"


def _has_pygod_runtime() -> bool:
    """检测 PyGOD 是否能实际工作（需要 pyg-lib 或 torch-sparse）。"""
    try:
        import pygod  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    try:
        import pyg_lib  # noqa: F401
        return True
    except ImportError:
        try:
            import torch_sparse  # noqa: F401
            return True
        except ImportError:
            return False


def load_synthetic_graph():
    """PyG KarateClub + 随机异常注入作为兜底数据。"""
    import torch
    from torch_geometric.datasets import KarateClub

    rng = np.random.RandomState(SEED)
    ds = KarateClub()
    g = ds[0]  # Data 对象，34 节点

    # 注入异常：随机选 5 个节点把特征加大噪声
    n = g.x.shape[0]
    n_anom = max(1, int(0.15 * n))
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    perturb = torch.from_numpy(rng.randn(n_anom, g.x.shape[1]).astype(np.float32) * 5.0)
    g.x[anom_idx] += perturb

    # ground truth label
    y_true = np.zeros(n, dtype=np.int64)
    y_true[anom_idx] = 1
    g.y = torch.from_numpy(y_true)

    print(
        f"[synthetic-karate] nodes={n}, edges={g.edge_index.shape[1]}, "
        f"anomaly_rate={y_true.mean():.4f}"
    )
    return "Synthetic-Karate", g, y_true


def load_dataset():
    """目前只用合成数据。GADBench 真实图加载需要 DGL，留给服务器 spec。"""
    return load_synthetic_graph()


def main() -> None:
    print(f"PyGOD runtime ready? {_has_pygod_runtime()}")
    if not _has_pygod_runtime():
        print(
            "[INFO] PyGOD 需要 pyg-lib 或 torch-sparse 才能采样邻居；"
            "本地 Windows 未安装这两个包，脚本会跑失败但仍记录到日志。"
        )

    from eval.logger import log_experiment
    from eval.metrics import evaluate_all
    from models.graph import CoLADetector, DOMINANTDetector

    dataset_name, graph, y_true = load_dataset()

    algos = [
        ("DOMINANT", DOMINANTDetector, {"hid_dim": 16, "num_layers": 2, "epoch": 5}),
        ("CoLA", CoLADetector, {"hid_dim": 16, "num_layers": 2, "epoch": 5}),
    ]

    results = []
    total_t0 = time.perf_counter()
    for name, Cls, kwargs in algos:
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
            det = Cls(contamination=0.15, random_state=SEED, **kwargs)
            t0 = time.perf_counter()
            det.fit(graph)
            result["fit_time"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = det.decision_function(graph)
            result["predict_time"] = time.perf_counter() - t0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = evaluate_all(y_true, scores)
            result["auc_roc"] = m["auc_roc"]
            result["auc_pr"] = m["auc_pr"]
            result["f1_best"] = m["f1_best"]
            print(
                f"  [{name:>10s}]  AUC-ROC={result['auc_roc']:.4f}  "
                f"AUC-PR={result['auc_pr']:.4f}  F1={result['f1_best']:.4f}  "
                f"fit={result['fit_time']:.2f}s"
            )
        except Exception as e:
            result["notes"] = (f"FAILED: {e!r}")[:200]
            print(f"  [{name:>10s}]  FAILED: {e!r}"[:120])
            tb = traceback.format_exc().splitlines()[-3:]
            for line in tb:
                print(f"    {line}")

        results.append(result)
        log_experiment(
            dataset_name=dataset_name,
            algorithm_name=name,
            auc_roc=result["auc_roc"],
            auc_pr=result["auc_pr"],
            f1_best=result["f1_best"],
            fit_time_sec=result["fit_time"],
            predict_time_sec=result["predict_time"],
            seed=SEED,
            notes=result["notes"],
            log_path=str(LOG_PATH),
        )

    total = time.perf_counter() - total_t0
    print("\n" + "=" * 70)
    print(f"Summary: {dataset_name}")
    print("=" * 70)
    for r in results:
        print(
            f"{r['name']:<10} | AUC-ROC={r['auc_roc']:.4f}  AUC-PR={r['auc_pr']:.4f}  "
            f"F1={r['f1_best']:.4f}  | {r['notes'][:50]}"
        )
    n_ok = sum(1 for r in results if not np.isnan(r["auc_roc"]))
    print(f"\n{n_ok}/{len(results)} algorithms succeeded in {total:.1f}s.")


if __name__ == "__main__":
    main()
