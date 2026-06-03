"""把 GADBench 图数据集导出成 .npz（在有 DGL 的机器 B 上运行）。

通用算法（Exp-2 / Exp-3）跑图模态时只用到节点特征矩阵
(X_train, X_test, y_train, y_test)，并不需要 DGL 图对象本身。
本脚本用 DGL 把这些矩阵 + mask 提取出来存成 .npz，拷到没有 DGL 的
机器 A 后，graph_adapter 会自动从 npz 直读（见 load_gadbench_dataset
的 npz 回退分支），从而让机器 A 也能跑 Exp-2 / Exp-3 的图模态。

用法（机器 B）：
    python -m scripts.export_graph_npz
    python -m scripts.export_graph_npz --datasets reddit amazon
    python -m scripts.export_graph_npz --out-dir data/graph_npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset

DEFAULT_DATASETS = ["tfinance", "reddit", "amazon", "weibo"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GADBench graphs to .npz")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "graph_npz"),
        help="输出目录，默认 data/graph_npz/。每个数据集一个 <name>.npz。",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for name in args.datasets:
        try:
            b = load_dataset(name, modality="graph", data_root=args.data_root)
            X_tr, X_te, y_tr, y_te = b.as_tuple()
            out_path = out_dir / f"{name}.npz"
            np.savez_compressed(
                out_path,
                X_train=np.asarray(X_tr, dtype=np.float64),
                X_test=np.asarray(X_te, dtype=np.float64),
                y_train=np.asarray(y_tr, dtype=np.int64),
                y_test=np.asarray(y_te, dtype=np.int64),
            )
            print(
                f"OK  {name:10s} -> {out_path}  "
                f"train={X_tr.shape} test={X_te.shape} "
                f"train_anom={np.mean(y_tr):.4f} test_anom={np.mean(y_te):.4f}"
            )
            ok += 1
        except Exception as e:
            print(f"ERR {name:10s} {e!r}")
            fail += 1

    print(f"\nDone: {ok} exported, {fail} failed. Output dir: {out_dir}")
    print("把整个目录拷到机器 A 的同一相对路径 (data/graph_npz/) 即可。")


if __name__ == "__main__":
    main()
