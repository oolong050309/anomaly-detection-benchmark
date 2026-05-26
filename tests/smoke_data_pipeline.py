"""数据管道轻量自检脚本。

用于在服务器上快速确认：统一加载入口可用、污染注入器可用、训练/测试四元组
形状合理。它不是完整测试集，只是交付前的 smoke test。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters import load_dataset
from data.contaminate import contaminate_supervised, contaminate_unsupervised


def main() -> None:
    """执行单个数据集加载和污染注入自检。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="cardio")
    args = parser.parse_args()

    bundle = load_dataset(args.dataset, data_root=args.data_root)
    X_train, X_test, y_train, y_test = bundle.as_tuple()
    print(bundle.name, bundle.modality, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_unsup, y_unsup, meta_unsup = contaminate_unsupervised(X_train, y_train, 0.05)
    _, y_sup, meta_sup = contaminate_supervised(X_train, y_train, 0.1)
    print("unsupervised", X_unsup.shape, y_unsup.shape, meta_unsup)
    print("supervised", y_sup.shape, meta_sup)


if __name__ == "__main__":
    main()
