"""生成数据集 EDA 摘要。

脚本会扫描服务器上的原始数据目录，汇总 ADBench、TSB-AD、GADBench 的样本数、
特征维度、异常率、路径和预处理策略，并写入 `data/eda_summary.json`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.adbench_adapter import iter_selected_adbench_names, load_adbench_dataset
from adapters.graph_adapter import load_gadbench_dataset
from adapters.timeseries_adapter import list_tsb_files, select_representative_tsb_files


def build_adbench_summary(data_root: Path) -> List[Dict[str, Any]]:
    """生成 ADBench 计划子集的 EDA 摘要。"""

    rows: List[Dict[str, Any]] = []
    for name in iter_selected_adbench_names():
        try:
            bundle = load_adbench_dataset(name, data_root=data_root)
            rows.append({"status": "ok", **bundle.metadata, "name": bundle.name, "modality": bundle.modality})
        except Exception as exc:  # 局部数据缺失时也继续生成剩余数据的摘要。
            rows.append({"status": "missing_or_error", "name": name, "source": "ADBench", "error": str(exc)})
    return rows


def build_tsb_summary(data_root: Path, max_files: int = 8) -> List[Dict[str, Any]]:
    """生成 TSB-AD 文件总览，并写入代表性时序文件清单。"""

    rows: List[Dict[str, Any]] = []
    try:
        selected = select_representative_tsb_files(data_root=data_root, max_files=max_files)
        selected_rel = [str(p) for p in selected]
        selected_path = data_root / "timeseries" / "selected_files.txt"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text("\n".join(selected_rel) + "\n", encoding="utf-8")
        for path in selected:
            rows.append(
                {
                    "status": "selected",
                    "name": path.stem,
                    "source": "TSB-AD-U",
                    "modality": "timeseries",
                    "path": str(path),
                }
            )
        rows.append(
            {
                "status": "available",
                "name": "TSB-AD-U",
                "source": "TSB-AD-U",
                "modality": "timeseries",
                "n_files": len(list_tsb_files(data_root=data_root)),
            }
        )
    except Exception as exc:
        rows.append({"status": "missing_or_error", "name": "TSB-AD-U", "source": "TSB-AD-U", "error": str(exc)})
    return rows


def build_gadbench_summary(data_root: Path) -> List[Dict[str, Any]]:
    """生成 GADBench 图数据的 EDA 摘要。"""

    rows: List[Dict[str, Any]] = []
    for name in ["tfinance", "reddit", "amazon", "weibo"]:
        try:
            bundle = load_gadbench_dataset(name, data_root=data_root, standardize=False)
            rows.append({"status": "ok", **bundle.metadata, "name": name, "modality": "graph"})
        except Exception as exc:
            rows.append({"status": "missing_or_error", "name": name, "source": "GADBench", "modality": "graph", "error": str(exc)})
    return rows


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/eda_summary.json"))
    args = parser.parse_args()

    rows = []
    rows.extend(build_adbench_summary(args.data_root))
    rows.extend(build_tsb_summary(args.data_root))
    rows.extend(build_gadbench_summary(args.data_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
