"""数据集筛选脚本：从原始 benchmark 中挑出本项目计划使用的子集。

执行后会把目标数据复制到 ``final_project/data/<modality>/`` 下，
并写一份 ``data/selected_files.json`` 清单，供 adapter 加载与上传到服务器使用。

用法：
    python scripts/select_datasets.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 原始 benchmark 在仓库根目录之外（D:\homework\datamining\）
WORKSPACE_ROOT = PROJECT_ROOT.parent

ADBENCH_ROOT = (
    WORKSPACE_ROOT / "ADBench-main" / "ADBench-main" / "adbench" / "datasets"
)
TSB_AD_U_ROOT = WORKSPACE_ROOT / "TSB-AD-U" / "TSB-AD-U"
GADBENCH_ROOT = (
    WORKSPACE_ROOT / "GADBench-master" / "GADBench-master" / "datasets" / "datasets"
)

# ---------------------------------------------------------------------------
# 选定的数据集（与团队分工文档保持一致）
# ---------------------------------------------------------------------------

# ADBench 表格：9 个
TABULAR_FILES = [
    "6_cardio.npz",          # Cardio
    "38_thyroid.npz",        # Thyroid
    "30_satellite.npz",      # Satellite
    "32_shuttle.npz",        # Shuttle
    "13_fraud.npz",          # Credit Card (fraud)
    "29_Pima.npz",           # Pima
    "2_annthyroid.npz",      # Annthyroid
    "23_mammography.npz",    # Mammography
    "28_pendigits.npz",      # Pendigits
]

# ADBench CV：4 个
CV_FILES = [
    "CIFAR10_0.npz",
    "CIFAR10_1.npz",
    "FashionMNIST_0.npz",
    "FashionMNIST_1.npz",
]

# ADBench NLP：4 个
NLP_FILES = [
    "20news_0.npz",
    "20news_1.npz",
    "agnews_0.npz",
    "amazon.npz",
]

# GADBench 图：4 个（DGL 二进制无扩展名）
GRAPH_FILES = [
    "tfinance",   # T-Finance
    "reddit",     # Reddit
    "amazon",     # Amazon
    "weibo",      # Weibo
]

# 时序：按领域分层抽样 8 条
# 优先选每个域里训练段较长（tr_>=1000）的文件，便于建模
TIMESERIES_DOMAIN_QUOTA = {
    "Facility": 1,
    "Medical": 1,
    "Sensor": 1,
    "WebService": 1,
    "Finance": 1,
    "Traffic": 1,
    "HumanActivity": 1,
    "Synthetic": 1,
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

# 文件名形如 ``001_NAB_id_1_Facility_tr_1007_1st_2014.csv``
# 域字段位于 ``_tr_`` 之前。
_TR_PATTERN = re.compile(r"_([A-Za-z]+)_tr_(\d+)_")


def parse_timeseries_filename(name: str) -> tuple[str | None, int | None]:
    """从 TSB-AD 文件名中解析 (domain, train_length)。"""
    m = _TR_PATTERN.search(name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def select_timeseries(quota: dict[str, int]) -> list[str]:
    """按域配额抽样：每个域优先取训练段较长（tr_ 较大）的第一条。"""
    files = sorted(p.name for p in TSB_AD_U_ROOT.glob("*.csv"))
    by_domain: dict[str, list[tuple[int, str]]] = {}
    for name in files:
        domain, tr_len = parse_timeseries_filename(name)
        if domain is None or domain not in quota:
            continue
        by_domain.setdefault(domain, []).append((tr_len or 0, name))

    chosen: list[str] = []
    for domain, n in quota.items():
        # 训练段从长到短排序；同样长度时按文件名排序保证可复现
        candidates = sorted(by_domain.get(domain, []), key=lambda x: (-x[0], x[1]))
        if not candidates:
            print(f"  [WARN] domain {domain} 没有候选文件")
            continue
        chosen.extend(name for _, name in candidates[:n])
    return chosen


def copy_files(
    src_dir: Path, dst_dir: Path, names: list[str], label: str
) -> list[str]:
    """复制文件到目标目录，已存在则跳过。返回实际成功复制的文件名。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for name in names:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            missing.append(name)
            continue
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            copied.append(name)
            continue
        shutil.copy2(src, dst)
        copied.append(name)
    print(f"  [{label}] 复制 {len(copied)}/{len(names)} 个文件 -> {dst_dir}")
    if missing:
        print(f"  [{label}] 缺失文件: {missing}")
    return copied


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"原始数据根目录: {WORKSPACE_ROOT}\n")

    manifest: dict[str, list[str]] = {}

    print("[1/5] 复制 ADBench 表格数据集")
    manifest["tabular"] = copy_files(
        ADBENCH_ROOT / "Classical",
        DATA_ROOT / "tabular",
        TABULAR_FILES,
        label="tabular",
    )

    print("\n[2/5] 复制 ADBench CV 数据集（ResNet18 特征）")
    manifest["cv"] = copy_files(
        ADBENCH_ROOT / "CV_by_ResNet18",
        DATA_ROOT / "cv",
        CV_FILES,
        label="cv",
    )

    print("\n[3/5] 复制 ADBench NLP 数据集（BERT 特征）")
    manifest["nlp"] = copy_files(
        ADBENCH_ROOT / "NLP_by_BERT",
        DATA_ROOT / "nlp",
        NLP_FILES,
        label="nlp",
    )

    print("\n[4/5] 按域分层抽样 TSB-AD 时序数据")
    ts_chosen = select_timeseries(TIMESERIES_DOMAIN_QUOTA)
    manifest["timeseries"] = copy_files(
        TSB_AD_U_ROOT,
        DATA_ROOT / "timeseries",
        ts_chosen,
        label="timeseries",
    )

    print("\n[5/5] 复制 GADBench 图数据集")
    manifest["graph"] = copy_files(
        GADBENCH_ROOT,
        DATA_ROOT / "graph",
        GRAPH_FILES,
        label="graph",
    )

    manifest_path = DATA_ROOT / "selected_files.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n清单已写入: {manifest_path}")

    total = sum(len(v) for v in manifest.values())
    print(f"\n汇总：共 {total} 个数据集")
    for k, v in manifest.items():
        print(f"  {k:12s}: {len(v):2d} 个")


if __name__ == "__main__":
    main()
