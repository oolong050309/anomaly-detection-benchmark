"""派生汇总脚本：把极端比例数据集和正常数据集分组统计。

Bug 2 修复（Requirement 2）：原始 ``results/exp1_results.csv`` 不动，本脚本
读 CSV → 加 ``extreme_imbalance`` 列 → 分 ``all/normal_only/extreme_only``
三组按算法聚合 mean/median，写出 annotated CSV 与 markdown 汇总。

CLI 用法
--------
::

    python -m scripts.analyze_results \\
        --input results/exp1_results.csv \\
        --modality timeseries \\
        --output results/exp1_summary_timeseries.md

详细设计：``.kiro/specs/ad-timeseries-exp1-fixes/design.md``
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = ("auc_roc", "auc_pr", "f1_best")


# ---------------------------------------------------------------------------
# CSV 清洗（解决多次跑后产生的重复行 / 旧 schema 行 / dataset 名前缀漂移）
# ---------------------------------------------------------------------------


def _strip_numeric_prefix(name: str) -> str:
    """``6_cardio`` → ``cardio``；无前缀的串原样返回。"""

    return re.sub(r"^\d+_", "", str(name), count=1)


def _build_dataset_name_map(names: pd.Series) -> Dict[str, str]:
    """构造 ``raw_name → canonical_name`` 映射。

    数据驱动：仅当 ``6_cardio`` 和 ``cardio`` **同时出现**时把前者折叠成后者。
    像 ``006_NAB_id_6_Traffic`` 这种没有同名「裸版」的不动。
    """

    distinct = pd.Series(names).dropna().astype(str).unique().tolist()
    distinct_set = set(distinct)
    mapping: Dict[str, str] = {}
    prefix_re = re.compile(r"^\d+_")
    for name in distinct:
        if prefix_re.match(name):
            stripped = _strip_numeric_prefix(name)
            if stripped != name and stripped in distinct_set:
                mapping[name] = stripped
    return mapping


def _rename_algorithm_column(df: pd.DataFrame) -> pd.DataFrame:
    """logger 写的是 ``algorithm_name``；汇总函数期望的是 ``algorithm``。

    若两者都不在则返回原 df；若只有 ``algorithm_name`` 则补一个 ``algorithm`` 别名
    （保持 ``algorithm_name`` 原列不动，方便回查）。
    """

    if "algorithm" in df.columns or "algorithm_name" not in df.columns:
        return df
    out = df.copy()
    out["algorithm"] = out["algorithm_name"]
    return out


def clean_results(
    df: pd.DataFrame,
    *,
    drop_legacy_schema: bool = True,
    drop_failed: bool = True,
    normalize_dataset_names: bool = True,
    deduplicate: bool = True,
    add_algorithm_alias: bool = True,
) -> pd.DataFrame:
    """返回 ``df`` 清洗后的副本。每一步都可单独关掉。

    步骤：

    1. ``drop_legacy_schema`` — 丢弃 ``experiment_name`` 为空的旧 schema 行
       （早期 logger 写出来缺一半字段，参与聚合会 NaN 污染）。
    2. ``drop_failed`` — 仅保留 ``status == 'success'``（如 TabPFN 尺寸超限的
       failed 行，AUC/F1 全 NaN，不应进均值）。
    3. ``normalize_dataset_names`` — 仅当 ``6_cardio`` 和 ``cardio`` 同时存在
       时把前者折叠成后者；时序的 ``006_NAB_id_6_Traffic`` 等保留前缀。
    4. ``deduplicate`` — 按 ``(dataset_name, algorithm_name, modality)`` 去重，
       保留 ``timestamp`` 最新一行。
    5. ``add_algorithm_alias`` — 若只有 ``algorithm_name`` 列，补一个
       ``algorithm`` 别名给下游 ``summarize_by_group`` 用。

    输入 df 不被修改。
    """

    out = df.copy()

    if drop_legacy_schema and "experiment_name" in out.columns:
        legacy = out["experiment_name"].isna() | (
            out["experiment_name"].astype(str).str.strip() == ""
        )
        if legacy.any():
            out = out.loc[~legacy].reset_index(drop=True)

    if drop_failed and "status" in out.columns:
        keep = out["status"].astype(str).str.strip().str.lower() == "success"
        out = out.loc[keep].reset_index(drop=True)

    if normalize_dataset_names and "dataset_name" in out.columns:
        mapping = _build_dataset_name_map(out["dataset_name"])
        if mapping:
            out["dataset_name"] = out["dataset_name"].astype(str).map(
                lambda n: mapping.get(n, n)
            )

    if deduplicate and {"dataset_name", "algorithm_name"}.issubset(out.columns):
        if "timestamp" in out.columns:
            ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
            order = ts.argsort(kind="stable")
            out = out.iloc[order].reset_index(drop=True)
        keys = ["dataset_name", "algorithm_name"]
        if "modality" in out.columns:
            keys.append("modality")
        out = out.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    if add_algorithm_alias:
        out = _rename_algorithm_column(out)

    return out


def annotate_extreme_imbalance(
    df: pd.DataFrame, lo: float = 0.01, hi: float = 0.99
) -> pd.DataFrame:
    """加 ``extreme_imbalance`` 列。

    阈值：``train_anomaly_rate`` 或 ``test_anomaly_rate`` 严格落在
    ``(lo, hi)`` 之外即为 extreme。边界值（恰好等于 lo 或 hi）算正常。

    纯函数，原 df 不变。
    """
    out = df.copy()
    tr = out.get("train_anomaly_rate")
    te = out.get("test_anomaly_rate")
    if tr is None or te is None:
        warnings.warn(
            "train_anomaly_rate / test_anomaly_rate 列缺失，extreme_imbalance 全填 False"
        )
        out["extreme_imbalance"] = False
        return out

    tr_num = pd.to_numeric(tr, errors="coerce")
    te_num = pd.to_numeric(te, errors="coerce")
    out["extreme_imbalance"] = (
        (tr_num < lo) | (tr_num > hi) | (te_num < lo) | (te_num > hi)
    ).fillna(False).astype(bool)
    return out


def summarize_by_group(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """按 ``all / normal_only / extreme_only`` 三组分别按 algorithm 聚合。

    要求 df 已经包含 ``extreme_imbalance`` 列。聚合的指标列只取 df 中实际存在的
    METRICS 子集（兼容老 CSV）。
    """
    if "extreme_imbalance" not in df.columns:
        raise ValueError("df 缺少 extreme_imbalance 列；先调用 annotate_extreme_imbalance")
    if "algorithm" not in df.columns:
        raise ValueError("df 缺少 algorithm 列")

    metrics = [m for m in METRICS if m in df.columns]
    if not metrics:
        raise ValueError(f"df 中没有可用指标列；期望至少含 {METRICS} 之一")

    # 仅保留 success 行（如果存在 status 列）
    if "status" in df.columns:
        df_use = df[df["status"].astype(str) == "success"].copy()
    else:
        df_use = df.copy()

    # 数值化，保证 groupby.agg 不报 dtype 错
    for m in metrics:
        df_use[m] = pd.to_numeric(df_use[m], errors="coerce")

    groups = {
        "all":          df_use,
        "normal_only":  df_use[~df_use["extreme_imbalance"].astype(bool)],
        "extreme_only": df_use[df_use["extreme_imbalance"].astype(bool)],
    }
    out: Dict[str, pd.DataFrame] = {}
    for name, sub in groups.items():
        if len(sub) == 0 or "algorithm" not in sub.columns:
            out[name] = pd.DataFrame()
            continue
        agg = sub.groupby("algorithm")[metrics].agg(["mean", "median"]).round(4)
        out[name] = agg
    return out


def render_markdown(summaries: Dict[str, pd.DataFrame], modality: str = "") -> str:
    """把三组聚合表渲染成 markdown 字符串。"""
    parts = [f"# Exp-1 派生汇总（modality={modality or 'all'}）\n"]
    parts.append(
        "极端比例数据集（train/test anomaly_rate ∉ [0.01, 0.99]）单独分组，"
        "避免污染算法平均排名。\n"
    )
    for name, table in summaries.items():
        parts.append(f"\n## Group: {name}\n")
        if table.empty:
            parts.append("_（该组无数据）_\n")
            continue
        try:
            md = table.to_markdown()
        except (ImportError, AttributeError):
            md = table.to_string()
        parts.append(md + "\n")
    return "\n".join(parts)


def _filter_modality(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    if modality == "all" or "modality" not in df.columns:
        return df
    return df[df["modality"].astype(str) == modality].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp-1 派生汇总：标注极端比例数据集，分组聚合算法指标。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="原始 results CSV 路径（不会被修改）。",
    )
    parser.add_argument(
        "--modality",
        default="timeseries",
        choices=["tabular", "timeseries", "graph", "all"],
        help="只对该模态汇总；'all' 不过滤。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="markdown 汇总输出路径。",
    )
    parser.add_argument(
        "--annotated-csv",
        default=None,
        help="annotated CSV 输出路径；缺省与 --input 同目录、加 .annotated.csv 后缀。",
    )
    parser.add_argument("--lo", type=float, default=0.01)
    parser.add_argument("--hi", type=float, default=0.99)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="跳过清洗（保留旧 schema 行 / failed 行 / 重复行 / 前缀名）。",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="清洗时保留 status='failed' 的行。",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_path}")

    df_raw = pd.read_csv(input_path)
    if not args.no_clean:
        df_raw = clean_results(df_raw, drop_failed=not args.keep_failed)
    df_mod = _filter_modality(df_raw, args.modality)
    df = annotate_extreme_imbalance(df_mod, lo=args.lo, hi=args.hi)

    annotated_csv = (
        Path(args.annotated_csv)
        if args.annotated_csv
        else input_path.with_suffix(".annotated.csv")
    )
    annotated_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(annotated_csv, index=False)

    summaries = summarize_by_group(df)
    md = render_markdown(summaries, modality=args.modality)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    n_all = len(df)
    n_extreme = int(df["extreme_imbalance"].sum())
    print(f"[analyze_results] 模态={args.modality}  rows={n_all}  extreme={n_extreme}")
    print(f"[analyze_results] annotated CSV → {annotated_csv}")
    print(f"[analyze_results] markdown 汇总 → {output_path}")


if __name__ == "__main__":
    main()
