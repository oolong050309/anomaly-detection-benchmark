"""Streamlit 界面共享工具：路径解析、结果加载、图表辅助。"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay, auc, roc_curve

from eval.analysis_utils import (
    latest_per_run_key,
    load_artifact_npz,
    parameter_proxy,
    read_experiment,
    successful_runs,
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 与项目计划书对齐的元信息
# ---------------------------------------------------------------------------
PROJECT_TITLE = "异常检测算法系统性对比研究"
PROJECT_TYPE = "选项 B — 算法系统性对比研究"
TEAM_MEMBERS = [
    ("徐云鹏", "2353583", "数据工程"),
    ("杨景翔", "2351576", "算法建模"),
    ("陈艺龙", "2352359", "评估分析"),
    ("李雪菲", "2354093", "工程交付"),
]
N_ALGORITHMS = 26
N_MODALITIES = 5
N_DATASETS_PLANNED = 26
CONTAMINATION_RATES = [0.0, 0.01, 0.05, 0.10, 0.20]
DEFAULT_SEEDS = [41, 42, 43]

MODALITY_LABELS = {
    "tabular": "表格",
    "cv": "计算机视觉 (CV)",
    "nlp": "自然语言 (NLP)",
    "timeseries": "时序",
    "graph": "图",
}

# 统一数据根 `data/` 下的形态子目录（与 selected_files.json 对齐）
MODALITY_DATA_SUBDIRS: dict[str, str] = {
    "tabular": "tabular",
    "cv": "cv",
    "nlp": "nlp",
    "timeseries": "timeseries",
    "graph": "graph",
}

PLAN_TIMESERIES_SHORT = [
    "006_NAB_id_6_Traffic",
    "149_Stock_id_1_Finance",
    "171_MITDB_id_2_Medical",
    "225_MGAB_id_1_Synthetic",
    "276_IOPS_id_17_WebService",
    "331_UCR_id_29_Facility",
    "337_UCR_id_35_HumanActivity",
    "550_SWaT_id_1_Sensor",
]

BENCHMARK_LABELS = {
    "ADBench": "ADBench (NeurIPS 2022)",
    "TSB-AD": "TSB-AD (NeurIPS 2024)",
    "GADBench": "GADBench (NeurIPS 2023)",
}

METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "f1_best": "F1@best",
}

METRIC_HINTS = {
    "auc_roc": "受试者工作特征曲线下面积，衡量整体排序能力，但在极度不平衡下偏乐观",
    "auc_pr": "精确率-召回率曲线下面积，对正类（异常）更敏感，是不平衡场景的主推指标",
    "f1_best": "遍历阈值取最优 F1，反映在最佳工作点上的精确率/召回率平衡",
}

METRIC_RECOMMENDED = "auc_pr"

ALGO_CATEGORIES: dict[str, list[str]] = {
    "统计/浅层集成": ["IQR", "ECOD", "COPOD"],
    "近邻/树": ["KNN", "LOF", "IForest"],
    "边界": ["OCSVM"],
    "监督集成": ["LR", "RF", "MLP", "XGBoost", "LightGBM", "TabPFN"],
    "深度重构": ["AutoEncoder", "DeepSVDD"],
    "时序专用": ["MatrixProfile", "MiniRocket", "LSTM-AE", "LSTM-Sup", "DADA"],
    "图专用": ["DOMINANT", "CoLA", "GCN", "BWGNN", "XGBGraph", "UNPrompt"],
}

SUPERVISED_ALGOS = {
    "XGBoost", "LightGBM", "RF", "LR", "MLP", "TabPFN",
    "XGBGraph", "BWGNN", "GCN", "LSTM-Sup", "LSTM", "MiniRocket",
}

RESEARCH_QUESTIONS = [
    ("Exp-1 基线对比", "干净数据下各算法在其适配模态内的精度排名如何？深度方法是否优于经典方法？"),
    ("Exp-2 污染鲁棒性", "污染率 {0,1,5,10,20}% 时，无监督保留异常 vs 有监督标签翻转的退化规律有何差异？"),
    ("Exp-3 跨模态泛化", "同一通用算法在 5 类形态上的排名是否稳定？哪些算法跨模态泛化更强？"),
    ("Exp-4 鲁棒防御", "对称标签翻转下，无监督去噪器 + Trim/Flip 策略能否挽救有监督模型？Trim 是否优于 Flip？"),
]

EXP4_BASE_MODELS = ["LightGBM", "XGBoost", "RF", "LR", "MLP", "TabPFN"]
EXP4_DENOISERS = ["IQR", "LOF", "KNN", "IForest", "ECOD", "COPOD", "OCSVM", "AutoEncoder"]
EXP4_FLIP_RATES = [0.0, 0.05, 0.10, 0.20]
EXP4_STRATEGIES = ["Trim", "Flip"]

PPT_DEMO_PRESETS = {
    "数据集浏览": {"page": "数据集浏览", "dataset": "cardio", "modality_filter": ["tabular"]},
    "基准对比-表格": {"page": "基准对比", "modality": "tabular", "dataset": "cardio", "metric": "auc_pr"},
    "污染率-20%": {"page": "污染率分析", "modality": "tabular", "highlight_rate": 20},
    "跨模态-雷达图": {"page": "跨模态对比", "algorithms": ["IForest", "XGBoost", "ECOD", "LightGBM", "KNN"]},
    "精度效率": {"page": "精度-效率", "modality": "tabular"},
}


def render_sidebar_metric_guide(selected: str) -> None:
    """在侧边栏展示三指标说明。"""
    st.sidebar.markdown("**评估指标说明**")
    for key in ["auc_pr", "auc_roc", "f1_best"]:
        label = METRIC_LABELS[key]
        desc = METRIC_HINTS[key]
        badge = " · **主推**" if key == METRIC_RECOMMENDED else ""
        prefix = "▸ " if key == selected else "　"
        weight = "**" if key == selected else ""
        st.sidebar.markdown(f"{prefix}{weight}{label}{badge}{weight}：{desc}")
    if selected == METRIC_RECOMMENDED:
        st.sidebar.success("当前选用 AUC-PR：更适合本项目极度不平衡的异常检测场景。")
    else:
        st.sidebar.info(
            "提示：本项目异常率普遍较低，建议与 AUC-PR 对照查看，避免单一 AUC-ROC 得出乐观结论。"
            if selected == "auc_roc"
            else "提示：F1@best 依赖阈值搜索，可与 AUC 指标交叉印证。"
        )


def get_project_root() -> Path:
    return ROOT


def default_results_dir() -> Path:
    env = os.getenv("AD_RESULTS_DIR")
    return Path(env).expanduser() if env else ROOT / "results"


def default_figures_dir() -> Path:
    env = os.getenv("AD_FIGURES_DIR")
    return Path(env).expanduser() if env else ROOT / "figures"


def default_data_root() -> Path:
    env = os.getenv("AD_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def algo_category(name: str) -> str:
    for cat, names in ALGO_CATEGORIES.items():
        if name in names:
            return cat
    return "其他"


@st.cache_data(show_spinner=False)
def load_eda_summary(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return pd.DataFrame(payload)
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_experiment_results(results_dir: str, exp_name: str) -> pd.DataFrame:
    df = read_experiment(results_dir, exp_name)
    if df.empty:
        return df
    return successful_runs(df)


def _find_local_artifact_by_run_key(row: pd.Series, results_dir: Path) -> Path | None:
    """按 dataset + algorithm（+ seed）在 results/artifacts/ 下匹配本地 npz。"""
    exp = str(row.get("experiment_name", "exp1") or "exp1")
    ds = str(row.get("dataset_name", "") or "").replace("/", "-")
    algo = str(row.get("algorithm_name", "") or "")
    if not ds or not algo:
        return None
    art_dir = results_dir / "artifacts" / exp
    if not art_dir.is_dir():
        return None
    matches = sorted(art_dir.glob(f"exp1_{ds}_{algo}_*.npz"))
    if not matches:
        matches = sorted(
            p for p in art_dir.glob("*.npz")
            if p.name.startswith(f"exp1_{ds}_") and f"_{algo}_" in p.name
        )
    if not matches:
        return None
    want_seed = row.get("seed")
    if want_seed is not None and pd.notna(want_seed):
        seed = int(want_seed)
        for path in matches:
            meta_path = path.with_suffix(".json")
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("seed") == seed:
                    return path
            except (json.JSONDecodeError, OSError):
                continue
    return matches[-1]


def resolve_artifact_path(
    raw_path: str,
    results_dir: Path,
    *,
    row: pd.Series | None = None,
) -> Path | None:
    if raw_path and isinstance(raw_path, str):
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
        normalized = raw_path.replace("\\", "/")
        marker = "/artifacts/"
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            local = results_dir / "artifacts" / suffix
            if local.exists():
                return local
        name = candidate.name
        if name:
            matches = list(results_dir.rglob(name))
            if len(matches) == 1:
                return matches[0]
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)
    if row is not None:
        return _find_local_artifact_by_run_key(row, results_dir)
    return None


def artifact_from_row(row: pd.Series, results_dir: Path) -> dict[str, np.ndarray] | None:
    raw = str(row.get("artifact_path", "") or "")
    resolved = resolve_artifact_path(raw, results_dir, row=row)
    if resolved is None:
        return None
    patched = row.copy()
    patched["artifact_path"] = str(resolved)
    return load_artifact_npz(patched)


def count_roc_artifacts(rows: pd.DataFrame, results_dir: Path) -> int:
    """当前数据集在本地可用的 score artifact 数量（按算法去重）。"""
    if rows.empty:
        return 0
    n = 0
    for _, row in rows.drop_duplicates("algorithm_name").iterrows():
        artifact = artifact_from_row(row, results_dir)
        if artifact and "y_true" in artifact and "scores" in artifact:
            n += 1
    return n


def show_roc_fallback_figure(figures_dir: Path, dataset: str) -> bool:
    """无本地 artifact 时展示预生成 ROC 或全局热力图。"""
    token = dataset.replace("/", "-")
    if show_prebuilt_figure(figures_dir, f"exp1_roc*{token}*"):
        return True
    return show_prebuilt_figure(figures_dir, "exp1_*heatmap*")


def dedupe_results(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return latest_per_run_key(df, keys)


def list_available_seeds(df: pd.DataFrame) -> list[int]:
    if df.empty or "seed" not in df.columns:
        return DEFAULT_SEEDS.copy()
    seeds = sorted(int(s) for s in df["seed"].dropna().unique())
    return seeds or DEFAULT_SEEDS.copy()


def filter_by_seeds(df: pd.DataFrame, seeds: Iterable[int] | None) -> pd.DataFrame:
    if df.empty or not seeds or "seed" not in df.columns:
        return df.copy()
    seed_set = {int(s) for s in seeds}
    return df[df["seed"].isin(seed_set)].copy()


def aggregate_metric(
    df: pd.DataFrame,
    group_keys: list[str],
    metric: str,
) -> pd.DataFrame:
    """按 group_keys 聚合，输出 mean / std / n_runs。"""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    cols = [k for k in group_keys if k in df.columns]
    if not cols:
        return df.copy()
    agg = (
        df.groupby(cols, as_index=False)[metric]
        .agg(mean="mean", std="std", n_runs="count")
        .rename(columns={"mean": metric})
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def format_metric_value(value: float, std: float | None = None) -> str:
    if std is not None and std > 0:
        return f"{value:.4f} ± {std:.4f}"
    return f"{value:.4f}"


def eda_field_int(row: pd.Series, *keys: str) -> int:
    """从 EDA 行读取整数字段；图数据常用 n_nodes 而非 n_samples。"""
    for key in keys:
        if key not in row.index:
            continue
        val = row[key]
        if pd.notna(val):
            return int(val)
    return 0


def eda_field_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    if key not in row.index:
        return default
    val = row[key]
    if pd.isna(val):
        return default
    return float(val)


def modality_data_dir(modality: str, data_root: Path) -> Path:
    """返回 `data/` 根下某形态的原始文件目录。"""
    sub = MODALITY_DATA_SUBDIRS.get(modality, modality)
    return data_root / sub


def _count_raw_files_in_dir(path: Path, modality: str) -> int:
    if not path.exists():
        return 0
    skip = {".gitkeep", "selected_files.txt", "README.md"}
    n = 0
    for item in path.rglob("*"):
        if not item.is_file() or item.name in skip:
            continue
        if modality == "timeseries" and item.suffix.lower() != ".csv":
            continue
        if modality in {"tabular", "cv", "nlp"} and item.suffix.lower() != ".npz":
            continue
        if modality == "graph" and item.suffix.lower() in {".py", ".json", ".txt", ".md"}:
            continue
        n += 1
    return n


def summarize_raw_data_availability(data_root: Path) -> dict[str, int]:
    """统计 `data/{modality}/` 下已就位的原始文件数。"""
    counts = {
        mod: _count_raw_files_in_dir(data_root / sub, mod)
        for mod, sub in MODALITY_DATA_SUBDIRS.items()
    }
    npz_dir = data_root / "graph_npz"
    if npz_dir.exists():
        counts["graph"] += sum(1 for p in npz_dir.glob("*.npz") if p.stat().st_size > 100)
    return counts


def load_selected_manifest(data_root: Path) -> dict[str, list[str]]:
    path = data_root / "selected_files.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def timeseries_short_name(fname: str) -> str:
    for token in PLAN_TIMESERIES_SHORT:
        if token in fname:
            return token
    return Path(fname).stem


def summarize_modality_coverage(data_root: Path, eda: pd.DataFrame) -> dict[str, dict[str, int]]:
    """本地文件数 + EDA ok 数 + selected_files 计划数。"""
    local = summarize_raw_data_availability(data_root)
    manifest = load_selected_manifest(data_root)
    out: dict[str, dict[str, int]] = {}
    for mod in MODALITY_DATA_SUBDIRS:
        planned = len(manifest.get(mod, []))
        eda_ok = 0
        if not eda.empty and "modality" in eda.columns:
            st_ok = eda.get("status", "ok").fillna("ok").eq("ok")
            eda_ok = int((st_ok & eda["modality"].astype(str).eq(mod)).sum())
        out[mod] = {"local": local.get(mod, 0), "eda_ok": eda_ok, "planned": planned}
    return out


def modality_status_display(info: dict[str, int]) -> tuple[str, str | None]:
    """数据集浏览页顶部状态卡：仅显示文件数与是否已就位。"""
    local, eda_ok, planned = info["local"], info["eda_ok"], info["planned"]
    n = local if local > 0 else (eda_ok if eda_ok > 0 else planned)
    if n > 0:
        return f"{n} 个文件", "已就位"
    return "未就位", None


def build_eda_browser_frame(eda: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """浏览页数据集表：EDA ok 行 + 计划子集（时序等）补全。"""
    manifest = load_selected_manifest(data_root)
    if eda.empty:
        base = pd.DataFrame()
    else:
        base = eda[eda.get("status", "ok").fillna("ok").eq("ok")].copy()

    seen: set[tuple[str, str]] = set()
    if not base.empty:
        seen = {
            (str(r["name"]), str(r["modality"]))
            for _, r in base.iterrows()
        }

    extras: list[dict[str, Any]] = []
    for csv_name in manifest.get("timeseries", []):
        short = timeseries_short_name(csv_name)
        key = (short, "timeseries")
        if key in seen:
            continue
        if not eda.empty:
            hit = eda[eda["name"].astype(str).eq(short)]
            if not hit.empty:
                extras.append(hit.iloc[0].to_dict())
                seen.add(key)
                continue
        extras.append({
            "name": short,
            "modality": "timeseries",
            "source": "TSB-AD-U",
            "status": "configured",
            "file": csv_name,
        })
        seen.add(key)

    if extras:
        base = pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
    return base


def dataset_data_dir_hint(modality: str, data_root: Path) -> str:
    """返回当前形态在 `data/` 下的目标目录。"""
    return f"`{modality_data_dir(modality, data_root)}`"


def dataset_load_user_message(modality: str, data_root: Path) -> str:
    """无原始数据时，数据集浏览页的友好提示（对应测试指南 D-06）。"""
    mod_label = MODALITY_LABELS.get(modality, modality)
    target = dataset_data_dir_hint(modality, data_root)
    graph_hint = (
        f"原始图文件请放入 {target}（文件名见 `data/selected_files.json`）。"
        "就位后可对**节点特征**做 PCA。"
    )
    if modality == "graph":
        return f"`{data_root}` 下尚未就位图数据，无法绘制节点特征 PCA。{graph_hint}"
    return (
        f"`{data_root}` 下尚未就位 {mod_label} 原始文件，无法绘制特征分布 / PCA。"
        f"请将 benchmark 文件放入 {target}（清单见 `data/selected_files.json`），"
        "或点击下方「下载 ADBench 演示文件」。"
        "侧边栏可修改数据根目录。"
    )


def eda_detail_label(row: pd.Series) -> str:
    """详情下拉标签；同名数据集附加形态以消歧。"""
    name = str(row["name"])
    mod = str(row.get("modality", ""))
    return f"{name} · {MODALITY_LABELS.get(mod, mod)}"


def eda_detail_options(filtered: pd.DataFrame) -> pd.DataFrame:
    """为详情 selectbox 返回带 `_label` 列的表（每行一条 EDA 记录）。"""
    part = filtered.reset_index(drop=True).copy()
    names = part["name"].astype(str)
    if names.duplicated(keep=False).any():
        part["_label"] = part.apply(eda_detail_label, axis=1)
    else:
        part["_label"] = names
    return part.sort_values("_label").reset_index(drop=True)


ADBENCH_DEMO_BASE = (
    "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets"
)


def fetch_adbench_demo_file(name: str, data_root: Path) -> tuple[bool, str]:
    """从 ADBench GitHub 拉取单个 .npz 到 `data/{modality}/`。"""
    from urllib.error import URLError
    from urllib.request import urlretrieve

    from adapters.adbench_adapter import ADBENCH_DATASETS, normalize_adbench_name

    key = normalize_adbench_name(name)
    if key not in ADBENCH_DATASETS:
        return False, f"{name} 不是 ADBench 数据集，请手动放入 {data_root}"
    spec = ADBENCH_DATASETS[key]
    dest_dir = modality_data_dir(spec["modality"], data_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec["file"]
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"已存在 {dest}"
    url = f"{ADBENCH_DEMO_BASE}/{spec['group']}/{spec['file']}"
    try:
        urlretrieve(url, dest)
    except (URLError, OSError) as exc:
        return False, f"下载失败：{exc}"
    if not dest.exists() or dest.stat().st_size < 100:
        return False, f"下载结果异常：{dest}"
    return True, f"已保存至 {dest}"


def subsample_for_pca(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int = 2500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """大样本时随机抽样以加速 PCA。"""
    arr = np.asarray(X, dtype=float)
    labels = np.asarray(y).astype(int).reshape(-1)
    if arr.shape[0] <= max_samples:
        return arr, labels, False
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], max_samples, replace=False)
    return arr[idx], labels[idx], True


def render_feature_pca_plots(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    context: str = "训练集",
    max_hist_features: int = 6,
    max_pca_samples: int = 2500,
    max_hist_dims: int = 512,
) -> None:
    """在 Streamlit 中渲染特征直方图 + PCA（高维仅 PCA，大样本自动抽样）。"""
    arr = np.asarray(X_train, dtype=float)
    labels = np.asarray(y_train).astype(int).reshape(-1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        st.info(f"当前数据形状 {arr.shape}，无法做 PCA 二维投影。")
        return

    if arr.shape[1] <= max_hist_dims:
        st.pyplot(plot_feature_distribution(arr, max_features=max_hist_features), clear_figure=True)
    else:
        st.caption(f"特征维度 {arr.shape[1]} 较高，跳过直方图，仅展示 PCA。")

    pca_x, pca_y, sampled = subsample_for_pca(arr, labels, max_samples=max_pca_samples)
    if sampled:
        st.caption(f"样本数 {arr.shape[0]:,}，PCA 随机抽样 {pca_x.shape[0]:,} 点（seed=42）。")

    title = f"PCA 2D projection ({context})"
    pca_fig = plot_pca_scatter(pca_x, pca_y, title=title)
    if pca_fig is None:
        st.info("有效样本不足，无法绘制 PCA。")
        return
    st.pyplot(pca_fig, clear_figure=True)


def contamination_rate_column(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if "contamination_rate" in df.columns and "label_flip_rate" in df.columns:
        return df["contamination_rate"].where(
            df["contamination_rate"].notna(),
            df["label_flip_rate"],
        )
    if "contamination_rate" in df.columns:
        return df["contamination_rate"]
    if "label_flip_rate" in df.columns:
        return df["label_flip_rate"]
    return pd.Series(np.nan, index=df.index)


def pollution_mode_label(row: pd.Series) -> str:
    mode = str(row.get("contamination_mode", "") or "")
    if "label_flip" in mode or mode.startswith("supervised"):
        return "有监督 · 标签翻转"
    if "unsupervised" in mode:
        return "无监督 · 保留异常比例"
    algo = str(row.get("algorithm_name", ""))
    return "有监督 · 标签翻转" if algo in SUPERVISED_ALGOS else "无监督 · 保留异常比例"


def list_datasets_from_eda(eda: pd.DataFrame) -> list[str]:
    if eda.empty or "name" not in eda.columns:
        return []
    ok = eda[eda.get("status", "ok").fillna("ok").eq("ok")]
    return sorted(ok["name"].astype(str).unique().tolist())


def list_datasets_from_exp(df: pd.DataFrame, modality: str | None = None) -> list[str]:
    if df.empty:
        return []
    part = df
    if modality:
        part = part[part["modality"].astype(str).eq(modality)]
    return sorted(part["dataset_name"].astype(str).unique().tolist())


def summarize_project_coverage(eda: pd.DataFrame, exp1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not eda.empty and "modality" in eda.columns:
        ok = eda[eda.get("status", "ok").fillna("ok").eq("ok")]
        for mod, grp in ok.groupby("modality"):
            mod_str = str(mod)
            rows.append({
                "mod_key": mod_str,
                "形态": MODALITY_LABELS.get(mod_str, mod_str),
                "EDA 数据集数": len(grp),
                "Exp-1 数据集数": 0,
                "Benchmark": ", ".join(sorted(grp["source"].dropna().astype(str).unique())) if "source" in grp.columns else "",
            })
    if not exp1.empty and rows:
        exp_mod = exp1.groupby("modality")["dataset_name"].nunique()
        for row in rows:
            key = row.pop("mod_key")
            if key in exp_mod.index:
                row["Exp-1 数据集数"] = int(exp_mod[key])
    out = pd.DataFrame(rows)
    if not out.empty:
        return out.drop(columns=[c for c in out.columns if c == "mod_key"], errors="ignore")
    return out


def check_experiment_files(results_dir: Path) -> dict[str, bool]:
    return {
        exp: (results_dir / f"{exp}_results.csv").exists()
        and (results_dir / f"{exp}_results.csv").stat().st_size > 0
        for exp in ("exp1", "exp2", "exp3", "exp4")
    }


def find_figures(figures_dir: Path, pattern: str) -> list[Path]:
    if not figures_dir.exists():
        return []
    matches = sorted(figures_dir.rglob(pattern))
    png_first = sorted(matches, key=lambda p: (p.suffix.lower() != ".png", p.name))
    return png_first


def list_algorithms_from_exp(
    df: pd.DataFrame,
    *,
    modality: str | None = None,
    dataset: str | None = None,
) -> list[str]:
    if df.empty:
        return []
    part = df
    if modality:
        part = part[part["modality"].astype(str).eq(modality)]
    if dataset:
        part = part[part["dataset_name"].astype(str).eq(dataset)]
    return sorted(part["algorithm_name"].astype(str).unique().tolist())


def show_prebuilt_figure_path(path: Path, *, caption: str | None = None) -> None:
    cap = caption or path.name
    if path.suffix.lower() == ".svg":
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        st.markdown(
            f'<img src="data:image/svg+xml;base64,{b64}" alt="{cap}" style="width:100%;">',
            unsafe_allow_html=True,
        )
        if cap:
            st.caption(cap)
    elif path.suffix.lower() == ".csv":
        st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
        if cap:
            st.caption(cap)
    else:
        st.image(str(path), caption=cap, use_container_width=True)


def show_figure_gallery(
    figures_dir: Path,
    catalog: list[tuple[str, str]],
    *,
    key: str,
    default_label: str | None = None,
) -> bool:
    """按目录扫描预生成图，提供下拉选择。catalog: [(显示名, glob), ...]"""
    options: list[str] = []
    mapping: dict[str, Path] = {}
    for label, pattern in catalog:
        for path in find_figures(figures_dir, pattern):
            display = f"{label} — {path.relative_to(figures_dir)}"
            options.append(display)
            mapping[display] = path
    if not options:
        st.caption("暂无可用图表。")
        return False
    default_idx = 0
    if default_label:
        for i, opt in enumerate(options):
            if default_label in opt:
                default_idx = i
                break
    choice = st.selectbox("选择图表", options, index=default_idx, key=key)
    show_prebuilt_figure_path(mapping[choice])
    return True


def render_advanced_figure_gallery(
    figures_dir: Path,
    catalog: list[tuple[str, str]],
    *,
    key: str,
    default_label: str | None = None,
) -> None:
    """有 CSV 时：将 figures/ 图折叠到「高级」区，主界面只保留交互图。"""
    with st.expander("高级", expanded=False):
        show_figure_gallery(figures_dir, catalog, key=key, default_label=default_label)


def render_fallback_figures(
    figures_dir: Path,
    catalog: list[tuple[str, str]],
    *,
    key: str,
    warning: str,
) -> None:
    """无 CSV 时：用静态图作为兜底展示。"""
    st.warning(warning)
    show_figure_gallery(figures_dir, catalog, key=key)


def show_prebuilt_figure(figures_dir: Path, pattern: str, *, caption: str | None = None) -> bool:
    matches = find_figures(figures_dir, pattern)
    if not matches:
        return False
    show_prebuilt_figure_path(matches[0], caption=caption or matches[0].name)
    return True


def plot_feature_distribution(X: np.ndarray, max_features: int = 6) -> plt.Figure:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_features = min(arr.shape[1], max_features)
    cols = 3
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < n_features:
            values = arr[:, idx]
            finite = values[np.isfinite(values)]
            ax.hist(finite, bins=30, color="#4c78a8", alpha=0.85)
            ax.set_title(f"特征 {idx + 1}", fontsize=10)
            ax.grid(alpha=0.2)
        else:
            ax.axis("off")
    fig.suptitle("训练集特征分布（前若干维）", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "PCA 2D projection (train)",
) -> plt.Figure | None:
    arr = np.asarray(X, dtype=float)
    labels = np.asarray(y).astype(int).reshape(-1)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    finite_mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite_mask]
    labels = labels[finite_mask]
    if len(arr) < 3:
        return None
    n_components = min(2, arr.shape[1], arr.shape[0])
    coords = PCA(n_components=n_components, random_state=42).fit_transform(arr)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, name, color in [(0, "normal", "#4c78a8"), (1, "anomaly", "#e45756")]:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1 if n_components > 1 else 0],
            s=12, alpha=0.55, label=name, c=color,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2" if n_components > 1 else "PC1")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_metric_bars(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    *,
    show_std: bool = False,
) -> plt.Figure:
    primary = metrics[0]
    plot_df = df.sort_values(primary, ascending=True).copy()
    y_pos = np.arange(len(plot_df))
    width = 0.35 if len(metrics) > 1 else 0.6
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df) + 1.5)))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        xerr = None
        if show_std and f"{metric}_std" in plot_df.columns:
            xerr = plot_df[f"{metric}_std"].to_numpy()
        ax.barh(
            y_pos + offset,
            plot_df[metric],
            xerr=xerr,
            height=width,
            label=METRIC_LABELS.get(metric, metric),
            color=colors[idx % len(colors)],
            alpha=0.9,
            capsize=3 if xerr is not None else 0,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["algorithm_name"])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("分数")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_roc_overlay(
    rows: pd.DataFrame,
    results_dir: Path,
    title: str,
    *,
    top_k: int = 8,
    sort_metric: str = "auc_roc",
    algorithms: list[str] | None = None,
) -> plt.Figure | None:
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = 0
    if algorithms:
        subset = rows[rows["algorithm_name"].isin(algorithms)].drop_duplicates("algorithm_name")
    else:
        metric_col = sort_metric if sort_metric in rows.columns else "auc_roc"
        subset = rows.sort_values(metric_col, ascending=False).head(top_k)
    for _, row in subset.iterrows():
        artifact = artifact_from_row(row, results_dir)
        if not artifact or "y_true" not in artifact or "scores" not in artifact:
            continue
        y_true = artifact["y_true"].astype(int).ravel()
        scores = artifact["scores"].astype(float).ravel()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr),
            estimator_name=str(row["algorithm_name"]),
        ).plot(ax=ax)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_contamination_curves(
    df: pd.DataFrame,
    *,
    metric: str = "auc_roc",
    pollution_filter: str | None = None,
    algorithms: list[str] | None = None,
) -> plt.Figure:
    part = df.copy()
    part["rate"] = contamination_rate_column(part)
    if pollution_filter == "unsupervised":
        part = part[part.apply(lambda r: "无监督" in pollution_mode_label(r), axis=1)]
    elif pollution_filter == "supervised":
        part = part[part.apply(lambda r: "有监督" in pollution_mode_label(r), axis=1)]
    if algorithms:
        part = part[part["algorithm_name"].isin(algorithms)]
    curve = (
        part.groupby(["algorithm_name", "rate"], as_index=False)[metric]
        .mean()
        .dropna(subset=["rate", metric])
        .sort_values(["algorithm_name", "rate"])
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    for idx, algo in enumerate(sorted(curve["algorithm_name"].unique())):
        g = curve[curve["algorithm_name"] == algo]
        is_sup = algo in SUPERVISED_ALGOS
        ax.plot(
            g["rate"] * 100,
            g[metric],
            marker="D" if is_sup else "o",
            linestyle="-" if is_sup else "--",
            linewidth=1.8,
            label=f"{algo} ({'监督' if is_sup else '无监督'})",
            color=cmap(idx % 20),
        )
    mode_note = {"unsupervised": "无监督污染", "supervised": "有监督标签翻转"}.get(pollution_filter or "", "全部")
    ax.set_title(f"Exp-2 污染退化曲线 · {mode_note}")
    ax.set_xlabel("污染率 / 标签翻转率 (%)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xticks([r * 100 for r in CONTAMINATION_RATES])
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def plot_efficiency_tradeoff(
    df: pd.DataFrame,
    metric: str = "auc_roc",
    *,
    algorithms: list[str] | None = None,
    algo_type: str = "全部",
    top_n: int | None = None,
) -> plt.Figure:
    part = df.copy()
    if algorithms:
        part = part[part["algorithm_name"].isin(algorithms)]
    if algo_type == "监督":
        part = part[part["algorithm_name"].isin(SUPERVISED_ALGOS)]
    elif algo_type == "无监督":
        part = part[~part["algorithm_name"].isin(SUPERVISED_ALGOS)]
    part["total_time_sec"] = part["fit_time_sec"].fillna(0) + part["predict_time_sec"].fillna(0)
    part["param_proxy"] = part.apply(parameter_proxy, axis=1)
    summary = (
        part.groupby("algorithm_name", as_index=False)
        .agg({metric: "mean", "total_time_sec": "median", "param_proxy": "median"})
        .sort_values(metric, ascending=False)
    )
    summary["total_time_sec"] = summary["total_time_sec"].clip(lower=0.01)
    annotate = summary.head(top_n) if top_n else summary
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sizes = 35 + 18 * np.sqrt(summary["param_proxy"].clip(lower=1).to_numpy())
    colors = ["#f58518" if a in SUPERVISED_ALGOS else "#4c78a8" for a in summary["algorithm_name"]]
    ax.scatter(
        summary["total_time_sec"], summary[metric],
        s=sizes, alpha=0.72, c=colors, edgecolor="black", linewidth=0.4,
    )
    for _, row in annotate.iterrows():
        ax.annotate(row["algorithm_name"], (row["total_time_sec"], row[metric]), fontsize=7, alpha=0.85)
    ax.set_xscale("log")
    ax.set_title(f"精度-效率权衡（{METRIC_LABELS.get(metric, metric)} vs 耗时）")
    ax.set_xlabel("训练+推理耗时中位数 (秒, log)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_modality_heatmap(mod_df: pd.DataFrame, metric: str = "auc_roc") -> plt.Figure:
    order = {"tabular": 1, "cv": 2, "nlp": 3, "timeseries": 4, "graph": 5}
    cols = sorted(mod_df.columns, key=lambda x: order.get(x, 99))
    matrix = mod_df[cols]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cols)), max(4, 0.35 * len(matrix) + 2)))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels([MODALITY_LABELS.get(c, c) for c in cols])
    ax.set_yticklabels(matrix.index)
    ax.set_title(f"Exp-3 通用算法 × 数据形态 ({METRIC_LABELS.get(metric, metric)})")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def plot_radar(mod_df: pd.DataFrame, algorithms: list[str], metric: str = "auc_roc") -> plt.Figure | None:
    radar = mod_df.loc[[a for a in algorithms if a in mod_df.index]].dropna(how="all")
    modalities = list(radar.columns)
    if len(modalities) < 3 or radar.empty:
        return None
    angles = np.linspace(0, 2 * np.pi, len(modalities), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    cmap = plt.get_cmap("Set1")
    for idx, algo in enumerate(radar.index):
        values = radar.loc[algo].fillna(0.4).to_numpy(dtype=float).tolist()
        values += values[:1]
        color = cmap(idx)
        ax.plot(angles, values, linewidth=2.2, marker="o", label=algo, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in modalities], fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"跨模态雷达图 ({METRIC_LABELS.get(metric, metric)})", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=9)
    return fig


def parse_exp4_algo(name: str) -> dict[str, str]:
    """解析 Exp-4 算法名：Standard_XGBoost / Defended_LightGBM_IQR_Trim。"""
    if name.startswith("Standard_"):
        return {"kind": "standard", "base": name[len("Standard_"):], "denoiser": "", "strategy": ""}
    if name.startswith("Defended_"):
        body = name[len("Defended_"):]
        strategy = ""
        if body.endswith("_Trim"):
            strategy, body = "Trim", body[:-5]
        elif body.endswith("_Flip"):
            strategy, body = "Flip", body[:-5]
        for base in EXP4_BASE_MODELS:
            prefix = f"{base}_"
            if body.startswith(prefix):
                return {
                    "kind": "defended",
                    "base": base,
                    "denoiser": body[len(prefix):],
                    "strategy": strategy,
                }
        return {"kind": "defended", "base": body, "denoiser": "", "strategy": strategy}
    return {"kind": "other", "base": name, "denoiser": "", "strategy": ""}


def exp4_label_flip_column(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if "label_flip_rate" in df.columns:
        return pd.to_numeric(df["label_flip_rate"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def filter_exp4_view(
    df: pd.DataFrame,
    *,
    dataset: str | None = None,
    base_model: str | None = None,
    denoiser: str | None = None,
    strategy: str | None = None,
) -> pd.DataFrame:
    part = df.copy()
    if dataset:
        part = part[part["dataset_name"].astype(str).eq(dataset)]
    if not base_model and not denoiser and not strategy:
        return part

    keep_rows = []
    for _, row in part.iterrows():
        parsed = parse_exp4_algo(str(row["algorithm_name"]))
        if base_model and parsed["base"] != base_model:
            continue
        if parsed["kind"] == "standard":
            if denoiser or strategy:
                continue
        else:
            if denoiser and parsed["denoiser"] != denoiser:
                continue
            if strategy and parsed["strategy"] != strategy:
                continue
        keep_rows.append(row)
    return pd.DataFrame(keep_rows) if keep_rows else part.iloc[0:0]


def plot_exp4_defense_curves(
    df: pd.DataFrame,
    *,
    base_model: str,
    denoisers: list[str] | None = None,
    strategies: list[str] | None = None,
    metric: str = "auc_roc",
) -> plt.Figure:
    part = df.copy()
    part["flip_rate"] = exp4_label_flip_column(part)
    curve = (
        part.groupby(["algorithm_name", "flip_rate"], as_index=False)[metric]
        .mean()
        .dropna(subset=["flip_rate", metric])
        .sort_values(["algorithm_name", "flip_rate"])
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    std_name = f"Standard_{base_model}"
    std_rows = curve[curve["algorithm_name"] == std_name]
    if not std_rows.empty:
        ax.plot(
            std_rows["flip_rate"] * 100,
            std_rows[metric],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            color="#d62728",
            label=f"Standard {base_model}（无防御）",
        )

    denoiser_list = denoisers or EXP4_DENOISERS
    strategy_list = strategies or EXP4_STRATEGIES
    cmap = plt.get_cmap("tab10")
    for idx, denoiser in enumerate(denoiser_list):
        for strategy in strategy_list:
            ls = "-" if strategy == "Trim" else "--"
            algo = f"Defended_{base_model}_{denoiser}_{strategy}"
            g = curve[curve["algorithm_name"] == algo]
            if g.empty:
                continue
            color = cmap(idx % 10)
            ax.plot(
                g["flip_rate"] * 100,
                g[metric],
                marker="D" if strategy == "Trim" else "s",
                linestyle=ls,
                linewidth=1.8,
                color=color,
                label=f"{denoiser} · {strategy}",
            )

    ax.set_title(f"Exp-4 鲁棒防御 · {base_model} · {METRIC_LABELS.get(metric, metric)}")
    ax.set_xlabel("标签翻转率 (%)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xticks([int(r * 100) for r in EXP4_FLIP_RATES])
    ax.margins(y=0.08)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def plot_exp4_ablation_bars(
    df: pd.DataFrame,
    *,
    base_model: str,
    flip_rate: float = 0.20,
    metric: str = "auc_roc",
    top_n: int = 12,
    denoisers: list[str] | None = None,
    strategies: list[str] | None = None,
    include_standard: bool = True,
) -> plt.Figure:
    part = df.copy()
    part["flip_rate"] = exp4_label_flip_column(part)
    snap = part[np.isclose(part["flip_rate"], flip_rate)]
    snap = filter_exp4_view(snap, base_model=base_model)
    if not include_standard:
        snap = snap[~snap["algorithm_name"].astype(str).str.startswith("Standard_")]
    if denoisers or strategies:
        filtered_rows = []
        for _, row in snap.iterrows():
            parsed = parse_exp4_algo(str(row["algorithm_name"]))
            if parsed["kind"] == "standard":
                if include_standard:
                    filtered_rows.append(row)
                continue
            if denoisers and parsed["denoiser"] not in denoisers:
                continue
            if strategies and parsed["strategy"] not in strategies:
                continue
            filtered_rows.append(row)
        snap = pd.DataFrame(filtered_rows) if filtered_rows else snap.iloc[0:0]
    if snap.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
        ax.axis("off")
        return fig

    ranking = (
        snap.groupby("algorithm_name", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=True)
        .tail(top_n)
    )
    colors = [
        "#d62728" if n.startswith("Standard_") else ("#2ca02c" if "Trim" in n else "#8c564b")
        for n in ranking["algorithm_name"]
    ]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(ranking) + 1.5)))
    ax.barh(ranking["algorithm_name"], ranking[metric], color=colors, alpha=0.9)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Exp-4 消融 · {base_model} · 翻转率 {int(flip_rate * 100)}%")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def try_load_dataset(name: str, modality: str, data_root: str) -> dict[str, Any] | None:
    try:
        from adapters import load_dataset

        bundle = load_dataset(name, modality=modality, data_root=data_root)
        X_train, _, y_train, _ = bundle.as_tuple()
        return {
            "X_train": np.asarray(X_train),
            "y_train": np.asarray(y_train),
            "metadata": bundle.metadata,
        }
    except Exception as exc:
        return {"error": str(exc)}
