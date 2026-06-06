"""Generate analysis figures from saved experiment CSVs and artifacts.

This module is intentionally read-only with respect to experiment execution: it
never trains models and only consumes ``results/*.csv`` plus artifact ``.npz``
files produced by the experiment scripts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, auc, roc_curve

from eval.analysis_utils import (
    ensure_dir,
    latest_per_run_key,
    load_artifact_npz,
    parameter_proxy,
    read_experiment,
    successful_runs,
    write_table_csv,
)


ROOT = Path(__file__).resolve().parent.parent


def _save(fig, path: str | Path, *, also_svg: bool = True) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if also_svg:
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def _empty_notice(path: str | Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=10)
    return _save(fig, path, also_svg=False)


def _heatmap(
    matrix: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    dataset_to_modality: dict[str, str] | None = None,
) -> Path:
    if matrix.empty:
        return _empty_notice(path, title, "No valid rows available.")
    
    # ------------------------------------------------------------------------
    # [Prettify Heatmap] Sort rows (algorithms) and columns (datasets)
    # ------------------------------------------------------------------------
    ALGO_GROUPS = {
        "Statistical/Proximity": ["IQR", "LOF", "KNN"],
        "Trees/Ensembles": ["IForest", "ECOD", "COPOD", "RF", "XGBoost", "LightGBM"],
        "Linear/Boundary": ["LR", "OCSVM"],
        "Deep/Neural": ["MLP", "AutoEncoder", "DeepSVDD", "TabPFN"],
        "TimeSeries": ["LSTM-AE", "LSTM-Sup", "MiniRocket", "MatrixProfile", "DADA"],
        "Graph": ["DOMINANT", "CoLA", "GCN", "BWGNN", "XGBGraph", "UNPrompt"]
    }
    
    # Create an ordered list of algorithms for the Y axis
    ordered_algos = []
    for group, algos in ALGO_GROUPS.items():
        ordered_algos.extend([a for a in algos if a in matrix.index])
    # Add any remaining ones that were not explicitly listed
    for a in matrix.index:
        if a not in ordered_algos:
            ordered_algos.append(a)
    
    # Create an ordered list of datasets for the X axis based on modality
    ordered_datasets = list(matrix.columns)
    if dataset_to_modality is not None:
        MODALITY_ORDER = {
            "tabular": 1,
            "cv": 2,
            "nlp": 3,
            "timeseries": 4,
            "graph": 5
        }
        # Sort columns primarily by modality order, then alphabetically by dataset name
        ordered_datasets.sort(key=lambda d: (MODALITY_ORDER.get(dataset_to_modality.get(d, ""), 99), d))
    
    # Reindex the matrix with the beautiful ordering
    matrix = matrix.loc[ordered_algos, ordered_datasets]
    # ------------------------------------------------------------------------

    h = max(4.0, 0.35 * len(matrix.index) + 2.0)
    w = max(7.0, 0.38 * len(matrix.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(w, h))
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Dataset / modality")
    ax.set_ylabel("Algorithm")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return _save(fig, path)


def _plot_table(df: pd.DataFrame, path: str | Path, title: str, max_rows: int = 20) -> Path:
    if df.empty:
        return _empty_notice(path, title, "No valid rows available.",)
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    fig_h = max(3, 0.35 * len(shown) + 1.5)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(shown.columns)), fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    return _save(fig, path, also_svg=False)


def plot_exp1(results_dir: str | Path, figures_dir: str | Path, metric: str = "auc_roc") -> list[Path]:
    out_dir = ensure_dir(Path(figures_dir) / "exp1")
    df = successful_runs(read_experiment(results_dir, "exp1"), metric=metric)
    df = latest_per_run_key(df, ["dataset_name", "algorithm_name", "modality"])
    outputs: list[Path] = []
    if df.empty:
        outputs.append(_empty_notice(out_dir / "exp1_no_results.png", "Exp-1", "No successful Exp-1 rows."))
        return outputs

    # Handle datasets with the same name but different modalities (e.g., 'amazon' in NLP and Graph)
    modality_counts = df.groupby("dataset_name")["modality"].nunique()
    conflict_datasets = modality_counts[modality_counts > 1].index
    df["display_name"] = df.apply(
        lambda row: f"{row['dataset_name']} ({row['modality']})" if row['dataset_name'] in conflict_datasets else row['dataset_name'],
        axis=1
    )

    pivot = df.pivot_table(index="algorithm_name", columns="display_name", values=metric, aggfunc="mean")
    
    # Create a mapping of dataset to modality
    dataset_to_modality = dict(zip(df["display_name"], df["modality"]))
    
    outputs.append(_heatmap(pivot, out_dir / f"exp1_{metric}_heatmap.png",
                            title=f"Exp-1 algorithm x dataset ({metric})",
                            cbar_label=metric,
                            dataset_to_modality=dataset_to_modality))

    ranks = df.pivot_table(index="display_name", columns="algorithm_name", values=metric, aggfunc="mean")
    rank_by_dataset = ranks.rank(axis=1, ascending=False, method="average")
    mean_rank = rank_by_dataset.mean(axis=0).sort_values()
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(mean_rank)), 5))
    ax.bar(mean_rank.index, mean_rank.values, color="#4c78a8")
    ax.set_title("Exp-1 average rank across datasets")
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_xlabel("Algorithm")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    outputs.append(_save(fig, out_dir / "exp1_average_rank.png"))

    eff = df.copy()
    eff["total_time_sec"] = eff["fit_time_sec"].fillna(0) + eff["predict_time_sec"].fillna(0)
    eff["param_proxy"] = eff.apply(parameter_proxy, axis=1)
    eff_summary = (
        eff.groupby("algorithm_name", as_index=False)
        .agg({metric: "mean", "total_time_sec": "median", "param_proxy": "median"})
        .sort_values(metric, ascending=False)
    )
    eff_summary["total_time_sec"] = eff_summary["total_time_sec"].clip(lower=0.01)
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sizes = 35 + 18 * np.sqrt(eff_summary["param_proxy"].clip(lower=1).to_numpy())
    
    SUPERVISED_ALGOS = {"XGBoost", "LightGBM", "RF", "LR", "MLP", "TabPFN", "XGBGraph", "BWGNN", "GCN", "LSTM-Sup"}
    colors = ["#f58518" if algo in SUPERVISED_ALGOS else "#4c78a8" for algo in eff_summary["algorithm_name"]]
    
    ax.scatter(eff_summary["total_time_sec"], eff_summary[metric], s=sizes, alpha=0.72, c=colors, edgecolor="black", linewidth=0.4)
    
    import matplotlib.lines as mlines
    unsup_handle = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='#4c78a8', markersize=8, markeredgecolor='black', markeredgewidth=0.4, label='Unsupervised')
    sup_handle = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='#f58518', markersize=8, markeredgecolor='black', markeredgewidth=0.4, label='Supervised')
    ax.legend(handles=[unsup_handle, sup_handle], loc="lower right", fontsize=8, frameon=True)
    
    texts = []
    for _, row in eff_summary.iterrows():
        texts.append(ax.text(row["total_time_sec"], row[metric], row["algorithm_name"], fontsize=8))
        
    ax.set_xscale("log")
    
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    except ImportError:
        pass
        
    ax.set_title("Exp-1 performance-efficiency tradeoff")
    ax.set_xlabel("Median fit+predict time (sec, symlog)")
    ax.set_ylabel(f"Mean {metric}")
    ax.grid(alpha=0.25)
    outputs.append(_save(fig, out_dir / "exp1_efficiency_tradeoff.png"))

    # ROC overlays for up to three representative datasets with available artifacts.
    available = df[df["artifact_path"].fillna("").astype(str).ne("")]
    dataset_order = available.groupby("dataset_name")[metric].mean().sort_values(ascending=False).index[:3]
    for ds_name in dataset_order:
        subset = available[available["dataset_name"].eq(ds_name)].sort_values(metric, ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        plotted = 0
        for _, row in subset.iterrows():
            artifact = load_artifact_npz(row)
            if not artifact or "y_true" not in artifact or "scores" not in artifact:
                continue
            y_true = artifact["y_true"].astype(int).ravel()
            scores = artifact["scores"].astype(float).ravel()
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, scores)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr), estimator_name=row["algorithm_name"]).plot(ax=ax)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue
        ax.set_title(f"ROC overlay: {ds_name}")
        ax.grid(alpha=0.25)
        outputs.append(_save(fig, out_dir / f"exp1_roc_{str(ds_name).replace('/', '-')}.png"))

    return outputs


def plot_exp2(results_dir: str | Path, figures_dir: str | Path, metric: str = "auc_roc") -> list[Path]:
    out_dir = ensure_dir(Path(figures_dir) / "exp2")
    df = successful_runs(read_experiment(results_dir, "exp2"), metric=metric)
    df = latest_per_run_key(df, ["modality", "dataset_name", "algorithm_name", "contamination_mode", "contamination_rate", "label_flip_rate"])
    outputs: list[Path] = []
    if df.empty:
        outputs.append(_empty_notice(out_dir / "exp2_no_results.png", "Exp-2", "No successful Exp-2 rows."))
        return outputs

    df = df.copy()
    df["rate"] = df["contamination_rate"].where(df["contamination_rate"].notna(), df["label_flip_rate"])
    curve = (
        df.groupby(["modality", "algorithm_name", "rate"], as_index=False)[metric]
        .mean()
        .dropna(subset=["rate", metric])
    )
    SUPERVISED_ALGOS = {"XGBoost", "LightGBM", "RF", "LR", "MLP", "TabPFN", "XGBGraph", "BWGNN", "GCN", "LSTM-Sup"}

    for modality, part in curve.groupby("modality"):
        # Make width slightly larger to fit external legend
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        algos = sorted(part["algorithm_name"].unique())
        cmap = plt.get_cmap("tab20")
        
        for idx, algo in enumerate(algos):
            g = part[part["algorithm_name"] == algo].sort_values("rate")
            is_sup = algo in SUPERVISED_ALGOS
            
            ls = "-" if is_sup else "--"
            marker = "D" if is_sup else "o"
            label = f"{algo} (Sup)" if is_sup else f"{algo} (Unsup)"
            
            ax.plot(g["rate"] * 100, g[metric], marker=marker, linestyle=ls, linewidth=1.8, markersize=5, label=label, color=cmap(idx % 20))

        ax.set_title(f"Exp-2 contamination degradation ({modality})")
        ax.set_xlabel("Training contamination / label flip rate (%)")
        ax.set_ylabel(metric)
        
        # By NOT setting ax.set_ylim(0, 1.03), Matplotlib will dynamically auto-scale the Y-axis.
        # This acts as a magnifying glass (放大纵轴截距) to clearly show gaps between closely performing models.
        ax.margins(y=0.08)
        ax.grid(alpha=0.25)
        
        # Place legend completely outside the plot box to prevent covering data
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, borderaxespad=0.)
        # Use tight_layout to ensure the external legend is saved
        fig.tight_layout()
        
        outputs.append(_save(fig, out_dir / f"exp2_degradation_{modality}.png"))

    baseline = curve.loc[curve.groupby(["modality", "algorithm_name"])["rate"].idxmin()]
    worst = curve.loc[curve.groupby(["modality", "algorithm_name"])["rate"].idxmax()]
    ranking = baseline.merge(
        worst,
        on=["modality", "algorithm_name"],
        suffixes=("_baseline", "_max_rate"),
    )
    ranking["auc_drop"] = ranking[f"{metric}_baseline"] - ranking[f"{metric}_max_rate"]
    ranking = ranking.sort_values(["modality", "auc_drop"], ascending=[True, True])
    table = ranking[["modality", "algorithm_name", "rate_baseline", f"{metric}_baseline", "rate_max_rate", f"{metric}_max_rate", "auc_drop"]]
    outputs.append(write_table_csv(table, out_dir / "exp2_robustness_ranking.csv"))
    outputs.append(_plot_table(table, out_dir / "exp2_robustness_ranking.png", "Exp-2 robustness ranking (smaller drop is better)"))

    cross = ranking.pivot_table(index="algorithm_name", columns="modality", values="auc_drop", aggfunc="mean")
    
    # Calculate the max absolute value to center the colormap at 0
    max_abs_drop = np.nanmax(np.abs(cross.values))
    
    outputs.append(_heatmap(cross, out_dir / "exp2_cross_modal_auc_drop_heatmap.png",
                            title="Exp-2 AUC drop by algorithm and modality",
                            cbar_label="AUC drop", cmap="coolwarm", vmin=-max_abs_drop, vmax=max_abs_drop))
    return outputs


def plot_exp3(results_dir: str | Path, figures_dir: str | Path, metric: str = "auc_roc") -> list[Path]:
    out_dir = ensure_dir(Path(figures_dir) / "exp3")
    df = successful_runs(read_experiment(results_dir, "exp3"), metric=metric)
    df = latest_per_run_key(df, ["modality", "dataset_name", "algorithm_name"])
    outputs: list[Path] = []
    if df.empty:
        outputs.append(_empty_notice(out_dir / "exp3_no_results.png", "Exp-3", "No successful Exp-3 rows."))
        return outputs

    mod = df.pivot_table(index="algorithm_name", columns="modality", values=metric, aggfunc="mean")
    mod = mod.loc[mod.mean(axis=1).sort_values(ascending=False).index]
    
    # Ensure a logical order for the modalities on the X-axis
    MODALITY_ORDER = {"tabular": 1, "cv": 2, "nlp": 3, "timeseries": 4, "graph": 5}
    ordered_cols = sorted(mod.columns, key=lambda x: MODALITY_ORDER.get(x, 99))
    mod = mod[ordered_cols]
    
    outputs.append(_heatmap(mod, out_dir / f"exp3_algorithm_modality_{metric}_heatmap.png",
                            title=f"Exp-3 algorithm x modality ({metric})",
                            cbar_label=metric, cmap="Blues"))

    # --- NEW: Cross-modal line chart (Parallel Coordinates style) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    SUPERVISED_ALGOS = {"XGBoost", "LightGBM", "RF", "LR", "MLP", "TabPFN", "XGBGraph", "BWGNN", "GCN", "LSTM-Sup"}
    cmap = plt.get_cmap("tab20")
    
    x_pos = np.arange(len(ordered_cols))
    for idx, algo in enumerate(mod.index):
        y_vals = mod.loc[algo].to_numpy()
        is_sup = algo in SUPERVISED_ALGOS
        
        ls = "-" if is_sup else "--"
        marker = "D" if is_sup else "o"
        label = f"{algo} (Sup)" if is_sup else f"{algo} (Unsup)"
        
        # Connect dots but break at NaNs automatically
        ax.plot(x_pos, y_vals, marker=marker, linestyle=ls, linewidth=1.8, markersize=5, label=label, color=cmap(idx % 20))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.capitalize() for m in ordered_cols], weight="bold")
    ax.set_title(f"Exp-3 Cross-modal performance profile ({metric})")
    ax.set_xlabel("Modality")
    ax.set_ylabel(metric)
    ax.margins(y=0.08)
    ax.grid(alpha=0.25)
    
    # Place legend completely outside the plot box
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, borderaxespad=0.)
    fig.tight_layout()
    outputs.append(_save(fig, out_dir / "exp3_cross_modal_lines.png"))
    # ----------------------------------------------------------------

    difficulty = mod.mean(axis=0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(difficulty.index, difficulty.values, color="#54a24b")
    ax.set_title("Exp-3 modality difficulty")
    ax.set_ylabel(f"Mean {metric} across algorithms")
    ax.set_xlabel("Modality")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", alpha=0.25)
    outputs.append(_save(fig, out_dir / "exp3_modality_difficulty.png"))

    radar = mod.dropna(how="all")
    modalities = list(radar.columns)
    if len(modalities) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(modalities), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        
        # Limit to top 5 algorithms to avoid clutter
        top_algos = radar.mean(axis=1).sort_values(ascending=False).head(5).index
        cmap = plt.get_cmap("Set1")
        
        for idx, algo in enumerate(top_algos):
            # Fill NaNs with a low baseline score rather than 0 so lines don't crash to the absolute center
            values = radar.loc[algo].fillna(0.4).to_numpy(dtype=float).tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2.5, marker="o", markersize=6, label=algo, color=cmap(idx))
            ax.fill(angles, values, alpha=0.1, color=cmap(idx))
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(modalities, fontsize=10, weight="bold")
        
        # Set y-limit from 0.4 to 1.0 to magnify the performance differences (放大雷达图的差距)
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], color="gray", size=8)
        
        ax.set_title("Exp-3 radar: cross-modal coverage (Magnified)", pad=20, fontsize=13)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, frameon=True)
        outputs.append(_save(fig, out_dir / "exp3_radar_top_algorithms.png"))
    else:
        outputs.append(_empty_notice(out_dir / "exp3_radar_top_algorithms.png", "Exp-3 radar", "Need at least three modalities."))
    return outputs


def plot_error_analysis(results_dir: str | Path, figures_dir: str | Path, metric: str = "auc_roc") -> list[Path]:
    out_dir = ensure_dir(Path(figures_dir) / "error_analysis")
    df = successful_runs(read_experiment(results_dir, "exp1"), metric=metric)
    df = latest_per_run_key(df, ["dataset_name", "algorithm_name"])
    if "artifact_path" not in df.columns:
        df["artifact_path"] = ""
    df = df[df["artifact_path"].fillna("").astype(str).ne("")]
    outputs: list[Path] = []
    if df.empty:
        outputs.append(_empty_notice(out_dir / "error_analysis_no_artifacts.png", "Error analysis", "No Exp-1 artifacts available."))
        return outputs

    selected = df.sort_values(metric, ascending=False).groupby("dataset_name", as_index=False).head(1).head(2)
    case_rows = []
    for _, row in selected.iterrows():
        artifact = load_artifact_npz(row)
        if not artifact or "y_true" not in artifact or "scores" not in artifact:
            continue
        y_true = artifact["y_true"].astype(int).ravel()
        scores = artifact["scores"].astype(float).ravel()
        thr = row.get("best_threshold")
        if not np.isfinite(thr):
            thr = np.quantile(scores, 0.9)
        pred = (scores >= thr).astype(int)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(scores[y_true == 0], bins=40, alpha=0.65, label="normal", color="#4c78a8", density=True)
        ax.hist(scores[y_true == 1], bins=40, alpha=0.65, label="anomaly", color="#e45756", density=True)
        ax.axvline(thr, color="black", linestyle="--", linewidth=1.2, label="best threshold")
        ax.set_title(f"Score distribution: {row['dataset_name']} / {row['algorithm_name']}")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        safe_name = f"{row['dataset_name']}_{row['algorithm_name']}".replace("/", "-")
        outputs.append(_save(fig, out_dir / f"score_distribution_{safe_name}.png"))

        fp_idx = np.flatnonzero((pred == 1) & (y_true == 0))
        fn_idx = np.flatnonzero((pred == 0) & (y_true == 1))
        fp_idx = fp_idx[np.argsort(scores[fp_idx])[::-1]][:5]
        fn_idx = fn_idx[np.argsort(scores[fn_idx])][:5]
        for kind, indices in [("FP", fp_idx), ("FN", fn_idx)]:
            for idx in indices:
                case_rows.append({
                    "dataset_name": row["dataset_name"],
                    "algorithm_name": row["algorithm_name"],
                    "case_type": kind,
                    "sample_index_in_artifact": int(idx),
                    "y_true": int(y_true[idx]),
                    "score": float(scores[idx]),
                    "threshold": float(thr),
                })

    if case_rows:
        cases = pd.DataFrame(case_rows)
        outputs.append(write_table_csv(cases, out_dir / "fp_fn_cases.csv"))
        outputs.append(_plot_table(cases, out_dir / "fp_fn_cases.png", "Representative FP/FN cases", max_rows=20))
    return outputs


def plot_exp4(results_dir: str | Path, figures_dir: str | Path, metric: str = "auc_roc") -> list[Path]:
    out_dir = ensure_dir(Path(figures_dir) / "exp4")
    df = successful_runs(read_experiment(results_dir, "exp4"), metric=metric)
    outputs: list[Path] = []
    if df.empty:
        outputs.append(_empty_notice(out_dir / "exp4_no_results.png", "Exp-4", "No successful Exp-4 rows. Please run Exp-4 first."))
        return outputs

    # Try loading Exp-1 results to extract clean reference baselines for unsupervised models
    df_exp1 = pd.DataFrame()
    try:
        df_exp1 = successful_runs(read_experiment(results_dir, "exp1"), metric=metric)
    except Exception:
        pass

    # Aggregate metric grouped by modality, algorithm, and flip rate
    curve = (
        df.groupby(["modality", "algorithm_name", "label_flip_rate"], as_index=False)[metric]
        .mean()
        .dropna(subset=["label_flip_rate", metric])
    )
    
    # We want to plot one comparison chart per modality
    for modality, part in curve.groupby("modality"):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        # Highly distinctive aesthetic color scheme for the supervised & defended lines
        color_map = {
            "Standard_LightGBM": "#d62728",              # Solid Red (Undefended danger)
            "Defended_LightGBM_IForest_Trim": "#2ca02c",  # Solid Green (Defended success)
            "Defended_LightGBM_IForest_Flip": "#8c564b",  # Solid Brown
            "Standard_XGBoost": "#ff7f0e",               # Solid Orange (Undefended danger)
            "Defended_XGBoost_IForest_Trim": "#1f77b4",   # Solid Blue (Defended success)
            "Defended_XGBoost_IForest_Flip": "#bcbd22",   # Solid Olive
        }
        
        # Filter part to only contain the key models to keep the PNG extremely neat
        part = part[part["algorithm_name"].isin(color_map.keys())]
        if part.empty:
            plt.close(fig)
            continue
        
        for algo, g in part.groupby("algorithm_name"):
            g = g.sort_values("label_flip_rate")
            is_defended = "Defended" in algo
            is_flip = "Flip" in algo
            
            ls = "-" if not is_flip else "--"
            marker = "D" if is_defended else "o"
            color = color_map.get(algo, None)
            
            ax.plot(g["label_flip_rate"] * 100, g[metric], marker=marker, linestyle=ls, linewidth=2.0, label=algo, color=color)
            
        # Draw standalone unsupervised baselines dynamically from Exp-1 as clean horizontal reference lines
        if not df_exp1.empty:
            baselines_spec = {
                "IForest": ("#7f7f7f", "Baseline_IForest"),
                "ECOD": ("#17becf", "Baseline_ECOD"),
                "AutoEncoder": ("#e377c2", "Baseline_AutoEncoder"),
            }
            # Filter Exp-1 to the current modality and our selected baselines
            mod_exp1 = df_exp1[(df_exp1["modality"] == modality) & (df_exp1["algorithm_name"].isin(baselines_spec.keys()))]
            if not mod_exp1.empty:
                mean_scores = mod_exp1.groupby("algorithm_name")[metric].mean()
                for algo, score in mean_scores.items():
                    color, label_name = baselines_spec[algo]
                    ax.axhline(score, color=color, linestyle=":", linewidth=2.0, label=label_name)
            
        ax.set_title(f"Exp-4 Robust Defense Performance ({modality.upper()})")
        ax.set_xlabel("Label flip contamination rate (%)")
        ax.set_ylabel(metric)
        ax.margins(y=0.08)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
        outputs.append(_save(fig, out_dir / f"exp4_defense_comparison_{modality}.png"))
        
    return outputs


def generate_all(results_dir: str | Path, figures_dir: str | Path, metric: str = "auc_roc") -> list[Path]:
    outputs = []
    outputs.extend(plot_exp1(results_dir, figures_dir, metric))
    outputs.extend(plot_exp2(results_dir, figures_dir, metric))
    outputs.extend(plot_exp3(results_dir, figures_dir, metric))
    # outputs.extend(plot_exp4(results_dir, figures_dir, metric))
    outputs.extend(plot_error_analysis(results_dir, figures_dir, metric))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark analysis figures.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"))
    parser.add_argument("--metric", default="auc_roc", choices=["auc_roc", "auc_pr", "f1_best"])
    parser.add_argument("--section", default="all", choices=["all", "exp1", "exp2", "exp3", "exp4", "error"])
    args = parser.parse_args()

    if args.section == "all":
        outputs = generate_all(args.results_dir, args.figures_dir, args.metric)
    elif args.section == "exp1":
        outputs = plot_exp1(args.results_dir, args.figures_dir, args.metric)
    elif args.section == "exp2":
        outputs = plot_exp2(args.results_dir, args.figures_dir, args.metric)
    elif args.section == "exp3":
        outputs = plot_exp3(args.results_dir, args.figures_dir, args.metric)
    elif args.section == "exp4":
        outputs = plot_exp4(args.results_dir, args.figures_dir, args.metric)
    else:
        outputs = plot_error_analysis(args.results_dir, args.figures_dir, args.metric)

    print(f"Generated {len(outputs)} analysis artifacts:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
