"""Statistical significance tests for benchmark results.

Implements Friedman ranking test and a Nemenyi post-hoc Critical Difference
diagram from saved Exp-1 CSV rows. Higher metric values are treated as better.
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
from scipy.stats import friedmanchisquare, studentized_range

from eval.analysis_utils import ensure_dir, latest_per_run_key, read_experiment, successful_runs, write_table_csv


ROOT = Path(__file__).resolve().parent.parent


def _critical_value(k: int, alpha: float) -> float:
    """Nemenyi q_alpha for average-rank differences."""

    return float(studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0))


def build_rank_matrix(results_dir: str | Path, metric: str = "auc_roc") -> pd.DataFrame:
    df = successful_runs(read_experiment(results_dir, "exp1"), metric=metric)
    df = latest_per_run_key(df, ["dataset_name", "algorithm_name"])
    if df.empty:
        return pd.DataFrame()
    # In a cross-modal benchmark, no single dataset has all 26 algorithms.
    # To prevent dropping all rows, we evaluate Friedman/Nemenyi only on the 15 
    # universal algorithms across the tabular, cv, and nlp datasets.
    scores = df.pivot_table(index="dataset_name", columns="algorithm_name", values=metric, aggfunc="mean")
    
    # We drop any algorithms that are missing in more than 50% of the datasets
    threshold = int(scores.shape[0] * 0.5)
    scores = scores.dropna(axis=1, thresh=threshold)
    
    # Then we drop any datasets that don't have results for all remaining algorithms
    scores = scores.dropna(axis=0, how="any")
    
    return scores


def friedman_nemenyi(results_dir: str | Path, metric: str = "auc_roc", alpha: float = 0.05) -> dict:
    scores = build_rank_matrix(results_dir, metric)
    if scores.shape[0] < 2 or scores.shape[1] < 3:
        return {
            "scores": scores,
            "ranks": pd.Series(dtype=float),
            "friedman_statistic": np.nan,
            "friedman_pvalue": np.nan,
            "critical_difference": np.nan,
            "pairwise": pd.DataFrame(),
        }

    ranks_by_dataset = scores.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks_by_dataset.mean(axis=0).sort_values()
    stat, pvalue = friedmanchisquare(*[scores[col].to_numpy() for col in scores.columns])
    k = scores.shape[1]
    n = scores.shape[0]
    q_alpha = _critical_value(k, alpha)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    pair_rows = []
    for i, a in enumerate(avg_ranks.index):
        for b in avg_ranks.index[i + 1:]:
            diff = abs(float(avg_ranks[a] - avg_ranks[b]))
            pair_rows.append({
                "algorithm_a": a,
                "algorithm_b": b,
                "rank_a": float(avg_ranks[a]),
                "rank_b": float(avg_ranks[b]),
                "rank_diff": diff,
                "critical_difference": float(cd),
                "significantly_different": bool(diff > cd),
            })

    return {
        "scores": scores,
        "ranks": avg_ranks,
        "friedman_statistic": float(stat),
        "friedman_pvalue": float(pvalue),
        "critical_difference": float(cd),
        "pairwise": pd.DataFrame(pair_rows),
    }


def plot_cd_diagram(ranks: pd.Series, cd: float, path: str | Path, title: str) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    if ranks.empty or not np.isfinite(cd):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.55, title, ha="center", fontsize=13, weight="bold")
        ax.text(0.5, 0.35, "Need at least 2 datasets and 3 complete algorithms.", ha="center", fontsize=10)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return out

    ranks = ranks.sort_values()
    fig_h = max(4.0, 0.28 * len(ranks) + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    min_rank = max(1, int(np.floor(ranks.min())) - 1)
    max_rank = int(np.ceil(ranks.max())) + 1
    ax.set_xlim(min_rank, max_rank)
    ax.set_ylim(-1, len(ranks) + 2)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_xaxis()

    y_base = len(ranks)
    for idx, (algo, rank) in enumerate(ranks.items()):
        # We drop the hlines completely and put everything on a single horizontal axis
        pass
        
    # Redraw everything on a single 1D axis for the classic Nemenyi CD look
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(min_rank, max_rank)
    ax.set_ylim(-1, 3)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    ax.set_title(title, pad=30)
    ax.invert_xaxis()
    
    # Draw the main axis line
    ax.hlines(0, min_rank, max_rank, color="black", linewidth=1.5)
    ax.set_xticks(np.arange(min_rank, max_rank + 1))
    
    # Plot dots and labels
    texts = []
    # Alternate drawing labels above and below the axis line
    for idx, (algo, rank) in enumerate(ranks.items()):
        ax.plot(rank, 0, "o", color="#4c78a8", markersize=6, zorder=5)
        # Alternate height: 0.2, -0.3, 0.4, -0.5 to prevent overlap
        y_offset = (0.2 + (idx % 3) * 0.25) * (1 if idx % 2 == 0 else -1)
        
        # Draw a small vertical stem line
        ax.vlines(rank, 0, y_offset, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        
        # Place the text
        val_str = f"{algo} ({rank:.2f})"
        texts.append(ax.text(rank, y_offset + (0.1 if y_offset > 0 else -0.1), val_str, 
                             va="center", ha="center", fontsize=8, weight="bold"))

    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), expand_points=(1.5, 1.5))
    except ImportError:
        pass

    # Critical difference ruler at top
    ruler_y = 2.0
    start = min_rank
    end = min(start + cd, max_rank)
    ax.hlines(ruler_y, start, end, color="#e45756", linewidth=3)
    ax.vlines([start, end], ruler_y - 0.2, ruler_y + 0.2, color="#e45756", linewidth=2)
    ax.text((start + end) / 2, ruler_y + 0.3, f"CD={cd:.2f}", ha="center", fontsize=10, weight="bold", color="#e45756")

    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def run_significance(
    results_dir: str | Path,
    figures_dir: str | Path,
    metric: str = "auc_roc",
    alpha: float = 0.05,
) -> list[Path]:
    out_dir = ensure_dir(Path(figures_dir) / "exp1")
    result = friedman_nemenyi(results_dir, metric=metric, alpha=alpha)
    outputs: list[Path] = []

    ranks = result["ranks"]
    if not ranks.empty:
        outputs.append(write_table_csv(
            ranks.rename("average_rank").reset_index().rename(columns={"index": "algorithm_name"}),
            out_dir / "friedman_average_ranks.csv",
        ))

    summary = pd.DataFrame([{
        "metric": metric,
        "n_datasets": int(result["scores"].shape[0]) if isinstance(result["scores"], pd.DataFrame) else 0,
        "n_algorithms": int(result["scores"].shape[1]) if isinstance(result["scores"], pd.DataFrame) else 0,
        "alpha": alpha,
        "friedman_statistic": result["friedman_statistic"],
        "friedman_pvalue": result["friedman_pvalue"],
        "critical_difference": result["critical_difference"],
    }])
    outputs.append(write_table_csv(summary, out_dir / "friedman_nemenyi_summary.csv"))
    if not result["pairwise"].empty:
        outputs.append(write_table_csv(result["pairwise"], out_dir / "nemenyi_pairwise.csv"))
    outputs.append(plot_cd_diagram(
        ranks,
        result["critical_difference"],
        out_dir / "nemenyi_cd_diagram.png",
        f"Nemenyi Critical Difference diagram ({metric})",
    ))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Friedman/Nemenyi significance analysis.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"))
    parser.add_argument("--metric", default="auc_roc", choices=["auc_roc", "auc_pr", "f1_best"])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    outputs = run_significance(args.results_dir, args.figures_dir, args.metric, args.alpha)
    print(f"Generated {len(outputs)} significance artifacts:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
