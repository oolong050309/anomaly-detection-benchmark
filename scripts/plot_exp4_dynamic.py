import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Plot Exp4: Dynamic Best Defender vs Standard Baseline")
    parser.add_argument("--input", type=str, default=str(ROOT / "results" / "exp4_results.csv"))
    parser.add_argument("--output", type=str, default=str(ROOT / "figures" / "exp4" / "exp4_dynamic_best_defense.png"))
    args = parser.parse_args()

    # Create output dir
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    # Exclude TabPFN to focus on conventional ML models
    df = df[~df["algorithm_name"].str.contains("TabPFN")]
    
    # We will plot 4 main baseline models
    base_models = ["LightGBM", "XGBoost", "RF", "MLP"]
    rates = [0.0, 0.05, 0.10, 0.20]

    # Setup matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    for i, bm in enumerate(base_models):
        ax = axes[i]
        
        std_aucs = []
        best_def_aucs = []
        best_def_names = []
        
        for r in rates:
            # Standard
            std_row = df[(df["algorithm_name"] == f"Standard_{bm}") & (df["label_flip_rate"] == r)]
            std_auc = std_row["auc_roc"].mean() if not std_row.empty else np.nan
            std_aucs.append(std_auc)
            
            # Defended
            def_df = df[(df["algorithm_name"].str.startswith(f"Defended_{bm}_")) & (df["label_flip_rate"] == r)]
            if not def_df.empty:
                best_name = def_df.groupby("algorithm_name")["auc_roc"].mean().idxmax()
                best_auc = def_df.groupby("algorithm_name")["auc_roc"].mean().max()
                
                # Extract denoiser name (e.g. Defended_LightGBM_IQR_Trim -> IQR_Trim)
                short_name = best_name.split(f"{bm}_")[1]
                
                best_def_names.append(short_name)
                best_def_aucs.append(best_auc)
            else:
                best_def_names.append("")
                best_def_aucs.append(np.nan)

        # Plot lines
        ax.plot(rates, std_aucs, marker='o', linestyle='--', color='#e74c3c', linewidth=2.5, markersize=8, label=f"Standard Base")
        ax.plot(rates, best_def_aucs, marker='s', linestyle='-', color='#3498db', linewidth=3, markersize=9, label=f"Dynamic Best Defender")
        
        # Annotate the specific denoiser chosen at each rate
        for x, y, label in zip(rates, best_def_aucs, best_def_names):
            if pd.notna(y):
                # Add text slightly above the point
                ax.text(x, y + 0.001, label, fontsize=10, ha='center', va='bottom', color='#2980b9', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # Annotate GAP
        for x, y_def, y_std in zip(rates, best_def_aucs, std_aucs):
            if pd.notna(y_def) and pd.notna(y_std):
                gap = y_def - y_std
                if gap > 0:
                    ax.text(x, y_def + 0.003, f"+{gap*100:.2f}%", fontsize=10, ha='center', va='bottom', color='green', fontweight='bold')
                elif gap < 0:
                    ax.text(x, y_def - 0.003, f"{gap*100:.2f}%", fontsize=10, ha='center', va='top', color='red', fontweight='bold')

        ax.set_title(f"Model: {bm}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Label Contamination Rate (Flip %)", fontsize=13)
        ax.set_ylabel("Average AUC-ROC", fontsize=13)
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{int(r*100)}%" for r in rates])
        
        # Legend styling
        ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)

    plt.suptitle("Robust Defense Crossover: Dynamic Best Defender vs Standard Baseline", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(args.output)
    print(f"✅ Successfully saved dynamic crossover plot to {args.output}")

if __name__ == "__main__":
    main()
