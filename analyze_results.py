"""Generate all evaluation and analysis artifacts from saved results."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.significance import run_significance
from eval.visualize import generate_all


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures/tables from experiment results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"))
    parser.add_argument("--metric", default="auc_roc", choices=["auc_roc", "auc_pr", "f1_best"])
    parser.add_argument("--skip-significance", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    outputs = generate_all(args.results_dir, args.figures_dir, args.metric)
    if not args.skip_significance:
        outputs.extend(run_significance(args.results_dir, args.figures_dir, args.metric, args.alpha))

    print(f"Generated {len(outputs)} analysis artifacts:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
