"""Run benchmark experiments end to end.

Examples:
    python run_all.py --exp exp1 --data-root /root/autodl-tmp/final_project/data
    python run_all.py --exp exp2 --modalities tabular timeseries
    python run_all.py --exp all --output-dir results/server_run_seed42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from models.device import cuda_available, get_preferred_device


ROOT = Path(__file__).resolve().parent
EXPERIMENT_MODULES = {
    "exp1": "experiments.exp1_baseline",
    "exp2": "experiments.exp2_contamination",
    "exp3": "experiments.exp3_cross_modal",
    "exp4": "experiments.exp4_defense",
}


def _run_module(module: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    print(f"\n{'=' * 80}", flush=True)
    print("Running:", " ".join(cmd), flush=True)
    print(f"{'=' * 80}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection benchmark experiments")
    parser.add_argument("--exp", choices=["exp1", "exp2", "exp3", "exp4", "all"], default="all")
    parser.add_argument("--data-root", default=None,
                        help="Server data root. Defaults to AD_DATA_ROOT or ./data inside adapters.")
    parser.add_argument("--output-dir", default=str(ROOT / "results"),
                        help="Directory for CSV logs and per-run score artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43],
                        help="List of random seeds to run (default: 41 42 43).")
    parser.add_argument("--modality", choices=["tabular", "timeseries", "graph", "all"],
                        default=None, help="Exp-1 modality filter.")
    parser.add_argument("--modalities", nargs="+",
                        choices=["tabular", "cv", "nlp", "timeseries", "graph"],
                        default=None, help="Exp-2/Exp-3 modality filter.")
    parser.add_argument("--analyze", action="store_true",
                        help="After experiments, run analyze_results.py to regenerate figures.")
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"),
                        help="Output directory used by --analyze.")
    parser.add_argument("--metric", default="auc_roc", choices=["auc_roc", "auc_pr", "f1_best"],
                        help="Primary metric for --analyze.")
    parser.add_argument("--skip-significance", action="store_true",
                        help="Skip Friedman/Nemenyi tests when running --analyze.")
    args = parser.parse_args()

    selected = ["exp1", "exp2", "exp3", "exp4"] if args.exp == "all" else [args.exp]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Device preference: {get_preferred_device()} "
        f"(CUDA available: {cuda_available()})",
        flush=True,
    )

    t0 = time.perf_counter()
    for seed in args.seeds:
        print(f"\n" + "*" * 80)
        print(f"*** STARTING RUN FOR SEED: {seed} ***")
        print("*" * 80)
        
        for exp in selected:
            extra = [
                "--output-dir", str(output_dir),
                "--seed", str(seed),
                "--timestamped"
            ]
            if args.data_root:
                extra.extend(["--data-root", args.data_root])
            if exp == "exp1" and args.modality:
                extra.extend(["--modality", args.modality])
            if exp == "exp2" and args.modalities:
                exp2_modalities = [m for m in args.modalities if m in {"tabular", "timeseries", "graph"}]
                if not exp2_modalities:
                    raise ValueError("Exp-2 only supports modalities: tabular, timeseries, graph")
                extra.extend(["--modalities", *exp2_modalities])
            if exp == "exp3" and args.modalities:
                extra.extend(["--modalities", *args.modalities])
            if exp == "exp4" and args.modalities:
                extra.extend(["--modalities", *args.modalities])
            _run_module(EXPERIMENT_MODULES[exp], extra)

    elapsed = time.perf_counter() - t0
    print(f"\nAll requested experiments completed in {elapsed:.1f}s")
    print(f"CSV logs and score artifacts are under: {output_dir}")

    if args.analyze:
        analyze_cmd = [
            sys.executable,
            str(ROOT / "analyze_results.py"),
            "--results-dir", str(output_dir),
            "--figures-dir", str(Path(args.figures_dir)),
            "--metric", args.metric,
        ]
        if args.skip_significance:
            analyze_cmd.append("--skip-significance")
        print(f"\n{'=' * 80}", flush=True)
        print("Running analysis:", " ".join(analyze_cmd), flush=True)
        print(f"{'=' * 80}", flush=True)
        subprocess.run(analyze_cmd, cwd=str(ROOT), check=True)
        print(f"Figures written to: {args.figures_dir}")


if __name__ == "__main__":
    main()
