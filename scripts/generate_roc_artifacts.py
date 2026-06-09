"""为本地交互式 ROC 生成 score artifact（y_true + scores npz）。

CSV 中 artifact_path 若指向服务器，可在 Windows 本地重跑当前数据集上的算法，
写入 ``results/artifacts/exp1/``；Streamlit 会自动按 dataset+algorithm 匹配。

示例：
    python -m scripts.generate_roc_artifacts --dataset cardio
    python -m scripts.generate_roc_artifacts --dataset cardio --fast
    python -m scripts.generate_roc_artifacts --dataset cardio --algorithms IForest XGBoost ECOD
    python -m scripts.generate_roc_artifacts --dataset 276_IOPS_id_17_WebService --modality timeseries
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset
from experiments.common import fit_and_score, save_run_artifact
from experiments.exp1_baseline import (
    _get_graph_algos,
    _get_tabular_algos,
    _get_timeseries_algos,
)

# 优先无 pyod 依赖；若已安装 pyod / xgboost 会自动跑通更多算法
FAST_TABULAR = [
    "IQR", "LOF", "KNN", "IForest", "ECOD", "COPOD", "LR", "RF", "XGBoost",
]
DEFAULT_SEED = 42


def _algo_table(modality: str) -> dict[str, tuple]:
    if modality in {"tabular", "cv", "nlp", "adbench"}:
        rows = _get_tabular_algos()
    elif modality in {"timeseries", "ts"}:
        rows = _get_timeseries_algos()
    elif modality in {"graph", "gadbench"}:
        rows = _get_graph_algos()
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return {name: (Cls, kwargs, needs_y) for name, Cls, kwargs, needs_y in rows}


def _load_bundle(dataset: str, modality: str, data_root: str | None):
    if modality in {"timeseries", "ts"}:
        return load_dataset(
            dataset,
            modality="timeseries",
            data_root=data_root,
            window_size=100,
            stride=10,
        )
    if modality in {"graph", "gadbench"}:
        return load_dataset(dataset, modality="graph", data_root=data_root)
    return load_dataset(dataset, data_root=data_root)


def _run_one(
    *,
    name: str,
    Cls,
    kwargs: dict,
    needs_y: bool,
    dataset: str,
    modality: str,
    output_dir: Path,
    X_tr,
    X_te,
    y_tr,
    y_te,
    seed: int,
    extras: dict | None = None,
) -> str | None:
    try:
        det = Cls(contamination=0.1, random_state=seed, **kwargs)
        if modality == "timeseries" and name == "MatrixProfile" and extras:
            fit_input = extras.get("train_seq")
            score_input = extras.get("test_seq")
        else:
            fit_input = (X_tr, y_tr) if needs_y else X_tr
            score_input = X_te
        if fit_input is None or score_input is None:
            raise ValueError("missing fit/score input")

        scores, fit_t, pred_t = fit_and_score(
            det,
            fit_input if isinstance(fit_input, tuple) else (fit_input,),
            score_input,
        )
        y_score = np.asarray(y_te).astype(int).reshape(-1)
        if len(scores) != len(y_score):
            raise ValueError(f"score length {len(scores)} != y_test {len(y_score)}")

        from experiments.common import make_run_id

        run_id = make_run_id("exp1", dataset, name)
        path = save_run_artifact(
            output_dir=output_dir,
            experiment_name="exp1",
            run_id=run_id,
            y_true=y_score,
            scores=scores,
            y_train=y_tr if needs_y else None,
            metadata={
                "run_id": run_id,
                "experiment_name": "exp1",
                "modality": modality,
                "dataset_name": dataset,
                "algorithm_name": name,
                "seed": seed,
                "fit_time_sec": fit_t,
                "predict_time_sec": pred_t,
                "source": "generate_roc_artifacts",
            },
        )
        from eval.metrics import evaluate_all

        m = evaluate_all(y_score, scores)
        print(
            f"  OK  {name:14s}  AUC-ROC={m['auc_roc']:.4f}  "
            f"-> {Path(path).name}"
        )
        return path
    except Exception as exc:
        print(f"  FAIL {name:14s}  {exc!r}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local ROC score artifacts")
    parser.add_argument("--dataset", required=True, help="e.g. cardio, reddit, 006_NAB_id_6_Traffic")
    parser.add_argument("--modality", default=None, help="tabular/cv/nlp/timeseries/graph; default infer")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fast", action="store_true", help="tabular 快速 8 算法（适合答辩演示）")
    parser.add_argument("--algorithms", nargs="*", help="指定算法名，默认该模态全部")
    args = parser.parse_args()

    from adapters.load_dataset import infer_modality

    try:
        modality = (args.modality or infer_modality(args.dataset)).lower()
    except KeyError as exc:
        print(f"ERROR: {exc}. 请显式指定 --modality timeseries|graph|tabular")
        sys.exit(1)
    if modality == "adbench":
        modality = "tabular"

    table = _algo_table(modality)
    if args.algorithms:
        names = args.algorithms
    elif args.fast and modality in {"tabular", "cv", "nlp"}:
        names = [n for n in FAST_TABULAR if n in table]
    else:
        names = list(table.keys())

    missing = [n for n in names if n not in table]
    if missing:
        print(f"Unknown algorithms for {modality}: {missing}")
        sys.exit(1)

    print(f"Dataset: {args.dataset} ({modality})  seed={args.seed}")
    print(f"Algorithms ({len(names)}): {', '.join(names)}")
    bundle = _load_bundle(args.dataset, modality, str(args.data_root))
    X_tr, X_te, y_tr, y_te = bundle.as_tuple()
    extras = dict(bundle.extras) if bundle.extras else {}
    if modality == "timeseries":
        extras.setdefault("train_seq", bundle.extras.get("raw_values", [])[: bundle.extras.get("train_end", 0)])
        extras.setdefault("test_seq", bundle.extras.get("raw_values", [])[bundle.extras.get("train_end", 0) :])

    ok = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            Cls, kwargs, needs_y = table[name]
            if _run_one(
                name=name,
                Cls=Cls,
                kwargs=kwargs,
                needs_y=needs_y,
                dataset=args.dataset,
                modality=modality,
                output_dir=args.results_dir,
                X_tr=X_tr,
                X_te=X_te,
                y_tr=y_tr,
                y_te=y_te,
                seed=args.seed,
                extras=extras,
            ):
                ok += 1

    art_dir = args.results_dir / "artifacts" / "exp1"
    print(f"\nDone: {ok}/{len(names)} artifacts in {art_dir}")
    print("Refresh Streamlit 基准对比页即可查看交互式 ROC。")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
