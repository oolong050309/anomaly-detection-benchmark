"""Exp-3：跨模态对比实验。

通用算法（接受二维 (n, d) numpy 数组的算法）在 5 类形态上运行，
对比同一算法在不同形态上的排名差异。

对于时序和图数据：
- 时序：使用 adapter 切窗后的二维窗口数组 ``(n_windows, window_size)``
- 图：使用图节点特征矩阵 ``(n_nodes, n_features)`` + 节点级 mask 划分

结果写入 ``results/exp3_results.csv``。

用法：
    python -m experiments.exp3_cross_modal
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset
from experiments.common import fit_and_score, record_failure, record_success
from experiments.exp1_baseline import (
    CV_DATASETS,
    GRAPH_DATASETS,
    NLP_DATASETS,
    TABULAR_DATASETS,
    TIMESERIES_DATASETS,
    _borrow_for_supervised,
)

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp3_results.csv")
OUTPUT_DIR = str(ROOT / "results")


def _get_universal_algos():
    """15 个通用算法（接受 (n, d) 特征矩阵），与 Exp-1 tabular / Exp-2 完全一致。

    跨模态对比的目的本就是观察同一算法在不同形态上的适用性差异，
    因此不预先剔除"可能不稳"的深度/基础模型 —— 它们在哪些模态稳、
    哪些模态失效（如 TabPFN 受 d<=100 限制，在 CV 512 维 / NLP 768 维上会失败），
    本身就是要报告的结论。失败会被 record_failure 如实记录。
    """
    from models import (
        AutoEncoderDetector, COPODDetector, DeepSVDDDetector,
        ECODDetector, IForestDetector, IQRDetector, KNNDetector,
        LightGBMDetector, LOFDetector, LogisticRegressionDetector,
        MLPDetector, OCSVMDetector, RandomForestDetector,
        TabPFNDetector, XGBoostDetector,
    )
    return [
        # (name, class, kwargs, needs_y)
        ("IQR", IQRDetector, {}, False),
        ("LOF", LOFDetector, {}, False),
        ("KNN", KNNDetector, {}, False),
        ("IForest", IForestDetector, {}, False),
        ("ECOD", ECODDetector, {}, False),
        ("COPOD", COPODDetector, {}, False),
        ("OCSVM", OCSVMDetector, {}, False),
        ("AutoEncoder", AutoEncoderDetector, {"epoch_num": 100}, False),
        ("DeepSVDD", DeepSVDDDetector, {"epochs": 100}, False),
        ("LR", LogisticRegressionDetector, {}, True),
        ("RF", RandomForestDetector, {"n_estimators": 100}, True),
        ("MLP", MLPDetector, {"max_iter": 200}, True),
        ("XGBoost", XGBoostDetector, {"n_estimators": 100}, True),
        ("LightGBM", LightGBMDetector, {"n_estimators": 100}, True),
        ("TabPFN", TabPFNDetector, {}, True),
    ]


# 跨模态对比：与 Exp-1 使用完全相同的全量数据集（直接复用 Exp-1 常量，
# 避免两处数据集列表漂移）。Exp-1 的 --modality tabular 把 CV/NLP 也并入
# tabular 跑，这里按真实模态拆开，5 种形态各自用各自的全量列表。
MODALITY_DATASETS = {
    "tabular": list(TABULAR_DATASETS),
    "cv": list(CV_DATASETS),
    "nlp": list(NLP_DATASETS),
    "timeseries": list(TIMESERIES_DATASETS),
    "graph": list(GRAPH_DATASETS),
}


def _load_as_2d(modality: str, ds_name: str, data_root=None):
    """统一返回 (X_train, X_test, y_train, y_test) 二维数组。"""
    if modality in ("tabular", "cv", "nlp"):
        bundle = load_dataset(ds_name, data_root=data_root)
        return bundle, bundle.as_tuple()

    if modality == "timeseries":
        bundle = load_dataset(ds_name, modality="timeseries",
                              data_root=data_root,
                              window_size=100, stride=10)
        return bundle, bundle.as_tuple()

    if modality == "graph":
        bundle = load_dataset(ds_name, modality="graph", data_root=data_root)
        # 把节点特征当表格数据用
        return bundle, bundle.as_tuple()

    raise ValueError(f"Unknown modality: {modality}")


def main():
    global SEED, LOG_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Exp-3: Cross-modal comparison")
    parser.add_argument("--modalities", nargs="+",
                        default=["tabular", "cv", "nlp", "timeseries", "graph"],
                        choices=["tabular", "cv", "nlp", "timeseries", "graph"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="把本次运行结果隔离到 <output-dir>/runs/exp3_<modality_tag>_<UTC>/ 子目录，"
             "避免和历史 CSV 叠加。会在该目录下生成 exp3_results.csv 与 artifacts/。",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="搭配 --timestamped 使用，给本次运行打额外标签，目录名变成 "
             "exp3_<modality_tag>_<run-tag>_<UTC>/。无 --timestamped 时此参数被忽略。",
    )
    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_DIR = str(Path(args.output_dir))

    # --timestamped: 把本次运行隔到独立目录，不污染历史 CSV
    if args.timestamped:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        modality_tag = "_".join(args.modalities)
        tag_part = f"_{args.run_tag}" if args.run_tag else ""
        run_dir = Path(OUTPUT_DIR) / "runs" / f"exp3_{modality_tag}{tag_part}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR = str(run_dir)
        print(f"[exp3] timestamped run dir → {OUTPUT_DIR}")

    LOG_PATH = str(Path(args.log_path) if args.log_path else Path(OUTPUT_DIR) / "exp3_results.csv")

    algos = _get_universal_algos()
    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    for modality, datasets in MODALITY_DATASETS.items():
        if modality not in args.modalities:
            continue
        print(f"\n{'='*70}\nModality: {modality} ({len(datasets)} datasets)\n{'='*70}")
        for ds in datasets:
            print(f"\n  --- {modality}/{ds} ---")
            try:
                bundle, (X_tr, X_te, y_tr, y_te) = _load_as_2d(modality, ds, args.data_root)
                print(f"  Loaded: train={X_tr.shape}, test={X_te.shape}, "
                      f"anomaly_rate={y_te.mean():.4f}")
            except Exception as e:
                print(f"  [SKIP] {e}")
                continue

            for name, Cls, kwargs, needs_y in algos:
                n_total += 1
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    if needs_y:
                        if len(np.unique(y_tr)) < 2:
                            # Bug 1 修复：从 test 借异常窗口
                            X_tr_use, y_tr_use, X_te_use, y_te_use, borrowed_n, borrow_note = (
                                _borrow_for_supervised(X_tr, y_tr, X_te, y_te, rng_seed=SEED)
                            )
                            if len(np.unique(y_tr_use)) < 2:
                                raise RuntimeError("y_train has single class (borrow failed)")
                        else:
                            X_tr_use, y_tr_use, X_te_use, y_te_use = X_tr, y_tr, X_te, y_te
                            borrow_note = ""
                        fit_args = (X_tr_use, y_tr_use)
                    else:
                        X_te_use, y_te_use = X_te, y_te
                        borrow_note = ""
                        fit_args = (X_tr,)
                    scores, fit_t, pred_t = fit_and_score(det, fit_args, X_te_use)

                    notes_str = f"exp3_modality={modality}"
                    if borrow_note:
                        notes_str += f"; {borrow_note}"

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = record_success(
                            log_path=LOG_PATH,
                            output_dir=OUTPUT_DIR,
                            experiment_name="exp3",
                            modality=modality,
                            dataset_name=f"{modality}/{ds}",
                            algorithm_name=name,
                            y_true=y_te_use,
                            scores=scores,
                            fit_time_sec=fit_t,
                            predict_time_sec=pred_t,
                            seed=SEED,
                            fit_params=kwargs,
                            notes=notes_str,
                            X_train=X_tr_use if needs_y else X_tr,
                            y_train=y_tr_use if needs_y else y_tr,
                        )
                    print(f"    [{name:>10s}] AUC-ROC={m['auc_roc']:.4f}")
                    n_ok += 1
                except Exception as e:
                    record_failure(
                        log_path=LOG_PATH,
                        experiment_name="exp3",
                        modality=modality,
                        dataset_name=f"{modality}/{ds}",
                        algorithm_name=name,
                        seed=SEED,
                        error=e,
                        fit_params=kwargs,
                        notes=f"exp3_modality={modality}",
                        X_train=X_tr if "X_tr" in locals() else None,
                        y_train=y_tr if "y_tr" in locals() else None,
                        y_test=y_te if "y_te" in locals() else None,
                    )
                    print(f"    [{name:>10s}] FAILED: {e!r}"[:80])

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-3 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"Artifacts: {Path(OUTPUT_DIR) / 'artifacts' / 'exp3'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
