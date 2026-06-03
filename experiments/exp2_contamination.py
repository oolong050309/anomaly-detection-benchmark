"""Exp-2: training-set contamination robustness.

The experiment uses the same contamination grid for three data shapes:
tabular feature matrices, time-series windows, and graph node features.

For unsupervised detectors, the training matrix is rebuilt so the retained
anomaly ratio is approximately 0% / 1% / 5% / 10% / 20%.
For supervised detectors, features are kept fixed and labels are flipped
symmetrically at the same rates. Each completed run saves CSV metrics plus the
raw score/label artifact needed for later plotting and error analysis.
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
from data.contaminate import contaminate_supervised, contaminate_unsupervised
from experiments.common import (
    anomaly_rate,
    as_json,
    fit_and_score,
    record_failure,
    record_success,
    shape0,
)
from experiments.exp1_baseline import _borrow_for_supervised
from eval.logger import log_experiment

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp2_results.csv")
OUTPUT_DIR = str(ROOT / "results")

CONTAMINATION_RATES = [0.0, 0.01, 0.05, 0.10, 0.20]

# 无监督污染需要一个「以正常样本为主」的训练集做基底。个别 TSB-AD 序列
# （如 149_Stock_id_1_Finance）训练段标签全为异常，切窗后正常窗口数为 0，
# 污染后的训练集会退化到 0~1 个样本：rate=0 时 0 样本（"X must contain at
# least one sample"），rate>0 时被迫只留 1 个样本（LOF/KNN 的 n_neighbors >=
# n_samples、AutoEncoder 训练循环不执行）。阈值取 25 以覆盖 LOF 默认
# n_neighbors=20（需 n_samples > 20）；正常数据集的训练窗口都在上千，不受影响。
MIN_TRAIN_SAMPLES = 25

DEFAULT_DATASETS = {
    "tabular": [
        "cardio", "thyroid", "satellite", "shuttle", "credit_card",
        "pima", "annthyroid", "mammography", "pendigits",
    ],
    "timeseries": [
        "006_NAB_id_6_Traffic",
        "149_Stock_id_1_Finance",
        "171_MITDB_id_2_Medical",
        "225_MGAB_id_1_Synthetic",
        "276_IOPS_id_17_WebService",
        "331_UCR_id_29_Facility",
        "337_UCR_id_35_HumanActivity",
        "550_SWaT_id_1_Sensor",
    ],
    "graph": ["tfinance", "reddit", "amazon", "weibo"],
    "cv": ["cifar10_0", "cifar10_1", "fashionmnist_0", "fashionmnist_1"],
    "nlp": ["20news_0", "20news_1", "agnews_0", "amazon"],
}


def _get_common_algos():
    """Common detectors that accept a 2D feature matrix."""
    from models import (
        AutoEncoderDetector,
        COPODDetector,
        DeepSVDDDetector,
        ECODDetector,
        IForestDetector,
        IQRDetector,
        KNNDetector,
        LightGBMDetector,
        LOFDetector,
        LogisticRegressionDetector,
        MLPDetector,
        OCSVMDetector,
        RandomForestDetector,
        TabPFNDetector,
        XGBoostDetector,
    )

    return [
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


def _load_as_2d(modality: str, ds_name: str, data_root=None):
    if modality in ("tabular", "cv", "nlp"):
        # CV(ResNet18 嵌入)/ NLP(BERT 嵌入)在 ADBench 里本就是 (n, d) 特征矩阵，
        # 与 tabular 同构，加载方式相同，污染（删异常行 / 翻标签）行独立、可直接套用。
        bundle = load_dataset(ds_name, data_root=data_root)
    elif modality == "timeseries":
        bundle = load_dataset(
            ds_name,
            modality="timeseries",
            data_root=data_root,
            window_size=100,
            stride=10,
        )
    elif modality == "graph":
        bundle = load_dataset(ds_name, modality="graph", data_root=data_root)
    else:
        raise ValueError(f"Unsupported modality for Exp-2: {modality}")
    return bundle, bundle.as_tuple()


def _record_skip(
    *,
    modality: str,
    dataset_name: str,
    algo_name: str,
    kwargs: dict,
    rate: float,
    mode: str,
    reason: str,
    X_fit,
    y_fit,
    y_test,
) -> None:
    """把一个无法定义的污染 case 记成 status='skipped'，而不是 failed。

    退化原因（如训练段无正常窗口 → 污染后样本数不足）不是算法 bug，也不是
    可修复的运行错误，强行跑只会抛 LOF/KNN/AutoEncoder 的二次报错。记成
    skipped 让下游 analyze_results 的 ``status=='success'`` 过滤天然排除它，
    既不污染汇总，也在 CSV 留下可追溯的原因。
    """
    log_experiment(
        dataset_name,
        algo_name,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        SEED,
        notes=f"target_rate={rate:.2f}; {reason}",
        log_path=str(LOG_PATH),
        experiment_name="exp2",
        modality=modality,
        status="skipped",
        contamination_mode=mode,
        contamination_rate=rate if mode == "unsupervised_anomaly_keep_ratio" else None,
        label_flip_rate=rate if mode == "supervised_label_flip" else None,
        n_train=shape0(X_fit),
        n_test=shape0(y_test),
        train_anomaly_rate=anomaly_rate(y_fit),
        test_anomaly_rate=anomaly_rate(y_test),
        fit_params=as_json(kwargs or {}),
        error_type="DegenerateContamination",
        error_message=reason,
    )


def _run_one(
    *,
    modality: str,
    dataset_name: str,
    algo_name: str,
    cls,
    kwargs: dict,
    needs_y: bool,
    rate: float,
    X_train,
    y_train,
    X_test,
    y_test,
) -> bool:
    mode = "supervised_label_flip" if needs_y else "unsupervised_anomaly_keep_ratio"
    try:
        if needs_y:
            X_fit, y_fit, meta = contaminate_supervised(X_train, y_train, rate, seed=SEED)
            if len(np.unique(y_fit)) < 2:
                # Bug 1 fix: borrow anomaly/normal from test set
                X_fit_aug, y_fit_aug, X_te_kept, y_te_kept, borrowed_n, borrow_note = (
                    _borrow_for_supervised(X_fit, y_fit, X_test, y_test, rng_seed=SEED)
                )
                if len(np.unique(y_fit_aug)) < 2:
                    raise RuntimeError("contaminated training labels contain a single class (borrow failed)")
                X_fit, y_fit = X_fit_aug, y_fit_aug
                meta["borrow_note"] = borrow_note
            fit_args = (X_fit, y_fit)
            label_flip_rate = rate
            train_for_log = y_fit
        else:
            X_fit, y_fit, meta = contaminate_unsupervised(X_train, y_train, rate, seed=SEED)
            # 退化保护：训练段无正常窗口时，污染后的训练集会塌缩到 0~1 个样本，
            # 对无监督算法没有可定义的语义（LOF/KNN/AE 会二次报错）。记成 skipped。
            n_fit = shape0(X_fit) or 0
            if n_fit < MIN_TRAIN_SAMPLES:
                n_normal_src = int(np.sum(np.asarray(y_train).astype(int) == 0))
                reason = (
                    f"degenerate unsupervised contamination: only {n_fit} train "
                    f"sample(s) after keep-ratio (source normal windows="
                    f"{n_normal_src}); need >= {MIN_TRAIN_SAMPLES}"
                )
                _record_skip(
                    modality=modality,
                    dataset_name=dataset_name,
                    algo_name=algo_name,
                    kwargs=kwargs,
                    rate=rate,
                    mode=mode,
                    reason=reason,
                    X_fit=X_fit,
                    y_fit=y_fit,
                    y_test=y_test,
                )
                print(f"    [{algo_name:>12s}] rate={rate:.2f} SKIPPED: {reason}"[:110])
                return False
            fit_args = (X_fit,)
            label_flip_rate = None
            train_for_log = y_fit

        det = cls(contamination=0.1, random_state=SEED, **kwargs)
        scores, fit_t, pred_t = fit_and_score(det, fit_args, X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = record_success(
                log_path=LOG_PATH,
                output_dir=OUTPUT_DIR,
                experiment_name="exp2",
                modality=modality,
                dataset_name=dataset_name,
                algorithm_name=algo_name,
                y_true=y_test,
                scores=scores,
                fit_time_sec=fit_t,
                predict_time_sec=pred_t,
                seed=SEED,
                fit_params=kwargs,
                notes=f"target_rate={rate:.2f}; meta={meta}",
                contamination_mode=mode,
                contamination_rate=rate if not needs_y else None,
                label_flip_rate=label_flip_rate,
                X_train=X_fit,
                y_train=train_for_log,
                extra_metadata={"contamination_meta": meta},
            )
        print(f"    [{algo_name:>12s}] rate={rate:.2f} AUC-ROC={m['auc_roc']:.4f}")
        return True
    except Exception as e:
        record_failure(
            log_path=LOG_PATH,
            experiment_name="exp2",
            modality=modality,
            dataset_name=dataset_name,
            algorithm_name=algo_name,
            seed=SEED,
            error=e,
            fit_params=kwargs,
            notes=f"target_rate={rate:.2f}",
            contamination_mode=mode,
            contamination_rate=rate if not needs_y else None,
            label_flip_rate=rate if needs_y else None,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test,
        )
        print(f"    [{algo_name:>12s}] rate={rate:.2f} FAILED: {e!r}"[:100])
        return False


def main():
    global SEED, LOG_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Exp-2: contamination robustness")
    parser.add_argument("--modalities", nargs="+", default=["tabular", "timeseries", "graph", "cv", "nlp"],
                        choices=["tabular", "timeseries", "graph", "cv", "nlp"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Override datasets for every selected modality.")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help="只跑指定算法（按名字，大小写不敏感），如 --algorithms AutoEncoder。"
                             "不传则跑全部 15 个通用算法。")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="把本次运行结果隔离到 <output-dir>/runs/exp2_<modality_tag>_<UTC>/ 子目录，"
             "避免和历史 CSV 叠加。会在该目录下生成 exp2_results.csv 与 artifacts/。",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="搭配 --timestamped 使用，给本次运行打额外标签，目录名变成 "
             "exp2_<modality_tag>_<run-tag>_<UTC>/。无 --timestamped 时此参数被忽略。",
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
        run_dir = Path(OUTPUT_DIR) / "runs" / f"exp2_{modality_tag}{tag_part}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR = str(run_dir)
        print(f"[exp2] timestamped run dir → {OUTPUT_DIR}")

    LOG_PATH = str(Path(args.log_path) if args.log_path else Path(OUTPUT_DIR) / "exp2_results.csv")

    algos = _get_common_algos()
    if args.algorithms:
        wanted = {a.lower() for a in args.algorithms}
        algos = [t for t in algos if t[0].lower() in wanted]
        if not algos:
            avail = ", ".join(t[0] for t in _get_common_algos())
            raise SystemExit(
                f"[exp2] --algorithms {args.algorithms} 未匹配到任何算法。可选: {avail}"
            )
        print(f"[exp2] 只跑指定算法: {[t[0] for t in algos]}")
    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    for modality in args.modalities:
        datasets = args.datasets or DEFAULT_DATASETS[modality]
        print(f"\n{'=' * 70}\nExp-2 {modality}: {len(datasets)} datasets\n{'=' * 70}")
        for ds in datasets:
            print(f"\n  --- {modality}/{ds} ---")
            try:
                bundle, (X_tr, X_te, y_tr, y_te) = _load_as_2d(modality, ds, args.data_root)
                print(
                    f"  Loaded: train={np.asarray(X_tr).shape}, test={np.asarray(X_te).shape}, "
                    f"train_anomaly_rate={np.mean(y_tr):.4f}, test_anomaly_rate={np.mean(y_te):.4f}"
                )
            except Exception as e:
                print(f"  [SKIP] Cannot load {modality}/{ds}: {e!r}")
                continue

            for rate in CONTAMINATION_RATES:
                print(f"  --- target contamination/flip rate={rate:.2f} ---")
                for name, cls, kwargs, needs_y in algos:
                    n_total += 1
                    ok = _run_one(
                        modality=bundle.modality if modality != "graph" else "graph",
                        dataset_name=ds,
                        algo_name=name,
                        cls=cls,
                        kwargs=kwargs,
                        needs_y=needs_y,
                        rate=rate,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_test=X_te,
                        y_test=y_te,
                    )
                    n_ok += int(ok)

    total_t = time.perf_counter() - total_t0
    print(f"\n{'=' * 70}")
    print(f"Exp-2 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"Artifacts: {Path(OUTPUT_DIR) / 'artifacts' / 'exp2'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
