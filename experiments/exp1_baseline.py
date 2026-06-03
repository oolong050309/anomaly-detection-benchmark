"""Exp-1：基准对比实验。

26 个算法 × 29 个数据集，默认超参，记录 AUC-ROC / AUC-PR / F1@best / 耗时。
通过 ``adapters.load_dataset`` 加载数据，由成员 A 的统一适配器负责所有预处理。

结果写入 ``results/exp1_results.csv``。

用法：
    python -m experiments.exp1_baseline
    python -m experiments.exp1_baseline --modality tabular
    python -m experiments.exp1_baseline --modality timeseries
    python -m experiments.exp1_baseline --modality graph
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
from models._param_count import count_parameters as _count_parameters_helper


# ---------------------------------------------------------------------------
# 模型参数量统计（不侵入 eval/logger.py：参数量塞进 fit_params JSON 的 _num_params 键）
# ---------------------------------------------------------------------------


def _count_model_params(detector, fit_params: dict | None = None) -> int | None:
    """提取 detector 的真实参数量；推断不出来返回 None。

    委托给 ``models._param_count.count_parameters``，覆盖：
    - PyTorch 模型（AutoEncoder / DeepSVDD / LSTM-AE/Sup / DADA / GNN / TabPFN 内核）
    - sklearn 线性 / MLP / 树集成
    - XGBoost / LightGBM
    - PyOD OCSVM / IForest（解包到 ``detector_``）
    - MiniRocket（按 ``num_kernels`` 反推）
    - IQR / ECOD / COPOD / KNN / LOF / MatrixProfile → 无可训练参数，返回 0
    - UNPrompt → 不缓存模型，返回 None

    None 语义：参数量推断不出来；下游画图时跳过即可。
    """

    return _count_parameters_helper(detector, fit_params=fit_params)


def _augment_kwargs_with_num_params(kwargs: dict, detector) -> dict:
    """把 _num_params 注入 fit_params dict 的副本，原 kwargs 不变。"""
    n = _count_model_params(detector, fit_params=kwargs)
    out = dict(kwargs)
    out["_num_params"] = int(n) if n is not None else None
    return out


# ---------------------------------------------------------------------------
# Bug 1 修复：从 test 集借异常 / 正常窗口给 supervised 训练集
# ---------------------------------------------------------------------------


def _borrow_for_supervised(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    min_anom_train: int = 5,
    min_norm_train: int = 5,
    rng_seed: int = 42,
):
    """从 test 借异常 + 正常窗口给 train，避免 supervised 单类 fit 失败。

    Bug 1 修复（Requirement 1）。被借走的窗口同步从 test 移除，防泄漏
    （Property 1）；返回的 ``note`` 字符串含 ``"borrowed N anomaly windows
    from test set"`` 标识用于写入 CSV ``notes`` 列（Property 2）。

    参数
    ----
    X_tr, y_tr : 训练窗口和标签
    X_te, y_te : 测试窗口和标签
    min_anom_train : 训练集至少应含的异常窗口数
    min_norm_train : 训练集至少应含的正常窗口数
    rng_seed : 抽样随机种子（与 ``SEED`` 对齐保证可复现）

    返回
    ----
    (X_tr_aug, y_tr_aug, X_te_kept, y_te_kept, borrowed_n, note)
        ``borrowed_n`` 为借走的窗口总数（异常+正常），
        ``note`` 为人类可读的 borrow 摘要（无借走时为空字符串）。
    """
    rng = np.random.default_rng(rng_seed)

    y_tr = np.asarray(y_tr).astype(int)
    y_te = np.asarray(y_te).astype(int)

    # 训练集已经双类 → 啥也不做
    if y_tr.sum() >= 1 and (y_tr == 0).sum() >= 1:
        return X_tr, y_tr, X_te, y_te, 0, ""

    need_anom = max(0, min_anom_train - int(y_tr.sum()))
    need_norm = max(0, min_norm_train - int((y_tr == 0).sum()))

    test_anom_idx = np.flatnonzero(y_te == 1)
    test_norm_idx = np.flatnonzero(y_te == 0)

    take_anom = (
        rng.choice(test_anom_idx,
                   size=min(need_anom, len(test_anom_idx)),
                   replace=False)
        if need_anom and len(test_anom_idx) > 0
        else np.array([], dtype=int)
    )
    take_norm = (
        rng.choice(test_norm_idx,
                   size=min(need_norm, len(test_norm_idx)),
                   replace=False)
        if need_norm and len(test_norm_idx) > 0
        else np.array([], dtype=int)
    )

    take = np.concatenate([take_anom, take_norm]).astype(int)
    if take.size == 0:
        return X_tr, y_tr, X_te, y_te, 0, "borrow_failed_no_test_samples"

    X_tr_aug = np.concatenate([X_tr, X_te[take]], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_te[take]], axis=0)
    keep_mask = np.ones(len(y_te), dtype=bool)
    keep_mask[take] = False
    X_te_kept = X_te[keep_mask]
    y_te_kept = y_te[keep_mask]

    note = (
        f"borrowed {len(take_anom)} anomaly windows from test set"
        f" (+ {len(take_norm)} normal)"
    )
    return X_tr_aug, y_tr_aug, X_te_kept, y_te_kept, int(take.size), note


# ---------------------------------------------------------------------------
# 数据集清单（与 adapters/adbench_adapter.py 的命名一致）
# ---------------------------------------------------------------------------

SEED = 42
LOG_PATH = str(ROOT / "results" / "exp1_results.csv")
OUTPUT_DIR = str(ROOT / "results")

TABULAR_DATASETS = [
    "cardio", "thyroid", "satellite", "shuttle", "credit_card",
    "pima", "annthyroid", "mammography", "pendigits",
]
CV_DATASETS = ["cifar10_0", "cifar10_1", "fashionmnist_0", "fashionmnist_1"]
NLP_DATASETS = ["20news_0", "20news_1", "agnews_0", "amazon"]

# 时序数据集：传 CSV 文件名片段，adapter 会自动定位
TIMESERIES_DATASETS = [
    "006_NAB_id_6_Traffic",
    "149_Stock_id_1_Finance",
    "171_MITDB_id_2_Medical",
    "225_MGAB_id_1_Synthetic",
    "276_IOPS_id_17_WebService",
    "331_UCR_id_29_Facility",
    "337_UCR_id_35_HumanActivity",
    "550_SWaT_id_1_Sensor",
]

GRAPH_DATASETS = ["tfinance", "reddit", "amazon", "weibo"]


# ---------------------------------------------------------------------------
# 算法注册表
# ---------------------------------------------------------------------------


def _get_tabular_algos():
    """15 个表格算法（9 无监督 + 6 有监督）。"""
    from models import (
        AutoEncoderDetector, COPODDetector, DeepSVDDDetector,
        ECODDetector, IForestDetector, IQRDetector, KNNDetector,
        LightGBMDetector, LOFDetector, LogisticRegressionDetector,
        MLPDetector, OCSVMDetector, RandomForestDetector,
        TabPFNDetector, XGBoostDetector,
    )
    return [
        # (name, class, extra_kwargs, needs_y)
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


def _get_timeseries_algos():
    """4 个时序算法 + 可选 DADA。"""
    from models.timeseries import (
        LSTMAutoEncoderDetector, LSTMSupervisedDetector,
        MatrixProfileDetector, MiniRocketDetector,
    )
    algos = [
        ("MatrixProfile", MatrixProfileDetector, {"window_size": 100}, False),
        ("MiniRocket", MiniRocketDetector, {"num_kernels": 5000}, True),
        ("LSTM-AE", LSTMAutoEncoderDetector, {"epochs": 100, "hidden_size": 64}, False),
        ("LSTM-Sup", LSTMSupervisedDetector, {"epochs": 100, "hidden_size": 64}, True),
    ]
    try:
        from models.timeseries.dada import DADADetector
        algos.append(("DADA", DADADetector, {}, False))
    except ImportError:
        pass
    return algos


def _get_graph_algos():
    """图算法（取决于 DGL/pyg-lib/PyGOD 是否可用）。"""
    def missing_detector_class(algo_name, err):
        class _MissingGraphDetector:
            def __init__(self, *args, **kwargs):
                self.algo_name = algo_name

            def fit(self, *args, **kwargs):
                raise RuntimeError(f"{algo_name} unavailable: {err!r}")

        return _MissingGraphDetector

    algos = []
    try:
        from models.graph import CoLADetector, DOMINANTDetector
        algos.append(("DOMINANT", DOMINANTDetector, {"epoch": 100, "hid_dim": 64}, False))
        algos.append(("CoLA", CoLADetector, {"epoch": 100, "hid_dim": 64}, False))
    except ImportError as e:
        algos.append(("DOMINANT", missing_detector_class("DOMINANT", e), {}, False))
        algos.append(("CoLA", missing_detector_class("CoLA", e), {}, False))
    try:
        from models.graph.gnn_supervised import (
            BWGNNDetector, GCNDetector, XGBGraphDetector,
        )
        algos.append(("GCN", GCNDetector, {"epochs": 100}, True))
        algos.append(("BWGNN", BWGNNDetector, {"epochs": 100}, True))
        algos.append(("XGBGraph", XGBGraphDetector, {}, True))
    except ImportError as e:
        algos.append(("GCN", missing_detector_class("GCN", e), {}, True))
        algos.append(("BWGNN", missing_detector_class("BWGNN", e), {}, True))
        algos.append(("XGBGraph", missing_detector_class("XGBGraph", e), {}, True))
    try:
        from models.graph.unprompt import UNPromptDetector
        algos.append(
            ("UNPrompt", UNPromptDetector,
             {"pretrain_epochs": 100, "prompt_epochs": 100}, False)
        )
    except ImportError as e:
        algos.append(("UNPrompt", missing_detector_class("UNPrompt", e), {}, False))
    return algos


# ---------------------------------------------------------------------------
# 通用 runner
# ---------------------------------------------------------------------------


def _run_one(name, Cls, kwargs, needs_y, ds_name, modality, fit_input, predict_input,
             y_test, X_train=None, y_train=None, test_index=None, notes="") -> bool:
    """通用算法运行器：fit -> decision_function -> evaluate -> log。"""
    try:
        det = Cls(contamination=0.1, random_state=SEED, **kwargs)
        if needs_y:
            fit_args = fit_input
        else:
            fit_args = (fit_input if not isinstance(fit_input, tuple) else fit_input[0],)
        scores, fit_t, pred_t = fit_and_score(det, fit_args, predict_input)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = record_success(
                log_path=LOG_PATH,
                output_dir=OUTPUT_DIR,
                experiment_name="exp1",
                modality=modality,
                dataset_name=ds_name,
                algorithm_name=name,
                y_true=y_test,
                scores=scores,
                fit_time_sec=fit_t,
                predict_time_sec=pred_t,
                seed=SEED,
                fit_params=_augment_kwargs_with_num_params(kwargs, det),
                notes=notes,
                X_train=X_train,
                y_train=y_train,
                test_index=test_index,
            )
        print(f"  [{name:>14s}] AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f} "
              f"F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
        return True
    except Exception as e:
        msg = f"FAILED: {e!r}"[:200]
        record_failure(
            log_path=LOG_PATH,
            experiment_name="exp1",
            modality=modality,
            dataset_name=ds_name,
            algorithm_name=name,
            seed=SEED,
            error=e,
            fit_params=kwargs,
            notes=msg,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test,
        )
        print(f"  [{name:>14s}] {msg[:80]}")
        return False


# ---------------------------------------------------------------------------
# 各模态实验主逻辑
# ---------------------------------------------------------------------------


def run_tabular(datasets, algos, data_root=None):
    print(f"\n{'='*70}")
    print(f"Exp-1 Tabular/CV/NLP: {len(algos)} algos × {len(datasets)} datasets")
    print(f"{'='*70}")
    n_ok, n_total = 0, 0
    for ds in datasets:
        print(f"\n--- Dataset: {ds} ---")
        try:
            bundle = load_dataset(ds, data_root=data_root)
            X_tr, X_te, y_tr, y_te = bundle.as_tuple()
            print(f"  Loaded: train={X_tr.shape}, test={X_te.shape}, "
                  f"anomaly_rate={y_te.mean():.4f}")
        except Exception as e:
            print(f"  [SKIP] Cannot load {ds}: {e}")
            continue
        for name, Cls, kwargs, needs_y in algos:
            n_total += 1
            fit_input = (X_tr, y_tr) if needs_y else X_tr
            if _run_one(
                name, Cls, kwargs, needs_y, ds, bundle.modality,
                fit_input, X_te, y_te, X_train=X_tr, y_train=y_tr,
            ):
                n_ok += 1
    return n_ok, n_total


def run_timeseries(datasets, algos, data_root=None):
    print(f"\n{'='*70}")
    print(f"Exp-1 Timeseries: {len(algos)} algos × {len(datasets)} datasets")
    print(f"{'='*70}")
    n_ok, n_total = 0, 0
    for ds in datasets:
        print(f"\n--- Dataset: {ds} ---")
        try:
            bundle = load_dataset(ds, modality="timeseries",
                                  data_root=data_root,
                                  window_size=100, stride=10)
            X_tr, X_te, y_tr, y_te = bundle.as_tuple()
            raw_values = bundle.extras["raw_values"]
            train_end = bundle.extras["train_end"]
            train_seq = raw_values[:train_end]
            test_seq = raw_values[train_end:]
            print(f"  Loaded: train_windows={X_tr.shape}, test_windows={X_te.shape}, "
                  f"test_anomaly_rate={y_te.mean():.4f}")
        except Exception as e:
            print(f"  [SKIP] Cannot load {ds}: {e}")
            continue

        for name, Cls, kwargs, needs_y in algos:
            n_total += 1

            # MatrixProfile: 用原始一维序列，再映射回窗口
            if name == "MatrixProfile":
                try:
                    det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                    t0 = time.perf_counter()
                    det.fit(train_seq)
                    fit_t = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    scores_seq = det.decision_function(test_seq)
                    pred_t = time.perf_counter() - t0
                    n_w = X_te.shape[0]
                    starts = np.clip(np.arange(n_w) * 10, 0, scores_seq.size - 1)
                    scores = scores_seq[starts]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = record_success(
                            log_path=LOG_PATH,
                            output_dir=OUTPUT_DIR,
                            experiment_name="exp1",
                            modality="timeseries",
                            dataset_name=ds,
                            algorithm_name=name,
                            y_true=y_te,
                            scores=scores,
                            fit_time_sec=fit_t,
                            predict_time_sec=pred_t,
                            seed=SEED,
                            fit_params=_augment_kwargs_with_num_params(kwargs, det),
                            X_train=train_seq,
                            y_train=y_tr,
                            extra_metadata={"score_alignment": "raw_sequence_start_to_window"},
                        )
                    print(f"  [{name:>14s}] AUC-ROC={m['auc_roc']:.4f} "
                          f"AUC-PR={m['auc_pr']:.4f} F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
                    n_ok += 1
                except Exception as e:
                    msg = f"FAILED: {e!r}"[:200]
                    record_failure(
                        log_path=LOG_PATH,
                        experiment_name="exp1",
                        modality="timeseries",
                        dataset_name=ds,
                        algorithm_name=name,
                        seed=SEED,
                        error=e,
                        fit_params=kwargs,
                        notes=msg,
                        X_train=train_seq,
                        y_train=y_tr,
                        y_test=y_te,
                    )
                    print(f"  [{name:>14s}] {msg[:80]}")
                continue

            # 无监督算法只用正常窗口训练
            if not needs_y:
                normal_mask = y_tr == 0
                X_tr_use = X_tr[normal_mask] if normal_mask.sum() >= 5 else X_tr
                fit_input = X_tr_use
                fit_X_tr = X_tr_use
                fit_y_tr = y_tr
                predict_input = X_te
                y_te_use = y_te
                run_notes = ""
            else:
                # Bug 1 修复（Requirement 1）：训练集单类时从 test 借窗口
                X_tr_aug, y_tr_aug, X_te_kept, y_te_kept, borrowed_n, borrow_note = (
                    _borrow_for_supervised(
                        X_tr, y_tr, X_te, y_te, rng_seed=SEED,
                    )
                )
                still_single_class = (
                    y_tr_aug.sum() == 0 or (y_tr_aug == 1).all()
                )
                if borrowed_n == 0 and still_single_class:
                    err = RuntimeError("borrow_failed_no_test_anomalies")
                    record_failure(
                        log_path=LOG_PATH,
                        experiment_name="exp1",
                        modality="timeseries",
                        dataset_name=ds,
                        algorithm_name=name,
                        seed=SEED,
                        error=err,
                        fit_params=kwargs,
                        notes="borrow_failed_no_test_anomalies",
                        X_train=X_tr,
                        y_train=y_tr,
                        y_test=y_te,
                    )
                    print(f"  [{name:>14s}] SKIP: borrow_failed_no_test_anomalies")
                    continue
                fit_input = (X_tr_aug, y_tr_aug)
                fit_X_tr = X_tr_aug
                fit_y_tr = y_tr_aug
                predict_input = X_te_kept
                y_te_use = y_te_kept
                run_notes = borrow_note

            if _run_one(
                name, Cls, kwargs, needs_y, ds, "timeseries",
                fit_input, predict_input, y_te_use,
                X_train=fit_X_tr, y_train=fit_y_tr, notes=run_notes,
            ):
                n_ok += 1
    return n_ok, n_total


def run_graph(datasets, algos, data_root=None):
    print(f"\n{'='*70}")
    print(f"Exp-1 Graph: {len(algos)} algos × {len(datasets)} datasets")
    print(f"{'='*70}")
    n_ok, n_total = 0, 0
    for ds in datasets:
        print(f"\n--- Dataset: {ds} ---")
        try:
            bundle = load_dataset(ds, modality="graph", data_root=data_root)
            graph = bundle.extras["graph"]
            masks = bundle.extras["masks"]
            y_all = bundle.extras["y_all"]
            test_idx = np.flatnonzero(masks["test_mask"])
            y_te = y_all[test_idx]
            print(f"  Loaded: nodes={graph.num_nodes()}, "
                  f"edges={graph.num_edges()}, test_nodes={len(test_idx)}, "
                  f"test_anomaly_rate={y_te.mean():.4f}")
        except Exception as e:
            print(f"  [SKIP] Cannot load {ds}: {e}")
            continue

        for name, Cls, kwargs, needs_y in algos:
            n_total += 1
            try:
                det = Cls(contamination=0.1, random_state=SEED, **kwargs)
                t0 = time.perf_counter()
                det.fit(graph)
                fit_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                scores_all = det.decision_function(graph)
                pred_t = time.perf_counter() - t0

                # 只在测试节点上评估
                scores_test = np.asarray(scores_all)[test_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = record_success(
                        log_path=LOG_PATH,
                        output_dir=OUTPUT_DIR,
                        experiment_name="exp1",
                        modality="graph",
                        dataset_name=ds,
                        algorithm_name=name,
                        y_true=y_te,
                        scores=scores_test,
                        fit_time_sec=fit_t,
                        predict_time_sec=pred_t,
                        seed=SEED,
                        fit_params=_augment_kwargs_with_num_params(kwargs, det),
                        X_train=bundle.X_train,
                        y_train=bundle.y_train,
                        test_index=test_idx,
                    )
                print(f"  [{name:>14s}] AUC-ROC={m['auc_roc']:.4f} "
                      f"AUC-PR={m['auc_pr']:.4f} F1={m['f1_best']:.4f} fit={fit_t:.2f}s")
                n_ok += 1
            except Exception as e:
                msg = f"FAILED: {e!r}"[:200]
                record_failure(
                    log_path=LOG_PATH,
                    experiment_name="exp1",
                    modality="graph",
                    dataset_name=ds,
                    algorithm_name=name,
                    seed=SEED,
                    error=e,
                    fit_params=kwargs,
                    notes=msg,
                    X_train=bundle.X_train if "bundle" in locals() else None,
                    y_train=bundle.y_train if "bundle" in locals() else None,
                    y_test=y_te if "y_te" in locals() else None,
                )
                print(f"  [{name:>14s}] {msg[:80]}")
    return n_ok, n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global SEED, LOG_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Exp-1: Baseline comparison")
    parser.add_argument("--modality",
                        choices=["tabular", "timeseries", "graph", "all"],
                        default="all", help="Which modality to run")
    parser.add_argument("--data-root", default=None,
                        help="Data root on the server. Defaults to AD_DATA_ROOT or ./data.")
    parser.add_argument("--output-dir", default=str(ROOT / "results"),
                        help="Directory for CSV logs and score artifacts.")
    parser.add_argument("--log-path", default=None,
                        help="CSV path. Defaults to <output-dir>/exp1_results.csv.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="把本次运行结果隔离到 <output-dir>/runs/<modality>_<UTC>/ 子目录，"
             "避免和历史 CSV 叠加。会在该目录下生成 exp1_results.csv 与 artifacts/。",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="搭配 --timestamped 使用，给本次运行打额外标签，目录名变成 "
             "<modality>_<run-tag>_<UTC>/。无 --timestamped 时此参数被忽略。",
    )
    parser.add_argument(
        "--algorithms",
        default=None,
        help="只跑指定算法，逗号分隔（如 'MiniRocket,LSTM-Sup'）。"
             "缺省跑模态全部算法。匹配大小写不敏感。",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="只跑指定数据集，逗号分隔（如 '006_NAB_id_6_Traffic,149_Stock_id_1_Finance'）。"
             "缺省跑模态全部数据集。",
    )
    args = parser.parse_args()

    # --algorithms / --datasets 解析为 set / list
    algo_filter = None
    if args.algorithms:
        algo_filter = {a.strip().lower() for a in args.algorithms.split(",") if a.strip()}
    dataset_filter = None
    if args.datasets:
        dataset_filter = [d.strip() for d in args.datasets.split(",") if d.strip()]

    def _filter_algos(algos):
        if algo_filter is None:
            return algos
        kept = [a for a in algos if a[0].lower() in algo_filter]
        if not kept:
            print(f"[WARN] --algorithms={args.algorithms} 没匹配到任何算法；本模态跳过。")
        return kept

    SEED = int(args.seed)
    OUTPUT_DIR = str(Path(args.output_dir))

    # --timestamped: 把本次运行隔到独立目录，不污染历史 CSV
    if args.timestamped:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag_part = f"_{args.run_tag}" if args.run_tag else ""
        run_dir = Path(OUTPUT_DIR) / "runs" / f"{args.modality}{tag_part}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR = str(run_dir)
        print(f"[exp1] timestamped run dir → {OUTPUT_DIR}")

    LOG_PATH = str(Path(args.log_path) if args.log_path else Path(OUTPUT_DIR) / "exp1_results.csv")

    total_t0 = time.perf_counter()
    n_ok, n_total = 0, 0

    if args.modality in ("tabular", "all"):
        algos = _filter_algos(_get_tabular_algos())
        if algos:
            datasets = TABULAR_DATASETS + CV_DATASETS + NLP_DATASETS
            if dataset_filter:
                datasets = [d for d in datasets if d in dataset_filter]
            if datasets:
                ok, total = run_tabular(datasets, algos, data_root=args.data_root)
                n_ok += ok
                n_total += total

    if args.modality in ("timeseries", "all"):
        algos = _filter_algos(_get_timeseries_algos())
        if algos:
            datasets = TIMESERIES_DATASETS
            if dataset_filter:
                datasets = [d for d in datasets if d in dataset_filter]
            if datasets:
                ok, total = run_timeseries(datasets, algos, data_root=args.data_root)
                n_ok += ok
                n_total += total

    if args.modality in ("graph", "all"):
        algos = _filter_algos(_get_graph_algos())
        if algos:
            datasets = GRAPH_DATASETS
            if dataset_filter:
                datasets = [d for d in datasets if d in dataset_filter]
            if datasets:
                ok, total = run_graph(datasets, algos, data_root=args.data_root)
                n_ok += ok
                n_total += total
        else:
            print("\n[INFO] No graph algorithms available "
                  "(DGL/pyg-lib/PyGOD not installed)")

    total_t = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Exp-1 Complete: {n_ok}/{n_total} succeeded in {total_t:.1f}s")
    print(f"Results: {LOG_PATH}")
    print(f"Artifacts: {Path(OUTPUT_DIR) / 'artifacts' / 'exp1'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
