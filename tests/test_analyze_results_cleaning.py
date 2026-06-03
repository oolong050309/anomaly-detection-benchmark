"""Tests for ``scripts.analyze_results.clean_results``.

清洗逻辑是在 spec 修复 Bug 1/2/3 之后追加的，用来处理多次跑后 CSV 累积的
质量问题：

- 旧 schema 行（``experiment_name`` 空、``modality`` / ``status`` 都缺失），
- ``status='failed'`` 行（如 TabPFN n>10000 / d>100 触发的尺寸限制），
- ``6_cardio`` ↔ ``cardio`` 这种带数字前缀的同义 dataset 名，
- 同一 ``(dataset, algorithm, modality)`` 多次跑出多行。

时序 dataset 名（``006_NAB_id_6_Traffic`` 等）没有同名「裸版」时不应被改动。
"""

from __future__ import annotations

import pandas as pd

from scripts.analyze_results import (
    _build_dataset_name_map,
    annotate_extreme_imbalance,
    clean_results,
    summarize_by_group,
)


def _make_row(**overrides):
    base = {
        "dataset_name": "cardio",
        "algorithm_name": "IForest",
        "auc_roc": 0.9,
        "auc_pr": 0.6,
        "f1_best": 0.5,
        "fit_time_sec": 0.1,
        "predict_time_sec": 0.01,
        "seed": 42,
        "timestamp": "2026-05-27T16:14:13Z",
        "notes": "",
        "experiment_name": "exp1",
        "modality": "tabular",
        "status": "success",
        "best_threshold": 0.0,
        "contamination_mode": "",
        "contamination_rate": float("nan"),
        "label_flip_rate": float("nan"),
        "n_train": 100,
        "n_test": 50,
        "train_anomaly_rate": 0.1,
        "test_anomaly_rate": 0.1,
        "fit_params": "{}",
        "artifact_path": "",
        "error_type": "",
        "error_message": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# clean_results 各步骤
# ---------------------------------------------------------------------------


def test_clean_drops_legacy_schema_rows() -> None:
    legacy = _make_row(experiment_name="", status="", modality="")
    fresh = _make_row()
    df = pd.DataFrame([legacy, fresh])

    cleaned = clean_results(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["experiment_name"] == "exp1"


def test_clean_drops_failed_rows_by_default() -> None:
    ok = _make_row()
    failed = _make_row(
        algorithm_name="TabPFN",
        auc_roc=float("nan"),
        status="failed",
        error_type="RuntimeError",
        error_message="TabPFN supports n_samples<=10000",
    )
    df = pd.DataFrame([ok, failed])

    cleaned = clean_results(df)

    assert (cleaned["status"] == "success").all()
    assert "TabPFN" not in set(cleaned["algorithm_name"])


def test_clean_keep_failed_keeps_failed_rows() -> None:
    df = pd.DataFrame(
        [
            _make_row(),
            _make_row(algorithm_name="TabPFN", status="failed"),
        ]
    )

    cleaned = clean_results(df, drop_failed=False)

    assert len(cleaned) == 2


def test_clean_collapses_prefixed_dataset_names_when_collision() -> None:
    df = pd.DataFrame(
        [
            _make_row(dataset_name="6_cardio", timestamp="2026-05-26T12:31:56Z"),
            _make_row(dataset_name="cardio", timestamp="2026-05-27T16:14:13Z"),
        ]
    )

    cleaned = clean_results(df)

    assert set(cleaned["dataset_name"]) == {"cardio"}
    # 折叠 + 去重 → 仅 timestamp 较新的一行存活
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["timestamp"] == "2026-05-27T16:14:13Z"


def test_clean_keeps_prefixed_name_without_collision() -> None:
    """时序的 ``006_NAB_id_6_Traffic`` 等没有「裸版」同名，应保留前缀。"""

    df = pd.DataFrame(
        [
            _make_row(
                dataset_name="006_NAB_id_6_Traffic",
                modality="timeseries",
                algorithm_name="MatrixProfile",
            ),
            _make_row(
                dataset_name="cardio",
                modality="tabular",
                algorithm_name="IForest",
            ),
        ]
    )

    cleaned = clean_results(df)

    assert "006_NAB_id_6_Traffic" in set(cleaned["dataset_name"])


def test_clean_dedups_by_dataset_algorithm_modality_keep_latest() -> None:
    df = pd.DataFrame(
        [
            _make_row(timestamp="2026-05-27T02:00:00Z", auc_roc=0.7),
            _make_row(timestamp="2026-05-27T16:00:00Z", auc_roc=0.9),
            _make_row(timestamp="2026-05-26T12:00:00Z", auc_roc=0.5),
        ]
    )

    cleaned = clean_results(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["auc_roc"] == 0.9


def test_clean_adds_algorithm_alias_for_summarize() -> None:
    """logger 写的是 ``algorithm_name``，summarize_by_group 期望 ``algorithm``。"""

    df = pd.DataFrame([_make_row(algorithm_name="LR")])
    cleaned = clean_results(df)
    assert "algorithm" in cleaned.columns
    assert cleaned.loc[0, "algorithm"] == "LR"
    # 原列不动
    assert cleaned.loc[0, "algorithm_name"] == "LR"


def test_clean_is_pure() -> None:
    df = pd.DataFrame(
        [
            _make_row(),
            _make_row(experiment_name=""),
            _make_row(status="failed", auc_roc=float("nan")),
        ]
    )
    snapshot = df.copy(deep=True)

    _ = clean_results(df)

    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# _build_dataset_name_map 单测
# ---------------------------------------------------------------------------


def test_build_dataset_name_map_no_collision_returns_empty() -> None:
    names = pd.Series(
        ["006_NAB_id_6_Traffic", "149_Stock_id_1_Finance", "550_SWaT_id_1_Sensor"]
    )
    assert _build_dataset_name_map(names) == {}


def test_build_dataset_name_map_collision_returns_mapping() -> None:
    names = pd.Series(["6_cardio", "cardio", "38_thyroid", "thyroid"])
    mapping = _build_dataset_name_map(names)
    assert mapping == {"6_cardio": "cardio", "38_thyroid": "thyroid"}


# ---------------------------------------------------------------------------
# 端到端：清洗 + annotate + summarize 不应被 TabPFN failed 行污染
# ---------------------------------------------------------------------------


def test_clean_then_summarize_excludes_failed_tabpfn() -> None:
    df = pd.DataFrame(
        [
            _make_row(algorithm_name="LR", auc_roc=0.9, auc_pr=0.8, f1_best=0.7),
            _make_row(
                algorithm_name="TabPFN",
                status="failed",
                auc_roc=float("nan"),
                auc_pr=float("nan"),
                f1_best=float("nan"),
            ),
        ]
    )

    cleaned = clean_results(df)
    annotated = annotate_extreme_imbalance(cleaned)
    summaries = summarize_by_group(annotated)

    algos = set(summaries["all"].index)
    assert "LR" in algos
    assert "TabPFN" not in algos
