"""eval/logger.py 测试。"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from eval.logger import HEADER, log_experiment


def _read_rows(path: str) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


def test_first_write_creates_header(tmp_path):
    log = tmp_path / "log.csv"
    log_experiment(
        "Cardio", "IQR", 0.85, 0.7, 0.65, 0.1, 0.01, 42,
        log_path=str(log),
    )
    rows = _read_rows(str(log))
    assert rows[0] == HEADER
    assert len(rows) == 2  # header + 1 data row


def test_append_does_not_duplicate_header(tmp_path):
    log = tmp_path / "log.csv"
    for i in range(3):
        log_experiment(
            "Cardio", f"alg{i}", 0.5, 0.5, 0.5, 0.0, 0.0, 42,
            log_path=str(log),
        )
    rows = _read_rows(str(log))
    assert rows[0] == HEADER
    assert len(rows) == 4  # header + 3 data rows


def test_creates_parent_directory(tmp_path):
    log = tmp_path / "deep" / "nested" / "log.csv"
    log_experiment(
        "X", "Y", 0.5, 0.5, 0.5, 0.0, 0.0, 0,
        log_path=str(log),
    )
    assert log.exists()


def test_none_and_nan_become_nan_string(tmp_path):
    log = tmp_path / "log.csv"
    log_experiment(
        "Cardio", "Failed", None, float("nan"), float("inf"),
        None, None, None,
        notes="oops",
        log_path=str(log),
    )
    rows = _read_rows(str(log))
    data = rows[1]
    assert data[2] == "nan"  # auc_roc
    assert data[3] == "nan"  # auc_pr
    assert data[4] == "nan"  # f1_best
    assert data[5] == "nan"  # fit_time
    assert data[6] == "nan"  # predict_time
    assert data[7] == ""     # seed (None)


def test_notes_strips_newlines(tmp_path):
    log = tmp_path / "log.csv"
    log_experiment(
        "X", "Y", 0.5, 0.5, 0.5, 0.0, 0.0, 0,
        notes="line1\nline2\rline3",
        log_path=str(log),
    )
    rows = _read_rows(str(log))
    notes = rows[1][9]
    assert "\n" not in notes
    assert "\r" not in notes


def test_timestamp_iso8601_utc(tmp_path):
    log = tmp_path / "log.csv"
    log_experiment(
        "X", "Y", 0.5, 0.5, 0.5, 0.0, 0.0, 0,
        log_path=str(log),
    )
    rows = _read_rows(str(log))
    ts = rows[1][8]
    # 形如 "2026-05-26T14:30:00Z"
    assert len(ts) == 20
    assert ts[10] == "T"
    assert ts.endswith("Z")
