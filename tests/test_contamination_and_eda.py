"""数据污染与 EDA 产物的本地测试。

这些测试不依赖服务器原始数据，适合在本地开发机和 CI 中快速运行。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.contaminate import contaminate_supervised, contaminate_unsupervised


def test_unsupervised_zero_contamination_keeps_only_normal_samples() -> None:
    """无监督污染率为 0 时，训练集应只保留正常样本。"""

    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    Xc, yc, meta = contaminate_unsupervised(X, y, contamination_rate=0.0, seed=42)

    assert Xc.shape[0] == 6
    assert np.all(yc == 0)
    assert meta["n_anomalies_output"] == 0


def test_unsupervised_contamination_rate_is_close_to_target() -> None:
    """无监督污染后，异常比例应接近目标污染率。"""

    X = np.arange(240).reshape(120, 2)
    y = np.array([0] * 100 + [1] * 20)

    _, yc, meta = contaminate_unsupervised(X, y, contamination_rate=0.1, seed=42)
    observed = float(np.mean(yc == 1))

    assert abs(observed - 0.1) < 0.02
    assert meta["n_anomalies_output"] == int(np.sum(yc == 1))


def test_supervised_flip_count_and_reproducibility() -> None:
    """有监督标签翻转数量应准确，且同 seed 可复现。"""

    X = np.arange(200).reshape(100, 2)
    y = np.array([0] * 80 + [1] * 20)

    _, y1, meta1 = contaminate_supervised(X, y, flip_rate=0.1, seed=42)
    _, y2, meta2 = contaminate_supervised(X, y, flip_rate=0.1, seed=42)

    assert meta1["n_flipped"] == 10
    assert meta2["n_flipped"] == 10
    assert np.array_equal(y1, y2)
    assert int(np.sum(y1 != y)) == 10


def test_eda_summary_contract() -> None:
    """EDA JSON 应能被前端和报告稳定读取。"""

    path = Path("data/eda_summary.json")
    rows = json.loads(path.read_text(encoding="utf-8"))

    assert len(rows) == 30
    assert "missing_or_error" not in {row.get("status") for row in rows}
    for row in rows:
        assert row.get("name")
        assert row.get("source")
        assert row.get("modality")
        assert row.get("status")

    graph_rows = {row["name"]: row for row in rows if row.get("modality") == "graph"}
    assert set(graph_rows) == {"tfinance", "reddit", "amazon", "weibo"}
    assert graph_rows["amazon"]["split"] == "predefined_masks"
    assert graph_rows["reddit"]["split"] == "generated_stratified_node_masks"
