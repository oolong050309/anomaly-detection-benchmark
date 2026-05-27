"""任务 11：DOMINANT + CoLA 在小图上的快速集成测试。

PyGOD 必须依赖 torch_geometric，若未安装则跳过整个文件。
另外 PyGOD 内部需要 ``pyg-lib`` 或 ``torch-sparse`` 做邻居采样；
本地 Windows 安装这两个包较麻烦，因此运行时若底层缺依赖，则跳过具体训练测试。
"""

from __future__ import annotations

import numpy as np
import pytest

torch_geometric = pytest.importorskip("torch_geometric")
pygod = pytest.importorskip("pygod")
torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from models.graph.cola import CoLADetector
from models.graph.dominant import DOMINANTDetector


# 检测是否可用 pyg-lib / torch-sparse；若不可用则跳过实训练测试
try:
    import pyg_lib  # noqa: F401
    _HAS_SPARSE = True
except ImportError:
    try:
        import torch_sparse  # noqa: F401
        _HAS_SPARSE = True
    except ImportError:
        _HAS_SPARSE = False


def _make_random_graph(
    n: int = 100, d: int = 16, edges: int = 400, seed: int = 0
) -> Data:
    rng = np.random.RandomState(seed)
    x = torch.from_numpy(rng.randn(n, d).astype(np.float32))
    src = torch.from_numpy(rng.randint(0, n, size=edges).astype(np.int64))
    dst = torch.from_numpy(rng.randint(0, n, size=edges).astype(np.int64))
    edge_index = torch.stack([src, dst], dim=0)
    y = torch.from_numpy((rng.rand(n) > 0.85).astype(np.int64))
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.mark.skipif(
    not _HAS_SPARSE,
    reason="PyGOD 需要 pyg-lib 或 torch-sparse 才能采样；本机未安装。",
)
def test_dominant_basic():
    g = _make_random_graph()
    det = DOMINANTDetector(hid_dim=16, num_layers=2, epoch=3, random_state=42)
    det.fit(g)
    scores = det.decision_function(g)
    assert scores.shape == (g.x.shape[0],)
    assert np.isfinite(scores).all()


@pytest.mark.skipif(
    not _HAS_SPARSE,
    reason="PyGOD 需要 pyg-lib 或 torch-sparse 才能采样；本机未安装。",
)
def test_cola_basic():
    g = _make_random_graph()
    det = CoLADetector(hid_dim=16, num_layers=2, epoch=3, random_state=42)
    det.fit(g)
    scores = det.decision_function(g)
    assert scores.shape == (g.x.shape[0],)
    assert np.isfinite(scores).all()


def test_graph_rejects_missing_edge_index():
    """图缺少 edge_index 时，基类 _validate_graph 应抛 ValueError。"""
    x = torch.randn(10, 4)
    g = Data(x=x)
    g.edge_index = None
    det = DOMINANTDetector(epoch=3, random_state=42)
    with pytest.raises(ValueError):
        det.fit(g)


def test_graph_rejects_unsupported_type():
    """非图对象应被拒绝。"""
    det = DOMINANTDetector(epoch=3, random_state=42)
    with pytest.raises(ValueError):
        det.fit("not a graph")
