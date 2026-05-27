"""GADBench 包装：GCN / BWGNN / XGBGraph。

底层实现来自 vendor 包 ``models/graph/_vendor/gadbench/``。

接口契约：
- 输入：DGL ``DGLGraph``（含 ``ndata['feature']``、``ndata['label']`` 和
  ``ndata['train_mask']``、``ndata['val_mask']``、``ndata['test_mask']``）。
- ``fit(graph, y=None)``：调用 GADBench 的训练逻辑，使用图中预设的 train_mask。
- ``decision_function(graph) -> np.ndarray``：返回所有节点的 class=1 概率，
  长度等于图节点数。

注意：GADBench 用 DGL 而非 PyG。我们的 ``GraphDetector`` 默认基类预期 PyG Data，
因此本模块重写了 ``fit`` 和 ``decision_function``，跳过基类的 PyG 校验，
直接转交给 GADBench 的训练逻辑。
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from models.base import GraphDetector, SupervisedDetector
from models.device import get_preferred_device


# ---------------------------------------------------------------------------
# Helper: 把图对象转换成 GADBench 期望的格式
# ---------------------------------------------------------------------------


def _ensure_dgl_graph(graph: Any):
    """把输入图统一为 DGL DGLGraph。

    支持三种输入：
    - DGL DGLGraph（直接返回）
    - PyG ``torch_geometric.data.Data``（转换为 DGL）
    - 已经预处理过、带 ``ndata['feature']`` 的图

    返回的图必须包含：``ndata['feature']``、``ndata['label']`` 以及
    ``train_mask`` / ``val_mask`` / ``test_mask``。
    """
    try:
        import dgl  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "[GADBench wrappers] DGL is required. Install via "
            "`pip install dgl` or follow the official guide for your CUDA."
        ) from e

    # DGL graph
    if hasattr(graph, "ndata") and hasattr(graph, "edges"):
        return graph

    # PyG Data -> DGL
    if hasattr(graph, "edge_index") and hasattr(graph, "x"):
        import dgl
        import torch

        ei = graph.edge_index
        n = int(graph.x.shape[0])
        g = dgl.graph((ei[0].long(), ei[1].long()), num_nodes=n)
        g.ndata["feature"] = graph.x.float()
        if hasattr(graph, "y") and graph.y is not None:
            g.ndata["label"] = graph.y.long()
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            if hasattr(graph, mask_name) and getattr(graph, mask_name) is not None:
                g.ndata[mask_name] = getattr(graph, mask_name).bool()
        return g

    raise ValueError(
        f"Unsupported graph type {type(graph).__name__}; "
        "expected dgl.DGLGraph or torch_geometric.data.Data"
    )


def _validate_gadbench_graph(graph) -> None:
    needed_node = ["feature", "label", "train_mask", "val_mask", "test_mask"]
    for key in needed_node:
        if key not in graph.ndata:
            raise ValueError(
                f"Graph is missing required ndata key '{key}'. "
                f"GADBench wrappers require: {needed_node}"
            )


def _fix_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass
    np.random.seed(int(seed))


def _select_device(prefer_cuda: bool = True) -> str:
    if not prefer_cuda:
        return "cpu"
    return get_preferred_device()


# ---------------------------------------------------------------------------
# 公共训练流程：用 GADBench 的 BaseGNNDetector / XGBGraphDetector 训练并取分数
# ---------------------------------------------------------------------------


def _build_gadbench_train_config(
    seed: int | None, epochs: int, patience: int, metric: str
) -> dict:
    return {
        "device": _select_device(),
        "epochs": int(epochs),
        "patience": int(patience),
        "metric": metric,
        "inductive": False,
        "seed": int(seed) if seed is not None else 42,
    }


def _run_gadbench_detector(detector_class, model_config, train_config, graph):
    """实例化 GADBench 的 detector 并训练，返回测试节点分数。

    GADBench 的 BaseDetector.train() 返回的是 score dict，
    我们要的是所有节点的概率分布。这里在训练完成后直接调用底层模型再 forward 一次。
    """
    import torch
    from models.graph._vendor.gadbench.detector import _DummyData
    # GADBench 的 BaseDetector 期望 data.graph，所以包一层
    data_wrapper = _DummyData(graph)
    det = detector_class(train_config, model_config, data_wrapper)
    test_score = det.train()
    # 训练完之后从 det.model 拿全图概率
    det.model.eval()
    with torch.no_grad():
        # GADBench 模型 forward(graph) 返回 logits
        logits = det.model(graph.to(train_config["device"]))
        probs = logits.softmax(1)[:, 1].detach().cpu().numpy()
    return probs, test_score


# ---------------------------------------------------------------------------
# GADBench 的 detector 期望 ``data.graph`` 而不是裸图，包一层
# ---------------------------------------------------------------------------

# 注意：GADBench detector 内部用 ``self.data.graph``，所以我们提供一个 _DummyData
# 把图包起来。这个类在 _vendor.gadbench.detector 里没有，需要本模块定义。


class _DummyData:
    """轻量 wrapper，让 GADBench 的 BaseDetector 能拿到 .graph 属性。"""

    def __init__(self, graph):
        self.graph = graph
        self.name = getattr(graph, "name", "user_provided")


# 把 _DummyData 注入回 vendor 模块的全局命名空间，
# 这样 _run_gadbench_detector 里的 from ... import _DummyData 才不至于失败
import models.graph._vendor.gadbench.detector as _gadbench_detector_mod  # noqa: E402

_gadbench_detector_mod._DummyData = _DummyData


# ---------------------------------------------------------------------------
# GCNDetector / BWGNNDetector
# ---------------------------------------------------------------------------


class _BaseGNNDetectorWrapper(GraphDetector, SupervisedDetector):
    """GCN / BWGNN 共享的训练逻辑（GADBench BaseGNNDetector 模板）。"""

    _MODEL_NAME: str = ""  # 子类覆盖，对应 GADBench param_space 里的 key

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        h_feats: int = 32,
        num_layers: int = 2,
        drop_rate: float = 0.0,
        lr: float = 0.01,
        epochs: int = 100,
        patience: int = 20,
        metric: str = "AUROC",
        **algo_kwargs: Any,
    ) -> None:
        # 直接调用 BaseDetector.__init__（绕过 SupervisedDetector 的额外校验）
        # 但我们仍依赖 SupervisedDetector.fit 来强制 y 检查
        super().__init__(contamination=contamination, random_state=random_state)
        self.h_feats = h_feats
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.metric = metric
        self._algo_kwargs = algo_kwargs
        self._scores: np.ndarray | None = None  # 全图节点分数

    # 重写 fit 以适配 GADBench 的 DGL 图（绕过基类的 PyG 校验）
    def fit(self, graph, **kwargs):  # type: ignore[override]
        from models.graph._vendor.gadbench.detector import BaseGNNDetector

        g = _ensure_dgl_graph(graph)
        _validate_gadbench_graph(g)
        _fix_torch_seed(self.random_state)

        train_config = _build_gadbench_train_config(
            self.random_state, self.epochs, self.patience, self.metric
        )
        model_config = {
            "model": self._MODEL_NAME,
            "h_feats": self.h_feats,
            "num_layers": self.num_layers,
            "drop_rate": self.drop_rate,
            "lr": self.lr,
            **self._algo_kwargs,
        }
        try:
            scores, _ = _run_gadbench_detector(
                BaseGNNDetector, model_config, train_config, g
            )
            self._scores = scores
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

        self.is_fitted_ = True
        return self

    def decision_function(self, graph) -> np.ndarray:  # type: ignore[override]
        self._check_fitted()
        if self._scores is None:
            raise RuntimeError(
                f"[{type(self).__name__}] scores not available; call fit() first"
            )
        return self._scores

    # GraphDetector 抽象钩子，本类自行管理训练，钩子留空
    def _fit(self, graph, y=None, **kwargs):
        pass

    def _decision_function(self, graph):  # pragma: no cover
        return self._scores


class GCNDetector(_BaseGNNDetectorWrapper):
    """GCN 图卷积网络（GADBench / Kipf & Welling 2017 风格）。"""

    _MODEL_NAME = "GCN"


class BWGNNDetector(_BaseGNNDetectorWrapper):
    """BWGNN 图谱方法（Beta Wavelet GNN，ICML 2022）。"""

    _MODEL_NAME = "BWGNN"

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        h_feats: int = 32,
        num_layers: int = 2,
        mlp_layers: int = 2,
        drop_rate: float = 0.0,
        lr: float = 0.01,
        epochs: int = 100,
        patience: int = 20,
        metric: str = "AUROC",
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(
            contamination=contamination,
            random_state=random_state,
            h_feats=h_feats,
            num_layers=num_layers,
            drop_rate=drop_rate,
            lr=lr,
            epochs=epochs,
            patience=patience,
            metric=metric,
            mlp_layers=mlp_layers,
            **algo_kwargs,
        )


# ---------------------------------------------------------------------------
# XGBGraphDetector
# ---------------------------------------------------------------------------


class XGBGraphDetector(GraphDetector, SupervisedDetector):
    """XGBoost + 图邻居聚合（GADBench NeurIPS 2023）。

    流程：先用无参数 GIN 做邻居聚合得到节点 embedding，
    再用 XGBoost 在 embedding 上做有监督分类。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        n_estimators: int = 100,
        num_layers: int = 2,
        agg: str = "mean",
        booster: str = "gbtree",
        eta: float = 0.1,
        subsample: float = 0.75,
        epochs: int = 0,        # XGBGraph 训练逻辑不用 epoch（对外保留参数一致性）
        patience: int = 20,
        metric: str = "AUROC",
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.n_estimators = n_estimators
        self.num_layers = num_layers
        self.agg = agg
        self.booster = booster
        self.eta = eta
        self.subsample = subsample
        self.epochs = epochs
        self.patience = patience
        self.metric = metric
        self._algo_kwargs = algo_kwargs
        self._scores: np.ndarray | None = None

    def fit(self, graph, **kwargs):  # type: ignore[override]
        from models.graph._vendor.gadbench.detector import XGBGraphDetector as _XGBGraph

        g = _ensure_dgl_graph(graph)
        _validate_gadbench_graph(g)
        _fix_torch_seed(self.random_state)

        train_config = _build_gadbench_train_config(
            self.random_state, self.epochs, self.patience, self.metric
        )
        model_config = {
            "model": "XGBGraph",
            "num_layers": self.num_layers,
            "agg": self.agg,
            "booster": self.booster,
            "n_estimators": self.n_estimators,
            "eta": self.eta,
            "subsample": self.subsample,
            **self._algo_kwargs,
        }
        try:
            data_wrapper = _DummyData(g)
            det = _XGBGraph(train_config, model_config, data_wrapper)
            det.train()
            # det.model 是 XGBClassifier；用所有节点的特征算概率
            X_all = det.source_graph.ndata["feature"].cpu().numpy()
            self._scores = det.model.predict_proba(X_all)[:, 1]
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

        self.is_fitted_ = True
        return self

    def decision_function(self, graph) -> np.ndarray:  # type: ignore[override]
        self._check_fitted()
        if self._scores is None:
            raise RuntimeError(
                f"[{type(self).__name__}] scores not available; call fit() first"
            )
        return self._scores

    def _fit(self, graph, y=None, **kwargs):  # pragma: no cover
        pass

    def _decision_function(self, graph):  # pragma: no cover
        return self._scores
