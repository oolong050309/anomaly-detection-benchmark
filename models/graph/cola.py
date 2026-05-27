"""CoLA 图对比学习异常检测（PyGOD 包装）。

委托 ``pygod.detector.CoLA``。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import GraphDetector


class CoLADetector(GraphDetector):
    """CoLA: Contrastive Learning for Anomaly Detection on Graphs (TNNLS 2021)。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    hid_dim : int, default=64
    num_layers : int, default=2
    epoch : int, default=100
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        hid_dim: int = 64,
        num_layers: int = 2,
        epoch: int = 100,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.epoch = epoch
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(self, graph, y=None, **kwargs):
        try:
            from pygod.detector import CoLA
        except ImportError as e:
            raise RuntimeError(
                "[CoLADetector] pygod 未安装"
            ) from e

        if self.random_state is not None:
            try:
                import torch
                torch.manual_seed(int(self.random_state))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.random_state))
            except ImportError:
                pass
            np.random.seed(int(self.random_state))

        try:
            self._model = CoLA(
                hid_dim=self.hid_dim,
                num_layers=self.num_layers,
                epoch=self.epoch,
                contamination=self.contamination,
                **self._algo_kwargs,
            )
            self._model.fit(graph)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, graph):
        assert self._model is not None
        try:
            scores = self._model.predict(graph, return_pred=False, return_score=True)
        except TypeError:
            if hasattr(self._model, "decision_function"):
                scores = self._model.decision_function(graph)
            else:
                scores = self._model.decision_score_
        return np.asarray(scores, dtype=np.float64).ravel()
