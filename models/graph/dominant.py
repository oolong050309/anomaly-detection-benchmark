"""DOMINANT 图异常检测（PyGOD 包装）。

委托 ``pygod.detector.DOMINANT``。需要 PyG ``Data`` 对象作为 fit 输入。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import GraphDetector
from models.device import cuda_available, cuda_device_index, get_preferred_device, maybe_add_supported_kwargs


class DOMINANTDetector(GraphDetector):
    """DOMINANT 图自编码器异常检测（SDM 2019）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    hid_dim : int, default=64
    num_layers : int, default=4
    epoch : int, default=100
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        hid_dim: int = 64,
        num_layers: int = 4,
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
            from pygod.detector import DOMINANT
        except ImportError as e:
            raise RuntimeError(
                "[DOMINANTDetector] pygod 未安装"
            ) from e

        # 固定随机种子
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
            model_kwargs = maybe_add_supported_kwargs(
                DOMINANT,
                self._algo_kwargs,
                {
                    "gpu": cuda_device_index() if cuda_available() else -1,
                    "device": get_preferred_device(),
                },
            )
            self._model = DOMINANT(
                hid_dim=self.hid_dim,
                num_layers=self.num_layers,
                epoch=self.epoch,
                contamination=self.contamination,
                **model_kwargs,
            )
            self._model.fit(graph)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, graph):
        assert self._model is not None
        try:
            # PyGOD 1.x: predict(data, return_pred=False, return_score=True)
            scores = self._model.predict(graph, return_pred=False, return_score=True)
        except TypeError:
            # 旧 API：直接读 decision_score_ 或 decision_function
            if hasattr(self._model, "decision_function"):
                scores = self._model.decision_function(graph)
            else:
                scores = self._model.decision_score_
        return np.asarray(scores, dtype=np.float64).ravel()
