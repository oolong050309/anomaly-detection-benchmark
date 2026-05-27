"""AutoEncoder 包装（PyOD）。

委托 ``pyod.models.auto_encoder.AutoEncoder``。基于 PyTorch 实现。
``_fit`` 起始处统一固定 PyTorch / NumPy 随机种子以保证可复现。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseDetector
from models.device import get_preferred_device, maybe_add_supported_kwargs


def _set_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:  # pragma: no cover
        pass


class AutoEncoderDetector(BaseDetector):
    """AutoEncoder（重构误差异常检测器）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    hidden_neuron_list : list[int], default=[64, 32]
        编码器隐藏层维度（解码器对称镜像）。
    epoch_num : int, default=20
    batch_size : int, default=64
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        hidden_neuron_list: list[int] | None = None,
        epoch_num: int = 20,
        batch_size: int = 64,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.hidden_neuron_list = (
            list(hidden_neuron_list) if hidden_neuron_list else [64, 32]
        )
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        _set_torch_seed(self.random_state)
        np.random.seed(self.random_state if self.random_state is not None else 42)

        from pyod.models.auto_encoder import AutoEncoder

        try:
            model_kwargs = maybe_add_supported_kwargs(
                AutoEncoder,
                self._algo_kwargs,
                {
                    "device": get_preferred_device(),
                    "random_state": self.random_state,
                },
            )
            self._model = AutoEncoder(
                hidden_neuron_list=self.hidden_neuron_list,
                epoch_num=self.epoch_num,
                batch_size=self.batch_size,
                contamination=self.contamination,
                **model_kwargs,
            )
            self._model.fit(X)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.decision_function(X)
