"""Deep SVDD 包装（PyOD）。

委托 ``pyod.models.deep_svdd.DeepSVDD``。基于 PyTorch 实现。
``n_features`` 在 ``_fit`` 内自动从 ``X.shape[1]`` 推断。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.autoencoder import _set_torch_seed
from models.base import BaseDetector
from models.device import cuda_device_index, get_preferred_device, maybe_add_supported_kwargs


class DeepSVDDDetector(BaseDetector):
    """Deep Support Vector Data Description（基于深度边界的异常检测）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    epochs : int, default=20
    batch_size : int, default=64
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        epochs: int = 100,
        batch_size: int = 64,
        **algo_kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self._algo_kwargs = algo_kwargs
        self._model: Any | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        _set_torch_seed(self.random_state)
        np.random.seed(self.random_state if self.random_state is not None else 42)

        try:
            from pyod.models.deep_svdd import DeepSVDD
        except ImportError as e:
            # pyod 2.x 的 deep_svdd 是 PyTorch 实现；老版可能拽进 tensorflow。
            raise RuntimeError(
                "[DeepSVDDDetector] 无法导入 PyTorch 版 DeepSVDD。"
                "请升级到 pyod>=2.0（深度模型统一为 PyTorch，无需 tensorflow）。"
            ) from e

        n_features = X.shape[1]

        try:
            device = get_preferred_device()
            model_kwargs = maybe_add_supported_kwargs(
                DeepSVDD,
                self._algo_kwargs,
                {
                    "device": device,
                    "use_cuda": device.startswith("cuda"),
                    "gpu": cuda_device_index(),
                },
            )
            import inspect
            _svdd_params = inspect.signature(DeepSVDD.__init__).parameters
            if "n_features" in _svdd_params:
                self._model = DeepSVDD(
                    n_features=n_features,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    contamination=self.contamination,
                    random_state=self.random_state,
                    **model_kwargs,
                )
            else:
                # pyod >=1.1: n_features removed, inferred automatically
                self._model = DeepSVDD(
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    contamination=self.contamination,
                    random_state=self.random_state,
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
