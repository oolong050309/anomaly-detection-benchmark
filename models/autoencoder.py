"""AutoEncoder 包装（PyOD）。

优先委托 PyTorch 后端的 AutoEncoder，避免依赖 tensorflow：
- pyod 2.x：``pyod.models.auto_encoder.AutoEncoder`` 本身就是 PyTorch 实现
- pyod 1.x：PyTorch 版在 ``pyod.models.auto_encoder_torch.AutoEncoder``，
  而 ``pyod.models.auto_encoder.AutoEncoder`` 是 TensorFlow 实现

导入顺序：auto_encoder_torch → auto_encoder，确保两台机器都走 PyTorch、
后端一致、结果可比。``_fit`` 起始处统一固定 PyTorch / NumPy 随机种子。
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


def _import_torch_autoencoder():
    """返回 PyTorch 后端的 pyod AutoEncoder 类。

    优先 ``auto_encoder_torch``（pyod 1.x 的 PyTorch 版），
    回退 ``auto_encoder``（pyod 2.x 已是 PyTorch 版）。
    两者都失败才抛错，错误信息提示安装 PyTorch 版而非 tensorflow。
    """
    try:
        from pyod.models.auto_encoder_torch import AutoEncoder  # pyod 1.x torch 版
        return AutoEncoder
    except ImportError:
        pass
    try:
        from pyod.models.auto_encoder import AutoEncoder  # pyod 2.x（torch）
        return AutoEncoder
    except ImportError as e:
        raise RuntimeError(
            "[AutoEncoderDetector] 找不到 PyTorch 版 AutoEncoder。"
            "pyod 1.x 请确认存在 auto_encoder_torch，pyod 2.x 直接用 auto_encoder。"
            "不要依赖 tensorflow 版。"
        ) from e


class AutoEncoderDetector(BaseDetector):
    """AutoEncoder（重构误差异常检测器）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    hidden_neuron_list : list[int], default=[64, 32]
        编码器隐藏层维度（解码器对称镜像）。
    epoch_num : int, default=100
    batch_size : int, default=64
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        hidden_neuron_list: list[int] | None = None,
        epoch_num: int = 100,
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

        import inspect

        AutoEncoder = _import_torch_autoencoder()

        try:
            model_kwargs = maybe_add_supported_kwargs(
                AutoEncoder,
                self._algo_kwargs,
                {
                    "device": get_preferred_device(),
                    "random_state": self.random_state,
                },
            )
            _ae_params = inspect.signature(AutoEncoder.__init__).parameters
            n_features = int(X.shape[1])

            # ---- epoch 参数名：epoch_num（torch 1.x）还是 epochs（2.x）----
            epoch_key = "epoch_num" if "epoch_num" in _ae_params else "epochs"

            # ---- hidden 层参数名 + 约束 ----
            if "hidden_neuron_list" in _ae_params:
                # pyod 1.x auto_encoder_torch：用 hidden_neuron_list，编码器维度，
                # 内部自动镜像解码器，无需手动对称；按特征维度裁剪即可。
                enc = [max(1, min(int(h), n_features)) for h in self.hidden_neuron_list]
                hidden_kwargs = {"hidden_neuron_list": enc}
            else:
                # pyod 2.x auto_encoder：用 hidden_neurons，要求对称 + 不超过特征数。
                enc = [max(1, min(int(h), n_features)) for h in self.hidden_neuron_list]
                hidden_kwargs = {"hidden_neurons": enc + enc[::-1]}

            ctor_kwargs = {
                epoch_key: self.epoch_num,
                "batch_size": self.batch_size,
                "contamination": self.contamination,
                **hidden_kwargs,
                **model_kwargs,
            }
            self._model = AutoEncoder(**ctor_kwargs)
            self._model.fit(X)
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.decision_function(X)
