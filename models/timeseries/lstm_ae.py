"""LSTM AutoEncoder 时序无监督异常检测。

PyTorch 手写：encoder LSTM 把窗口压缩到 hidden_size，decoder LSTM 重构，
重构 MSE 即每个窗口的异常分数。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import TimeSeriesDetector
from models.device import get_preferred_device


class _LSTMAE:
    """轻量级编码器-解码器 LSTM 自编码器。在 _fit 内构造，避免 import torch 失败。"""

    @staticmethod
    def build(input_size: int, hidden_size: int, num_layers: int):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.decoder = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.out = nn.Linear(hidden_size, input_size)

            def forward(self, x):
                # x: (B, T, input_size)
                _, (h, _) = self.encoder(x)
                # h[-1]: (B, hidden_size) -> 沿时间维复制成 (B, T, hidden_size)
                last = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
                dec, _ = self.decoder(last)
                return self.out(dec)

        return Net()


class LSTMAutoEncoderDetector(TimeSeriesDetector):
    """LSTM 自编码器（窗口重构误差作为异常分数）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    hidden_size : int, default=64
    num_layers : int, default=1
    epochs : int, default=100
    batch_size : int, default=32
    lr : float, default=1e-3
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        hidden_size: int = 64,
        num_layers: int = 1,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model: Any | None = None
        self._device: str = "cpu"

    @staticmethod
    def _to_3d(X: np.ndarray) -> np.ndarray:
        # (n, w) -> (n, w, 1)；(w,) -> (1, w, 1)
        if X.ndim == 1:
            return X.reshape(1, -1, 1)
        if X.ndim == 2:
            return X[:, :, np.newaxis]
        if X.ndim == 3:
            return X
        raise ValueError(
            f"LSTMAutoEncoderDetector expects 1D/2D/3D X, got shape {X.shape}"
        )

    def _fit(self, X, y=None, **kwargs):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            raise RuntimeError(
                "[LSTMAutoEncoderDetector] torch 未安装"
            ) from e

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

        device = get_preferred_device()
        self._device = device

        X3 = self._to_3d(X)
        x_tensor = torch.from_numpy(X3.astype(np.float32))
        loader = DataLoader(
            TensorDataset(x_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        try:
            net = _LSTMAE.build(
                input_size=X3.shape[2],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            ).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            loss_fn = torch.nn.MSELoss()

            net.train()
            for ep in range(self.epochs):
                total_loss = 0.0
                n_seen = 0
                for (batch,) in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = net(batch)
                    loss = loss_fn(out, batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * batch.size(0)
                    n_seen += batch.size(0)
                if (ep + 1) % max(1, self.epochs // 5) == 0:
                    print(
                        f"  [LSTM-AE] epoch {ep + 1}/{self.epochs} "
                        f"loss={total_loss / max(n_seen, 1):.4f}"
                    )
            self._model = net
        except Exception as e:
            raise RuntimeError(
                f"[{type(self).__name__}] 训练失败: {e}"
            ) from e

    def _decision_function(self, X):
        import torch

        assert self._model is not None
        X3 = self._to_3d(X)
        self._model.eval()
        
        mses = []
        with torch.no_grad():
            for i in range(0, len(X3), self.batch_size):
                batch_x = X3[i:i + self.batch_size]
                x_tensor = torch.from_numpy(batch_x.astype(np.float32)).to(self._device)
                out = self._model(x_tensor)
                mse = ((out - x_tensor) ** 2).mean(dim=(1, 2))
                mses.append(mse.detach().cpu().numpy())
                
        return np.concatenate(mses).astype(np.float64)
