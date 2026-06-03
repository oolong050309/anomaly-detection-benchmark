"""LSTM 监督式时序异常检测。

PyTorch 手写：单层 LSTM 取最后一个时刻的隐状态 → Linear → sigmoid。
``decision_function`` 返回 sigmoid 概率（异常类）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import SupervisedDetector, TimeSeriesDetector
from models.device import get_preferred_device


class _LSTMClassifier:
    @staticmethod
    def build(input_size: int, hidden_size: int, num_layers: int):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x: (B, T, input_size)
                _, (h, _) = self.lstm(x)
                logits = self.fc(h[-1])  # (B, 1)
                return logits.squeeze(-1)  # (B,)

        return Net()


class LSTMSupervisedDetector(TimeSeriesDetector, SupervisedDetector):
    """LSTM 二分类时序异常检测。

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
        if X.ndim == 1:
            return X.reshape(1, -1, 1)
        if X.ndim == 2:
            return X[:, :, np.newaxis]
        if X.ndim == 3:
            return X
        raise ValueError(
            f"LSTMSupervisedDetector expects 1D/2D/3D X, got shape {X.shape}"
        )

    def _fit(self, X, y=None, **kwargs):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            raise RuntimeError(
                "[LSTMSupervisedDetector] torch 未安装"
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
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))

        # 类不平衡：用 pos_weight 给正样本（异常）加权
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        pos_weight_value = n_neg / max(n_pos, 1.0)

        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        try:
            net = _LSTMClassifier.build(
                input_size=X3.shape[2],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            ).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            pos_weight = torch.tensor([pos_weight_value], device=device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            net.train()
            for ep in range(self.epochs):
                total_loss = 0.0
                n_seen = 0
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    logits = net(batch_x)
                    loss = loss_fn(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * batch_x.size(0)
                    n_seen += batch_x.size(0)
                if (ep + 1) % max(1, self.epochs // 5) == 0:
                    print(
                        f"  [LSTM-Sup] epoch {ep + 1}/{self.epochs} "
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
        
        preds = []
        with torch.no_grad():
            for i in range(0, len(X3), self.batch_size):
                batch_x = X3[i:i + self.batch_size]
                x_tensor = torch.from_numpy(batch_x.astype(np.float32)).to(self._device)
                logits = self._model(x_tensor)
                probs = torch.sigmoid(logits)
                preds.append(probs.detach().cpu().numpy())
                
        return np.concatenate(preds).astype(np.float64)
