"""DADA 时序基础模型包装（ICLR 2025）。

底层实现来自 vendor 包 ``models/timeseries/_vendor/dada/``。
DADA 是一个 zero-shot 时序异常检测器，预训练在多领域时序上，使用时直接加载权重。

接口契约：
- 输入：一维序列 ``(T,)`` 或二维窗口 ``(n_windows, window_size)``。
- ``fit(X)``：仅加载预训练权重（不训练）；``y`` 被忽略。
- ``decision_function(X)``：对每个窗口返回标量异常分数。
  - 输入是二维 ``(n_windows, w)`` 时：返回 ``(n_windows,)``，分数 = 该窗口所有点
    异常分数的均值。
  - 输入是一维 ``(T,)`` 时：先把序列切成不重叠的 ``seq_len`` 长窗口推理，
    然后把每点分数还原回长度 T 的数组。

注意：DADA 预训练时使用 ``seq_len=100, patch_len=5``。预训练权重的形状与之绑定。
我们的包装类只暴露这两个值的 getter，不允许外部修改。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import TimeSeriesDetector


_DADA_DIR = Path(__file__).resolve().parent / "_vendor" / "dada"


def _select_device(prefer_cuda: bool = True) -> str:
    try:
        import torch
        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


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


class DADADetector(TimeSeriesDetector):
    """DADA 时序基础模型异常检测器（zero-shot）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
        DADA 推理时若使用 ``mask_mode='c'/'random'`` 会引入随机性，本参数控制 seed。
    seq_len : int, default=100
        推理窗口长度（与预训练权重绑定，不要修改）。
    norm : int, default=0
        ``0`` 不做窗口内归一化；``1`` 做。
    copies : int, default=10
        每个输入做多少次随机 mask 推理求方差，用于异常分数。
    mask_mode : str, default='c'
        ``'c'`` 对称 mask、``'random'`` 随机 mask、``'nomask'`` 不 mask。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        seq_len: int = 100,
        norm: int = 0,
        copies: int = 10,
        mask_mode: str = "c",
        device: str = "auto",
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.seq_len = int(seq_len)
        self.norm = int(norm)
        self.copies = int(copies)
        self.mask_mode = mask_mode
        self.device = device
        self._model: Any | None = None
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # BaseDetector 钩子
    # ------------------------------------------------------------------

    def _fit(self, X, y=None, **kwargs):
        """加载预训练权重；DADA 是 zero-shot，不在用户数据上训练。"""
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("[DADADetector] torch 未安装") from e

        if not _DADA_DIR.exists() or not (_DADA_DIR / "pytorch_model.bin").exists():
            raise RuntimeError(
                f"[DADADetector] 预训练权重缺失：期望路径 {_DADA_DIR}\n"
                "请确认 models/timeseries/_vendor/dada/ 下有 pytorch_model.bin。"
            )

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "[DADADetector] transformers 未安装；运行 `pip install transformers`"
            ) from e

        _fix_torch_seed(self.random_state)
        device = self.device if self.device != "auto" else _select_device()
        self._device = device

        try:
            # trust_remote_code=True 会执行 vendor/dada/configuration_DADA.py
            # 与 modeling_DADA.py 中的代码
            self._model = AutoModel.from_pretrained(
                str(_DADA_DIR), trust_remote_code=True
            ).to(device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(
                f"[DADADetector] 加载预训练权重失败: {e}"
            ) from e

    def _decision_function(self, X):
        import torch

        if self._model is None:
            raise RuntimeError(
                f"[{type(self).__name__}] _model is None; call fit() first"
            )

        # 输入归一化到形如 (B, T, C) 的 float32 张量
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            # 一维序列：切窗 + 推理
            return self._score_long_sequence(X_arr)
        if X_arr.ndim == 2:
            # (n_windows, window_size)：每窗口当作一段独立序列
            n_windows, w = X_arr.shape
            scores_per_window = []
            for i in range(n_windows):
                seq = X_arr[i]
                # 用同样的滑窗逻辑（对窗口内做 padding/截断到 seq_len）
                seq_padded = self._pad_or_truncate(seq, self.seq_len)
                pt = (
                    torch.from_numpy(seq_padded).reshape(1, self.seq_len, 1)
                    .to(self._device)
                )
                with torch.no_grad():
                    score = self._model.infer(pt, norm=self.norm)
                # score: (1, seq_len)，取均值作为窗口分数
                scores_per_window.append(float(score.mean().item()))
            return np.asarray(scores_per_window, dtype=np.float64)
        raise ValueError(
            f"[DADADetector] expected 1D or 2D X, got shape {X_arr.shape}"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
        if seq.size >= target_len:
            return seq[:target_len].astype(np.float32)
        pad = np.zeros(target_len - seq.size, dtype=np.float32)
        return np.concatenate([seq.astype(np.float32), pad])

    def _score_long_sequence(self, seq: np.ndarray) -> np.ndarray:
        """对一维长序列，按 ``seq_len`` 不重叠切窗推理，再拼回点级分数。"""
        import torch

        T = seq.size
        L = self.seq_len
        scores = np.zeros(T, dtype=np.float64)

        # 不重叠切窗
        idx = 0
        while idx < T:
            end = min(idx + L, T)
            chunk = seq[idx:end]
            chunk_pad = self._pad_or_truncate(chunk, L)
            pt = (
                torch.from_numpy(chunk_pad).reshape(1, L, 1).to(self._device)
            )
            with torch.no_grad():
                s = self._model.infer(pt, norm=self.norm)
            s_np = s.detach().cpu().numpy().reshape(-1)  # (L,)
            scores[idx:end] = s_np[: end - idx]
            idx = end

        return scores
