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
from models.device import get_preferred_device


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
    batch_size : int, default=32
        长序列推理时一次 forward 的 chunk 数。OOM 时自动减半重试。
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
        batch_size: int = 32,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.seq_len = int(seq_len)
        self.norm = int(norm)
        self.copies = int(copies)
        self.mask_mode = mask_mode
        self.device = device
        self.batch_size = int(batch_size)
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
        device = self.device if self.device != "auto" else get_preferred_device()
        self._device = device

        try:
            # 直接用 vendored 的 DADA 类加载权重，绕开 AutoModel 与 transformers 配置注册的兼容问题
            import json
            import torch as _torch
            from models.timeseries._vendor.dada import DADA, DADAConfig

            config_path = _DADA_DIR / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            # DADAConfig 期望参数为 list，剥离 transformers 自动注入的字段
            for key in ("transformers_version", "torch_dtype", "architectures"):
                config_dict.pop(key, None)
            config = DADAConfig(**config_dict)

            model = DADA(config)
            state_dict = _torch.load(
                _DADA_DIR / "pytorch_model.bin", map_location="cpu"
            )
            model.load_state_dict(state_dict, strict=False)
            self._model = model.to(device)
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
        """对一维长序列，按 ``seq_len`` 不重叠切窗推理，再拼回点级分数。

        批量化版本：先把所有不重叠 chunk（最后一个 zero-pad 到 ``seq_len``）
        ``stack`` 成 ``[n_chunks, seq_len]``，然后按 ``self.batch_size`` 分批
        ``forward`` 一次性算分。OOM 时自动把 batch_size 减半重试，降到 1 仍 OOM
        抛 ``RuntimeError``。

        输出形状/顺序 / 数值与逐 chunk 循环版本等价（同权重、同切分、同 dtype）。
        """
        import torch

        T = int(seq.size)
        L = self.seq_len
        scores = np.zeros(T, dtype=np.float64)

        if T == 0:
            return scores

        # 1) 先把所有不重叠 chunk 收齐（与原循环切法 1:1 等价：
        #    最后一个 chunk 长度 < L 时 zero-pad 到 L，但写回时只取前 (end-idx) 个点）。
        starts: list[int] = []
        chunk_list: list[np.ndarray] = []
        idx = 0
        while idx < T:
            end = min(idx + L, T)
            chunk = seq[idx:end]
            chunk_pad = self._pad_or_truncate(chunk, L)
            starts.append(idx)
            chunk_list.append(chunk_pad)
            idx = end

        # [n_chunks, L] float32 张量；reshape 到 (n_chunks, L, 1) 喂模型。
        all_chunks = torch.from_numpy(np.stack(chunk_list, axis=0))

        n_chunks = all_chunks.shape[0]
        bs = max(int(self.batch_size), 1)

        # 2) 按 batch_size 分批 forward；外层一个 with torch.no_grad()，
        #    内层每个批次包 try/except OOM 自动减半重试。
        with torch.no_grad():
            i = 0
            outs_np: list[np.ndarray] = []
            while i < n_chunks:
                batch_cpu = all_chunks[i : i + bs]
                try:
                    batch = batch_cpu.reshape(batch_cpu.shape[0], L, 1).to(
                        self._device
                    )
                    s = self._model.infer(batch, norm=self.norm)
                    # s: (batch, L) -> 搬到 CPU、转 numpy
                    s_np = s.detach().cpu().numpy().reshape(batch_cpu.shape[0], L)
                    outs_np.append(s_np)
                    i += bs
                except torch.cuda.OutOfMemoryError:
                    # 释放显存，batch_size 减半重试同一批起点；不前移 i。
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    bs = bs // 2
                    if bs < 1:
                        raise RuntimeError(
                            "DADA OOM even at batch_size=1"
                        )

        out_all = np.concatenate(outs_np, axis=0)  # [n_chunks, L]

        # 3) 写回原序列，最后一段只取前 (end-idx) 个点（与原版对齐）。
        for k, start in enumerate(starts):
            end = min(start + L, T)
            scores[start:end] = out_all[k, : end - start]

        return scores
