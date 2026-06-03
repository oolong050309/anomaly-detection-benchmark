"""MatrixProfile 时序异常检测（无监督）。

使用 ``stumpy.stump`` 计算 matrix profile —— 每个子序列的最近邻距离。
距离越大表示越罕见，即异常分数越大。

GPU 环境注意事项
----------------
``stumpy`` 在 ``import`` 时就让 numba 编译 GPU kernel（``gpu_aamp.py``）。
若进程可见 GPU，而 numba/CUDA 版本不兼容，会在 import 阶段直接崩
（``_gpu_searchsorted_right`` TypingError）。这无法在运行时绕过。

解决办法：跑实验前在 **进程启动时** 设环境变量 ``NUMBA_DISABLE_CUDA=1``。
它只禁用 numba 的 CUDA（stumpy 走 CPU），**不影响 PyTorch 的 GPU**
（torch 用自己的 CUDA，不经过 numba）。因此时序模态可同时：
LSTM/DADA 用 GPU（CUDA_VISIBLE_DEVICES=0），MatrixProfile 用 CPU。

    export CUDA_VISIBLE_DEVICES=0
    export NUMBA_DISABLE_CUDA=1
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import TimeSeriesDetector


class MatrixProfileDetector(TimeSeriesDetector):
    """基于 Matrix Profile 的时序异常检测。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
        Matrix Profile 是确定性算法，参数保留为接口一致。
    window_size : int, default=100
        子序列窗口长度。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        window_size: int = 100,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.window_size = window_size
        self._train_seq: np.ndarray | None = None

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        # MatrixProfile 需要拼接整段序列；fit 阶段记录训练段供拼接评分
        self._train_seq = X.astype(np.float64).ravel() if X.ndim == 2 else X.copy()

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        try:
            import stumpy
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "[MatrixProfileDetector] stumpy 未安装"
            ) from e

        seq = X.astype(np.float64).ravel() if X.ndim == 2 else X.copy()
        m = self.window_size
        if seq.size <= m:
            raise RuntimeError(
                f"[MatrixProfileDetector] sequence length {seq.size} <= window_size {m}"
            )
        try:
            if self._train_seq is not None and self._train_seq.size > m:
                # AB-join: score each test subsequence by its nearest training
                # subsequence, so Exp-2 train contamination can affect scores.
                # T_A=test, T_B=train。AB-join 必须 ignore_trivial=False，
                # 否则 stumpy 会套用 self-join 的平凡匹配排除逻辑，可能返回
                # 空 / 异常 profile（曾导致 distances[-1] IndexError）。
                mp = stumpy.stump(
                    seq, m=m, T_B=self._train_seq, ignore_trivial=False
                )
            else:
                mp = stumpy.stump(seq, m=m)
        except Exception as e:
            raise RuntimeError(
                f"[MatrixProfileDetector] 训练失败: {e}"
            ) from e

        # mp[:, 0] 是每个子序列起点的最近邻距离，长度为 len(seq)-m+1
        # 把每个子序列的分数赋给其起点位置，末尾不足一窗的位置填最后一个有效分数
        distances = np.asarray(mp[:, 0], dtype=np.float64).ravel()
        # stumpy 在某些 AB-join / 退化输入下可能返回空或含 inf/nan 的 profile，
        # 这里做健壮化处理，避免 distances[-1] 越界或分数含非有限值。
        if distances.size == 0:
            return np.zeros(seq.size, dtype=np.float64)
        distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)

        # 对齐到 seq 长度
        scores = np.empty(seq.size, dtype=np.float64)
        fill = min(distances.size, seq.size)
        scores[:fill] = distances[:fill]
        scores[fill:] = distances[fill - 1] if fill > 0 else 0.0

        # 如果输入是 2D 窗口形式 (n_windows, w)，按窗口起点取分数
        if X.ndim == 2:
            n_windows, w = X.shape
            # 简化处理：返回每个窗口对应的 matrix profile 分数（按窗口起点）
            stride = max(1, (seq.size - w) // max(1, n_windows - 1))
            window_scores = np.array(
                [scores[min(i * stride, seq.size - 1)] for i in range(n_windows)]
            )
            return window_scores

        return scores
