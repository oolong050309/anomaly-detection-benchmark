"""MatrixProfile 时序异常检测（无监督）。

使用 ``stumpy.stump`` 计算 matrix profile —— 每个子序列的最近邻距离。
距离越大表示越罕见，即异常分数越大。
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
                mp = stumpy.stump(seq, m=m, T_B=self._train_seq)
            else:
                mp = stumpy.stump(seq, m=m)
        except Exception as e:
            raise RuntimeError(
                f"[MatrixProfileDetector] 训练失败: {e}"
            ) from e

        # mp[:, 0] 是每个子序列起点的最近邻距离，长度为 len(seq)-m+1
        # 为了得到与输入序列等长的分数数组，把每个子序列的分数赋给其起点位置，
        # 末尾不足一个窗口的位置填最后一个有效分数
        distances = np.asarray(mp[:, 0], dtype=np.float64)
        # 对齐到 seq 长度
        scores = np.empty(seq.size, dtype=np.float64)
        scores[: distances.size] = distances
        scores[distances.size :] = distances[-1]

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
