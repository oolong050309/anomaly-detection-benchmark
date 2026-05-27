"""统计基线：IQRDetector。

基于 Tukey 1.5×IQR 围栏法的多维异常分数：

  对每个特征 j：
      lower_j = Q1_j - 1.5 * IQR_j
      upper_j = Q3_j + 1.5 * IQR_j
      contrib_j(x) = max(0, x_j - upper_j) + max(0, lower_j - x_j)

  最终样本异常分数 = aggregation 聚合（max / mean / sum），默认 max。

设计意图：
- 自实现、无外部依赖（只用 numpy），用于验证 BaseDetector 模板方法是否正确。
- 常量列（IQR=0）的贡献被强制置 0，避免在数据有冗余特征时产生除零或 NaN。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseDetector


_VALID_AGGREGATION = {"max", "mean", "sum"}


class IQRDetector(BaseDetector):
    """基于 1.5×IQR 围栏法的统计异常检测器。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
        IQR 是确定性算法，不使用此参数（保留是为了接口一致性）。
    aggregation : str, default="max"
        多维特征聚合方式：``"max"`` / ``"mean"`` / ``"sum"``。
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        aggregation: str = "max",
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        if aggregation not in _VALID_AGGREGATION:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATION}, "
                f"got {aggregation!r}"
            )
        self.aggregation = aggregation

        # fit 后填充
        self.q1_: np.ndarray | None = None
        self.q3_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None
        self.constant_mask_: np.ndarray | None = None

    # ---- BaseDetector 钩子 ----

    def _fit(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        # X 已被基类校验为 (n, d) float64 且无 NaN/Inf
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_ = self.q1_ - 1.5 * self.iqr_
        self.upper_ = self.q3_ + 1.5 * self.iqr_
        self.constant_mask_ = self.iqr_ == 0

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.lower_ is not None and self.upper_ is not None
        assert self.constant_mask_ is not None

        # contrib shape (n, d)
        upper_excess = np.maximum(0.0, X - self.upper_)
        lower_excess = np.maximum(0.0, self.lower_ - X)
        contrib = upper_excess + lower_excess

        if self.constant_mask_.any():
            contrib[:, self.constant_mask_] = 0.0

        if self.aggregation == "max":
            scores = contrib.max(axis=1)
        elif self.aggregation == "mean":
            scores = contrib.mean(axis=1)
        else:  # "sum"
            scores = contrib.sum(axis=1)

        return scores
