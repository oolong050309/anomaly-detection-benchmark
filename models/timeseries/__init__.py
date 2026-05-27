"""时序专属算法包。"""

from __future__ import annotations

from models.timeseries.lstm_ae import LSTMAutoEncoderDetector
from models.timeseries.lstm_supervised import LSTMSupervisedDetector
from models.timeseries.matrix_profile import MatrixProfileDetector
from models.timeseries.minirocket import MiniRocketDetector

__all__ = [
    "MatrixProfileDetector",
    "MiniRocketDetector",
    "LSTMAutoEncoderDetector",
    "LSTMSupervisedDetector",
]
