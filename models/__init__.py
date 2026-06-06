"""Anomaly detection models package.

显式导出全部 21 个 Detector + 4 个基类，便于团队成员通过统一入口调用：

    from models import IForestDetector, XGBoostDetector
    from models import LSTMAutoEncoderDetector, DOMINANTDetector
"""

from __future__ import annotations

# ---- 4 个基类 ----
from models.base import (
    BaseDetector,
    GraphDetector,
    SupervisedDetector,
    TimeSeriesDetector,
)

# ---- 1 个自实现 ----
from models.statistical import IQRDetector

# ---- 1 个鲁棒防御器 ----
from models.defense import RobustDefenseWrapper

# ---- 6 个浅层 PyOD 包装 ----
from models.lof import LOFDetector
from models.knn import KNNDetector
from models.iforest import IForestDetector
from models.ecod_copod import ECODDetector, COPODDetector
from models.ocsvm import OCSVMDetector

# ---- 2 个深度 PyOD 包装 ----
from models.autoencoder import AutoEncoderDetector
from models.deep_svdd import DeepSVDDDetector

# ---- 6 个表格有监督 ----
from models.supervised import (
    LogisticRegressionDetector,
    RandomForestDetector,
    MLPDetector,
    XGBoostDetector,
    LightGBMDetector,
    TabPFNDetector,
)

__all__ = [
    # 基类
    "BaseDetector",
    "SupervisedDetector",
    "TimeSeriesDetector",
    "GraphDetector",
    # 表格无监督（9 个）
    "IQRDetector",
    "LOFDetector",
    "KNNDetector",
    "IForestDetector",
    "ECODDetector",
    "COPODDetector",
    "OCSVMDetector",
    "AutoEncoderDetector",
    "DeepSVDDDetector",
    # 表格有监督（6 个）
    "LogisticRegressionDetector",
    "RandomForestDetector",
    "MLPDetector",
    "XGBoostDetector",
    "LightGBMDetector",
    "TabPFNDetector",
    # 鲁棒防御器
    "RobustDefenseWrapper",
]


# ---- 第二阶段（5 个 vendor 进来的算法），条件导入，缺依赖时静默 ----
try:
    from models.graph.gnn_supervised import (
        BWGNNDetector,
        GCNDetector,
        XGBGraphDetector,
    )
    __all__ += ["GCNDetector", "BWGNNDetector", "XGBGraphDetector"]
except Exception:  # pragma: no cover
    pass

try:
    from models.graph.unprompt import UNPromptDetector
    __all__.append("UNPromptDetector")
except Exception:  # pragma: no cover
    pass

try:
    from models.timeseries.dada import DADADetector
    __all__.append("DADADetector")
except Exception:  # pragma: no cover
    pass
