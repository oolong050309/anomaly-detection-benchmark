"""DADA vendor 包。

来源：https://github.com/iambowen/DADA
论文：Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks
       and Dual Adversarial Decoders (ICLR 2025)

文件清单：
    modeling_DADA.py       - 模型主体（继承 HuggingFace PreTrainedModel）
    configuration_DADA.py  - 模型配置类
    config.json            - 默认超参 JSON
    pytorch_model.bin      - 预训练权重 (~7 MB)
    LICENSE                - 原始仓库未提供 LICENSE，引用时请遵循论文要求

使用方式：
    from transformers import AutoModel
    model_dir = "path/to/this/dada/folder"
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    score = model.infer(x)   # x: (B, T, C)
"""

from .modeling_DADA import DADA, Model
from .configuration_DADA import DADAConfig

__all__ = ["DADA", "Model", "DADAConfig"]
