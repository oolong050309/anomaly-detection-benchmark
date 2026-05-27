"""UNPrompt vendor 包。

来源：https://github.com/mala-lab/UNPrompt
论文：UNPrompt - Universal Prompt Learning for Graph Anomaly Detection (IJCAI 2025)
LICENSE：MIT (见同目录 LICENSE 文件)

文件清单：
    model.py   - GCN backbone + SimplePrompt / GPFplusAtt prompt + Projection
    utils.py   - 图加载与稀疏矩阵工具函数
    LICENSE    - 原始 MIT 协议

使用方式：
    from models.graph._vendor.unprompt.model import Model, GPFplusAtt, Projection

注意：UNPrompt 不是 zero-shot；需要在源图上预训练，再在目标图上做 prompt-tuning。
预训练权重需要单独下载或自己 pretrain（见仓库 pretrain.py）。
"""

from .model import GCN, GPFplusAtt, Model, Projection, SimplePrompt

__all__ = ["GCN", "Model", "SimplePrompt", "GPFplusAtt", "Projection"]
