"""ADBench 适配器：加载 Classical / CV_by_ResNet18 / NLP_by_BERT 的 npz 文件。

职责：
- 统一读取 npz -> (X, y)，X 转 float64
- 对未标准化的数据集做 StandardScaler（已归一化的可跳过）
- 实现分层 train_test_split
- 屏蔽底层形态差异，对外提供统一接口
"""
