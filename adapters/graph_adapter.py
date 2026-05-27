"""GADBench 图适配器。

职责：
- 用 DGL load_graphs 读取二进制图文件
- 提取节点特征、标签、预设的 train/val/test mask
- 对节点特征做 StandardScaler
"""
