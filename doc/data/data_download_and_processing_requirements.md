# 成员 A：数据下载与处理要求

本文档整理自 `doc/团队分工文档.pdf`，用于明确数据工程负责人的具体工作范围、处理规范和交付文件。

## 1. 职责范围

成员 A 负责项目的数据工程部分，包括：

- 数据获取与本地目录组织
- 数据清洗、格式统一与预处理
- 表格、CV、NLP、时序、图数据的加载适配
- 训练集污染率注入器实现
- 统一数据加载接口封装
- EDA 统计文件与数据血缘文档
- 报告第 3 章「数据集与预处理」主笔
- 答辩时负责 10 分钟全程讲解，需要熟悉全部实验结果和算法逻辑

## 2. 数据来源与计划使用数据集

本项目使用 3 个 benchmark，覆盖 5 类数据形态，共计划使用 29 个数据集。

### 2.1 ADBench：表格 + CV + NLP

- 来源：Han et al., NeurIPS 2022
- GitHub：https://github.com/Minqi824/ADBench
- 数据格式：`.npz`
- 标准字段：
  - `X`：特征矩阵
  - `y`：标签，`0` 表示正常，`1` 表示异常
- 本地期望位置：`ADBench-main/adbench/datasets/`

计划使用子集：9 个表格 + 4 个 CV + 4 个 NLP，共 17 个。

表格数据集：

| 数据集 | 样本数 | 维度 | 异常率 | 领域 |
| --- | ---: | ---: | ---: | --- |
| Cardio | 1,831 | 21 | 9.6% | 医疗 |
| Thyroid | 3,772 | 6 | 2.5% | 医疗 |
| Satellite | 6,435 | 36 | 31.6% | 遥感 |
| Shuttle | 49,097 | 9 | 7.2% | 工业 |
| Credit Card | 284,807 | 30 | 0.17% | 金融 |
| Pima | 768 | 8 | 35% | 医疗 |
| Annthyroid | 7,200 | 6 | 7.4% | 医疗 |
| Mammography | 11,183 | 6 | 2.3% | 医疗 |
| Pendigits | 6,870 | 16 | 2.3% | 数字识别 |

CV 数据集：

- `CIFAR10_0`
- `CIFAR10_1`
- `FashionMNIST_0`
- `FashionMNIST_1`

每个 CV 数据集约为 `5000 x 512` 维 ResNet-18 特征。

NLP 数据集：

- `20news_0`
- `20news_1`
- `agnews_0`
- `amazon`

每个 NLP 数据集约为 `2000~10000 x 768` 维 BERT embedding。

### 2.2 TSB-AD：时序数据

- 来源：Liu et al., NeurIPS 2024
- GitHub：https://github.com/TheDatumOrg/TSB-AD
- 完整数据下载：https://www.thedatum.org/datasets/TSB-AD-U.zip
- 数据格式：`.csv`
- 标准列：
  - `Data`：数值
  - `Label`：标签，`0` 表示正常，`1` 表示异常
- 文件名包含 `tr_` 索引，用于标记训练截止点
- 本地期望位置：`TSB-AD-main/Datasets/`

计划使用方式：

- 按领域分层抽样 8 条代表性时序
- 覆盖 Facility、Medical、Sensor、Web、Finance、Traffic 等领域，每类 1~2 条
- 将最终选择结果写入 `data/timeseries/selected_files.txt`，保证实验可复现

### 2.3 GADBench：图数据

- 来源：Tang et al., NeurIPS 2023
- GitHub：https://github.com/squareRoot3/GADBench
- 数据下载：https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1
- 数据格式：DGL 二进制图文件
- 数据内容：
  - 节点特征
  - 邻接关系
  - 节点标签
  - 预设 `train/val/test mask`
- 本地期望位置：`GADBench-master/datasets/`

计划使用子集：

| 数据集 | 节点数 | 边数 | 异常率 | 类型 |
| --- | ---: | ---: | ---: | --- |
| T-Finance | 39,357 | 21M | 4.6% | 金融交易图 |
| Reddit | 10,984 | 168K | 3.3% | 用户-帖子图 |
| Amazon | 11,944 | 4.4M | 6.9% | 电商评论图 |
| Weibo | 8,405 | 408K | 10.3% | 社交网络 |

## 3. 数据处理要求

### 3.1 ADBench 加载与预处理

需要实现 `adapters/adbench_adapter.py`。

具体要求：

- 加载 Classical、CV、NLP 的 `.npz` 文件
- 读取 `X` 和 `y`
- 将特征统一转换为 `float64`
- 对未标准化的数据集做 `StandardScaler`
- 对已经归一化的数据集可跳过标准化，例如 `fraud`
- 对整数型且取值范围较大的数据集必须注意标准化，例如 `cover` 的范围约为 `-173~7173`
- 实现分层 `train_test_split`，保证训练集和测试集中的异常比例尽量一致

### 3.2 TSB-AD 时序处理

需要实现 `adapters/timeseries_adapter.py`。

具体要求：

- 下载完整 TSB-AD 数据集
- 解析文件名中的 `tr_` 索引，确定训练段和测试段的分界点
- 只使用训练段统计量做 z-score 归一化，避免测试集信息泄漏
- 实现滑窗切分
- `window_size` 和 `stride` 需要可配置
- 实现窗口标签对齐，窗口内是否异常需要有明确规则并保持一致
- 将最终抽样的 8 条代表性时序记录到 `data/timeseries/selected_files.txt`

### 3.3 GADBench 图加载

需要实现 `adapters/graph_adapter.py`。

具体要求：

- 使用 `DGL load_graphs` 加载二进制图文件
- 提取节点特征
- 提取节点标签
- 提取预设 `train_mask`、`val_mask`、`test_mask`
- 对节点特征做 `StandardScaler`
- 保留原始图结构，供图异常检测算法调用

### 3.4 污染率注入器

需要实现 `data/contaminate.py`。

污染率注入器要支持两种污染方式。

无监督污染：

- 用于无监督算法的 Exp-2 污染率实验
- 控制训练集中异常样本的保留比例
- 污染率档位：`0%`、`5%`、`10%`、`20%`
- `0%` 表示训练集纯净，即训练集中不保留异常样本

有监督污染：

- 用于有监督算法的 Exp-2 污染率实验
- 采用标签双向对称翻转
- 随机将指定比例样本的标签做 `0 <-> 1` 互换
- 污染率档位：`5%`、`10%`、`20%`

通用要求：

- 固定随机种子 `seed`
- 保证每次运行可复现
- 输出污染后的训练数据和对应标签
- 保留污染配置，便于实验日志记录

### 3.5 统一加载接口

需要封装统一数据加载函数。

输入：

- 数据集名称
- 数据形态或自动推断规则
- 可选参数，例如测试集比例、随机种子、是否标准化、滑窗参数、污染率

输出标准格式：

```python
(X_train, X_test, y_train, y_test)
```

接口目标：

- 屏蔽底层数据形态差异
- 让算法侧可以用统一方式调用数据
- 对图数据和时序数据，如确实需要额外对象，可在标准返回值之外提供扩展字段或专用加载函数，但需要在文档中说明

## 4. EDA 与数据血缘要求

### 4.1 EDA 统计文件

需要输出 `data/eda_summary.json`，也可以额外输出 CSV 版本供检查。

每个数据集至少统计：

- 数据集名称
- 数据形态：tabular、cv、nlp、timeseries、graph
- 样本数或节点数
- 特征维度
- 异常样本数
- 异常率
- 特征值范围
- `dtype`
- 是否需要标准化
- 训练/测试划分方式
- 原始文件路径
- 预处理后文件路径或加载方式

该统计文件需要供 Streamlit 前端读取，并体现在报告第 3 章中。

### 4.2 数据血缘文档

需要记录每个数据集的完整处理链路，并体现在报告第 3 章。

每个数据集建议记录：

- 数据来源与下载链接
- 原始文件格式
- 原始字段说明
- 标签含义
- 本地存放位置
- 预处理步骤
- 标准化策略
- 划分策略
- 是否参与污染率实验
- 与实验编号的对应关系

## 5. 成员 A 交付文件清单

| 交付物 | 路径 |
| --- | --- |
| ADBench 适配器 | `adapters/adbench_adapter.py` |
| 时序适配器 | `adapters/timeseries_adapter.py` |
| 图适配器 | `adapters/graph_adapter.py` |
| 污染率注入器 | `data/contaminate.py` |
| EDA 统计文件 | `data/eda_summary.json` |
| 时序抽样记录 | `data/timeseries/selected_files.txt` |
| 数据血缘文档 | 建议放在 `doc/data_lineage.md` 或报告第 3 章草稿 |
| 报告章节 | 第 3 章「数据集与预处理」 |

## 6. 时间节点

- `2026-05-26` 至 `2026-05-28`：完成数据加载与预处理
- `2026-06-01`：数据 pipeline 跑通，并支持 Exp-1 全部运行
- `2026-06-09` 至 `2026-06-12`：完成报告第 3 章
- `2026-06-13` 至 `2026-06-16`：答辩准备

## 7. 验收标准

数据部分完成时，应满足：

- 三类 benchmark 的计划数据均可被加载
- 表格、CV、NLP、时序、图数据有清晰的本地路径和来源记录
- 所有特征类型、标签含义和异常率统计可追溯
- 训练/测试划分固定随机种子，可复现
- 标准化不会使用测试集统计量
- 污染率注入器能分别支持无监督污染和有监督标签翻转
- 算法侧能通过统一接口拿到标准训练/测试数据
- `data/eda_summary.json` 可直接供前端和报告使用
