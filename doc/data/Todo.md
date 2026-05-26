# 成员 A 数据工程 Todo

本文档记录异常检测 benchmark 项目中，成员 A 接下来需要完成的数据下载、处理、适配和文档任务。详细要求见 `doc/data_download_and_processing_requirements.md`。

## 0. 当前状态

- [x] 已阅读 `doc/团队分工文档.pdf`
- [x] 已整理成员 A 数据下载与处理要求：`doc/data_download_and_processing_requirements.md`
- [x] 已确认服务器 `/root/autodl-tmp/final_project` 属于本项目：git remote 为 `https://github.com/oolong050309/anomaly-detection-benchmark.git`，README 标题为 `anomaly-detection-benchmark`
- [x] 已在服务器下载 ADBench 计划子集到 `/root/autodl-tmp/final_project/data/raw/ADBench`
- [x] 已在服务器下载并解压 TSB-AD-U 到 `/root/autodl-tmp/final_project/data/raw/TSB-AD-U`
- [x] 已在服务器克隆 TSB-AD 与 GADBench 源码仓库到 `/root/autodl-tmp/final_project/repos`
- [x] GADBench 图数据包已上传并整理到 `/root/autodl-tmp/final_project/data/raw/GADBench`
- [x] 已创建数据适配器、污染注入器、EDA 输出脚本和对接说明文档

## 1. 仓库与目录初始化

- [x] 创建项目目录骨架：
  - `adapters/`
  - `data/tabular/`
  - `data/cv/`
  - `data/nlp/`
  - `data/timeseries/`
  - `data/graph/`
  - `results/`
  - `figures/`
- [x] 创建数据目录说明文件，避免误提交大体积原始数据
- [x] 确认 `.gitignore` 已忽略 `.env`、原始数据文件、模型权重、缓存文件
- [x] 确认远程服务器上的数据根目录为 `/root/autodl-tmp/final_project/data`

## 2. 数据下载

### 2.1 ADBench

- [x] 从 ADBench 镜像下载计划使用的 `.npz` 子集
- [x] 确认 `.npz` 数据文件位置：`/root/autodl-tmp/final_project/data/raw/ADBench`
- [x] 整理计划使用数据集：
  - 表格：Cardio、Thyroid、Satellite、Shuttle、Credit Card、Pima、Annthyroid、Mammography、Pendigits
  - CV：CIFAR10_0、CIFAR10_1、FashionMNIST_0、FashionMNIST_1
  - NLP：20news_0、20news_1、agnews_0、amazon
- [x] 记录原始文件路径和实际文件名

### 2.2 TSB-AD

- [x] 下载 TSB-AD 仓库：https://github.com/TheDatumOrg/TSB-AD
- [x] 下载完整数据：https://www.thedatum.org/datasets/TSB-AD-U.zip
- [x] 解压到服务器数据目录
- [x] 按领域分层选出 8 条代表性时序
- [x] 将选择结果写入 `data/timeseries/selected_files.txt`

### 2.3 GADBench

- [x] 下载或克隆 GADBench：https://github.com/squareRoot3/GADBench
- [x] 下载图数据：https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1
- [x] 确认可被 DGL `load_graphs` 读取
- [x] 整理计划使用图数据：T-Finance、Reddit、Amazon、Weibo

图数据当前目录：

- `/root/autodl-tmp/final_project/data/raw/GADBench/tfinance`
- `/root/autodl-tmp/final_project/data/raw/GADBench/reddit`
- `/root/autodl-tmp/final_project/data/raw/GADBench/amazon`
- `/root/autodl-tmp/final_project/data/raw/GADBench/weibo`

备注：`amazon` 使用数据自带 mask；`tfinance`、`reddit`、`weibo` 未包含 train/test mask，适配器会用固定 `seed=42` 生成分层节点 mask。

## 3. 数据适配器实现

### 3.1 ADBench 适配器

- [x] 新建 `adapters/adbench_adapter.py`
- [x] 支持读取 `.npz` 中的 `X` 和 `y`
- [x] 将 `X` 统一转为 `float64`
- [x] 对未标准化数据使用训练集统计量做标准化
- [x] 对已归一化数据支持跳过标准化
- [x] 实现分层 `train_test_split`
- [x] 输出标准格式 `(X_train, X_test, y_train, y_test)`

### 3.2 时序适配器

- [x] 新建 `adapters/timeseries_adapter.py`
- [x] 读取 TSB-AD `.csv`
- [x] 解析文件名中的 `tr_` 训练截止索引
- [x] 只用训练段统计量做 z-score 标准化
- [x] 实现可配置滑窗：`window_size`、`stride`
- [x] 明确并实现窗口标签对齐规则
- [x] 输出标准训练/测试窗口和标签

### 3.3 图适配器

- [x] 新建 `adapters/graph_adapter.py`
- [x] 使用 DGL `load_graphs` 加载图文件
- [x] 提取节点特征、标签、`train_mask`、`val_mask`、`test_mask`
- [x] 对节点特征做训练集 mask 统计量标准化
- [x] 保留图结构供图算法调用

## 4. 污染率注入器

- [x] 新建 `data/contaminate.py`
- [x] 实现无监督污染：
  - `0%`：训练集纯净
  - `5%`、`10%`、`20%`：按比例保留训练集异常样本
- [x] 实现有监督污染：
  - `5%`、`10%`、`20%`：随机样本标签 `0 <-> 1` 双向翻转
- [x] 所有污染逻辑固定 `seed=42`
- [x] 返回污染后的数据、标签和污染配置
- [x] 为污染函数补充最小自检脚本：`tests/smoke_data_pipeline.py`

## 5. 统一数据加载接口

- [x] 设计统一入口：`adapters/load_dataset.py`
- [x] 输入数据集名称、数据形态、随机种子、测试集比例、标准化配置
- [x] 输出 `(X_train, X_test, y_train, y_test)`
- [x] 对时序和图数据的额外返回对象写清楚说明
- [x] 确保算法负责人可以不关心底层数据来源直接调用

## 6. EDA 与数据血缘

- [x] 在服务器运行 `scripts/generate_eda_summary.py` 生成 `data/eda_summary.json`
- [x] 可选生成 `data/eda_summary.csv`
- [x] 每个数据集记录：
  - 数据集名称
  - 数据形态
  - 样本数或节点数
  - 特征维度
  - 异常样本数
  - 异常率
  - 特征值范围
  - dtype
  - 是否标准化
  - 训练/测试划分方式
  - 原始文件路径
- [x] 新建数据对接说明文档：`doc/data_integration_guide.md`
- [x] 新建数据血缘文档：`doc/data_lineage.md`
- [x] 为报告第 3 章准备素材

## 7. 验证与交付

- [x] 写一个最小 smoke test，能加载每类数据各 1 个样例
- [x] 检查训练/测试划分可复现
- [x] 检查标准化没有使用测试集统计量
- [x] 检查异常率和 PDF 中记录基本一致
- [x] 检查 Streamlit 可直接读取 `data/eda_summary.json`
- [x] 在服务器上跑通数据 pipeline
- [X] 将成员 A 交付物提交到 git

## 8. 推荐提交顺序

1. `docs: add data engineering requirements and todo`
2. `feat: add dataset directory structure`
3. `feat: add ADBench adapter`
4. `feat: add time-series adapter`
5. `feat: add graph adapter`
6. `feat: add contamination injector`
7. `feat: add EDA summary generation`
8. `docs: add data lineage documentation`
