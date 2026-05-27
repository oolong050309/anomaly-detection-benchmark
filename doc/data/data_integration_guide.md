# 数据对接说明

本文档说明成员 A 提供的数据适配器、污染注入器和 EDA 输出如何对接算法、评估和前端模块。

## 1. 统一数据入口

推荐优先使用：

```python
from adapters import load_dataset

bundle = load_dataset("cardio", data_root="data")
X_train, X_test, y_train, y_test = bundle.as_tuple()
```

`load_dataset()` 会根据数据集名称或显式 `modality` 自动路由到：

- `adapters/adbench_adapter.py`
- `adapters/timeseries_adapter.py`
- `adapters/graph_adapter.py`

返回对象是 `DatasetBundle`，其中：

- `X_train`, `X_test`, `y_train`, `y_test`：算法侧最常用的四元组
- `extras`：图结构、原始序列、mask 等补充信息
- `metadata`：来源、路径、切分方式、标准化信息、EDA 统计

## 2. ADBench 数据

示例：

```python
bundle = load_dataset("credit_card", data_root="data")
```

约定：

- 输入 `name` 使用小写或常见别名即可
- 输出特征为 `float64`
- 训练/测试划分为分层抽样
- 默认对未归一化数据做训练集统计标准化

可用名称包括：

- `cardio`
- `thyroid`
- `satellite`
- `shuttle`
- `credit_card` / `fraud`
- `pima`
- `annthyroid`
- `mammography`
- `pendigits`
- `cifar10_0`
- `cifar10_1`
- `fashionmnist_0`
- `fashionmnist_1`
- `20news_0`
- `20news_1`
- `agnews_0`
- `amazon`

## 3. 时序数据

示例：

```python
bundle = load_dataset("003_NAB_id_3_WebService_tr_1362_1st_1462.csv", modality="timeseries", data_root="data")
```

约定：

- 文件名中的 `tr_` 用作训练截止点
- 仅使用训练段统计量做 z-score
- 默认按滑窗切分
- 窗口标签规则默认 `any`

如果想直接挑代表性文件，可先读：

```python
from adapters.timeseries_adapter import select_representative_tsb_files
files = select_representative_tsb_files(data_root="data")
```

并把结果写入 `data/timeseries/selected_files.txt`。

## 4. 图数据

示例：

```python
bundle = load_dataset("reddit", modality="graph", data_root="data")
graph = bundle.extras["graph"]
masks = bundle.extras["masks"]
```

约定：

- 依赖 DGL 的 `load_graphs`
- 优先读取节点特征、标签、`train_mask`、`val_mask`、`test_mask`
- 节点特征默认按训练 mask 标准化
- 图算法可直接使用 `extras["graph"]`

当前说明：

- GADBench 源码仓库已下载
- 图数据包尚未成功下载，因此图加载器会在缺文件时明确报错

## 5. 污染注入

无监督污染：

```python
from data.contaminate import contaminate_unsupervised
Xc, yc, meta = contaminate_unsupervised(X_train, y_train, contamination_rate=0.05, seed=42)
```

有监督污染：

```python
from data.contaminate import contaminate_supervised
Xc, yc, meta = contaminate_supervised(X_train, y_train, flip_rate=0.1, seed=42)
```

约定：

- 无监督污染用于 Exp-2 的无监督算法
- 有监督污染用于 Exp-2 的监督算法
- 默认固定种子 `42`

## 6. EDA 输出

生成脚本：

```bash
python scripts/generate_eda_summary.py --data-root data --output data/eda_summary.json
```

输出内容：

- 每个数据集的样本数、维度、异常率、dtype、值范围
- 数据来源、切分方式和标准化策略
- `TSB-AD` 的代表性文件清单会自动写入 `data/timeseries/selected_files.txt`
- `GADBench` 目前会标记为 `blocked`

## 7. 算法侧接入建议

算法模块只需要依赖四元组：

```python
X_train, X_test, y_train, y_test = bundle.as_tuple()
```

如果是图模型或时序模型，再从 `bundle.extras` 里取额外对象，避免把不同模态的特殊字段塞进统一接口里。
