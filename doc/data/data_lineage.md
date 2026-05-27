# 数据血缘与预处理记录

本文档记录成员 A 数据部分的来源、服务器路径、处理步骤和当前可交付状态。

## 1. 服务器数据根目录

```text
/root/autodl-tmp/final_project/data
```

当前已生成：

- `data/eda_summary.json`
- `data/timeseries/selected_files.txt`

## 2. ADBench

来源：

- GitHub：https://github.com/Minqi824/ADBench
- 下载镜像：`https://jihulab.com/BraudoCC/ADBench_datasets`

服务器路径：

```text
/root/autodl-tmp/final_project/data/raw/ADBench
```

已选数据：

- 表格：Cardio、Thyroid、Satellite、Shuttle、Credit Card、Pima、Annthyroid、Mammography、Pendigits
- CV：CIFAR10_0、CIFAR10_1、FashionMNIST_0、FashionMNIST_1
- NLP：20news_0、20news_1、agnews_0、amazon

文件映射：

- `cardio` -> `Classical/6_cardio.npz`
- `thyroid` -> `Classical/38_thyroid.npz`
- `satellite` -> `Classical/30_satellite.npz`
- `shuttle` -> `Classical/32_shuttle.npz`
- `credit_card` -> `Classical/13_fraud.npz`
- `pima` -> `Classical/29_Pima.npz`
- `annthyroid` -> `Classical/2_annthyroid.npz`
- `mammography` -> `Classical/23_mammography.npz`
- `pendigits` -> `Classical/28_pendigits.npz`
- `cifar10_0` -> `CV_by_ResNet18/CIFAR10_0.npz`
- `cifar10_1` -> `CV_by_ResNet18/CIFAR10_1.npz`
- `fashionmnist_0` -> `CV_by_ResNet18/FashionMNIST_0.npz`
- `fashionmnist_1` -> `CV_by_ResNet18/FashionMNIST_1.npz`
- `20news_0` -> `NLP_by_BERT/20news_0.npz`
- `20news_1` -> `NLP_by_BERT/20news_1.npz`
- `agnews_0` -> `NLP_by_BERT/agnews_0.npz`
- `amazon` -> `NLP_by_BERT/amazon.npz`

处理步骤：

- 读取 `.npz` 中的 `X` 和 `y`
- `X` 转为 `float64`
- 分层 `train_test_split`
- 默认使用训练集统计量做 z-score 标准化
- `credit_card/fraud` 默认跳过标准化

对接代码：

- `adapters/adbench_adapter.py`

## 3. TSB-AD

来源：

- GitHub：https://github.com/TheDatumOrg/TSB-AD
- 完整数据：https://www.thedatum.org/datasets/TSB-AD-U.zip

服务器路径：

```text
/root/autodl-tmp/final_project/data/raw/TSB-AD-U
```

处理步骤：

- 读取 `.csv`
- 解析文件名中的 `tr_` 索引作为训练/测试截止点
- 只使用训练段统计量做 z-score
- 默认滑窗参数：`window_size=100`，`stride=10`
- 默认窗口标签规则：窗口内任一点异常则窗口为异常

当前代表性时序：

详见：

```text
data/timeseries/selected_files.txt
```

对接代码：

- `adapters/timeseries_adapter.py`

## 4. GADBench

来源：

- GitHub：https://github.com/squareRoot3/GADBench
- Google Drive 数据包：`1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1`

服务器路径：

```text
/root/autodl-tmp/final_project/data/raw/GADBench
```

已整理图文件：

- `tfinance`
- `reddit`
- `amazon`
- `weibo`

处理步骤：

- 使用 DGL `load_graphs` 读取图文件
- 提取节点特征和标签
- 若文件自带 `train_mask/test_mask`，优先使用原始 mask
- 若缺少 mask，使用固定随机种子 `42` 生成分层节点划分
- 节点特征可按训练 mask 统计量标准化

当前划分情况：

- `amazon`：使用预设 mask
- `tfinance`、`reddit`、`weibo`：生成分层节点 mask

对接代码：

- `adapters/graph_adapter.py`

## 5. EDA 输出

生成命令：

```bash
python scripts/generate_eda_summary.py \
  --data-root /root/autodl-tmp/final_project/data \
  --output /root/autodl-tmp/final_project/data/eda_summary.json
```

当前状态：

- 共 30 行
- 状态包括：`ok`、`selected`、`available`
- 无 `missing_or_error`

对接文件：

- `data/eda_summary.json`
- `data/timeseries/selected_files.txt`

## 6. 污染注入

对接代码：

- `data/contaminate.py`

无监督污染：

- `contamination_rate=0.0`：训练集纯净
- `0.05/0.10/0.20`：控制训练集中保留的异常比例

有监督污染：

- `flip_rate=0.05/0.10/0.20`：对训练标签做 `0 <-> 1` 对称翻转

所有随机过程默认使用 `seed=42`。
