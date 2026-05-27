# anomaly-detection-benchmark

跨模态异常检测算法对比研究：在表格、图像、文本、时序、图 5 类数据上，系统对比算法在训练集污染下的鲁棒性差异。数据挖掘课程期末项目。

## 当前交付状态

成员 A 的数据工程部分已完成到可对接状态：

- ADBench 计划子集已下载并可通过适配器加载
- TSB-AD-U 已下载、解压，并选出 8 条代表性时序
- GADBench 图数据已上传服务器，4 个计划图均可读取
- 已生成 `data/eda_summary.json` 和 `data/eda_summary.csv`
- 已实现统一数据加载入口、污染注入器和 smoke test

## 目录说明

```text
adapters/                 数据适配器
  adbench_adapter.py       ADBench 表格/CV/NLP 数据加载
  timeseries_adapter.py    TSB-AD 时序数据加载和滑窗切分
  graph_adapter.py         GADBench 图数据加载
  load_dataset.py          统一数据加载入口
data/
  contaminate.py           训练集污染注入工具
  eda_summary.json         数据集 EDA 摘要
  eda_summary.csv          数据集 EDA 表格
  timeseries/
    selected_files.txt     8 条代表性时序清单
scripts/
  generate_eda_summary.py  生成 EDA 摘要
tests/
  smoke_data_pipeline.py   数据管道轻量自检
doc/
  data_integration_guide.md              算法/前端对接说明
  data_lineage.md                        数据血缘与预处理记录
  data_download_and_processing_requirements.md
```

## 快速使用

算法侧推荐只使用统一入口：

```python
from adapters import load_dataset

bundle = load_dataset("cardio", data_root="data")
X_train, X_test, y_train, y_test = bundle.as_tuple()
```

图数据如需原始 DGL 图对象：

```python
bundle = load_dataset("reddit", modality="graph", data_root="data")
graph = bundle.extras["graph"]
masks = bundle.extras["masks"]
```

## 污染注入

无监督污染：

```python
from data.contaminate import contaminate_unsupervised

Xc, yc, meta = contaminate_unsupervised(X_train, y_train, contamination_rate=0.05, seed=42)
```

有监督标签翻转：

```python
from data.contaminate import contaminate_supervised

Xc, yc, meta = contaminate_supervised(X_train, y_train, flip_rate=0.1, seed=42)
```

## 生成 EDA 摘要

在服务器上运行：

```bash
python scripts/generate_eda_summary.py \
  --data-root /root/autodl-tmp/final_project/data \
  --output /root/autodl-tmp/final_project/data/eda_summary.json
```

本地仓库已包含最近一次生成的：

- `data/eda_summary.json`
- `data/eda_summary.csv`
- `data/timeseries/selected_files.txt`

## 测试

本地轻量测试：

```bash
python -m pytest tests -q
```

如果本地没有原始数据，真实数据集成测试会自动跳过。

服务器真实数据测试：

```bash
cd /root/autodl-tmp/final_project
AD_DATA_ROOT=/root/autodl-tmp/final_project/data python -m pytest tests -vv --tb=short
```

最近一次结果：

```text
本地：4 passed, 6 skipped
服务器：10 passed in 20.17s
```

详细测试说明见 `doc/test_report.md`。

## 运行三组实验

服务器数据准备好后，在仓库根目录运行：

```bash
python run_all.py \
  --exp all \
  --data-root /root/autodl-tmp/final_project/data \
  --output-dir results/server_run_seed42 \
  --seed 42
```

也可以单独运行：

```bash
python run_all.py --exp exp1 --modality all
python run_all.py --exp exp2 --modalities tabular timeseries graph
python run_all.py --exp exp3 --modalities tabular cv nlp timeseries graph
```

每组实验会写出：

- `exp1_results.csv` / `exp2_results.csv` / `exp3_results.csv`：指标、耗时、模态、污染率、样本规模、参数、失败原因、artifact 路径。
- `artifacts/<exp>/*.npz`：每次运行的 `y_true`、`scores`、必要时的 `y_train` / `test_index`。
- `artifacts/<exp>/*.json`：run_id、参数、指标、污染元数据和补充说明。

这些产物足够支持后续 ROC/PR 曲线、污染退化曲线、跨模态排名、错误分析和显著性检验；只有换算法、换数据划分或换超参数时才需要重跑。

## 服务器数据位置

原始数据不提交到 git，统一放在服务器：

```text
/root/autodl-tmp/final_project/data/raw/ADBench
/root/autodl-tmp/final_project/data/raw/TSB-AD-U
/root/autodl-tmp/final_project/data/raw/GADBench
```

## 注意事项

- `.env` 包含服务器密码，禁止提交
- 原始数据、压缩包、模型权重禁止提交
- 所有随机过程默认使用 `seed=42`
- 标准化只使用训练集统计量，避免测试集信息泄漏
- GADBench 中 `amazon` 使用预设 mask；`tfinance`、`reddit`、`weibo` 使用固定种子生成分层节点 mask
