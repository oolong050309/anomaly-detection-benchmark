# 数据对接说明

本文档说明成员 A 提供的数据适配器、污染注入器和 EDA 输出如何对接算法、评估和前端模块。

**当前状态（2026-05）**：三类 benchmark 原始数据已下载至**服务器**，适配器与集成测试已通过（`doc/data/test_report.md`：服务器 **10 passed**）。算法 / 实验 / Streamlit **无需自行解析原始文件**，统一 `import` 即可。

详细路径与血缘见：`doc/data/data_lineage.md` · 计划子集清单见：`data/selected_files.json`

---

## 0. 数据根目录与运行环境

### 服务器（完整原始数据，推荐跑实验）

```text
AD_DATA_ROOT=/root/autodl-tmp/final_project/data
```

原始文件布局：

```text
data/raw/ADBench/          # 表格 + CV + NLP（.npz）
data/raw/TSB-AD-U/         # 时序（.csv，含 TSB-AD-U 子目录）
data/raw/GADBench/         # 图（tfinance / reddit / amazon / weibo）
```

### 本地 Windows（演示 / 轻量开发）

- **元数据**（已入 git）：`data/eda_summary.json`、`data/selected_files.json`
- **原始大文件**（可选）：`data/tabular/`、`data/cv/`、`data/nlp/`、`data/timeseries/`、`data/graph/` 或 `data/graph_npz/`
- ADBench 17 个 `.npz` 可用 `python -m scripts.fetch_demo_data --all-selected` 从 GitHub 拉取；时序 / 图建议 `scp` 服务器子集（见 `doc/streamlit_test_guide.md` §7.4）

适配器会**自动**在扁平子目录与 `data/raw/*` 之间查找，无需改调用代码。

---

## 1. 统一数据入口

```python
from adapters import load_dataset

# 表格（默认路由 ADBench）
bundle = load_dataset("cardio", data_root="data")
X_train, X_test, y_train, y_test = bundle.as_tuple()

# 显式指定形态
bundle = load_dataset("reddit", modality="graph", data_root="data")
bundle = load_dataset("006_NAB_id_6_Traffic", modality="timeseries", data_root="data")
```

环境变量（与 `data_root` 参数二选一，参数优先）：

```bash
export AD_DATA_ROOT=/root/autodl-tmp/final_project/data   # Linux 服务器
# Windows PowerShell:
# $env:AD_DATA_ROOT = "D:\destop\anomaly-detection-benchmark\data"
```

`load_dataset()` 自动路由到：

| 模态 | 适配器 |
|------|--------|
| tabular / cv / nlp | `adapters/adbench_adapter.py` |
| timeseries | `adapters/timeseries_adapter.py` |
| graph | `adapters/graph_adapter.py` |

返回 `DatasetBundle`：

| 字段 | 含义 |
|------|------|
| `X_train`, `X_test`, `y_train`, `y_test` | 算法侧标准四元组 |
| `extras` | 图对象、mask、原始序列等 |
| `metadata` | 来源、路径、切分、标准化、EDA 统计 |

---

## 2. ADBench（表格 + CV + NLP，共 17 个文件）

```python
bundle = load_dataset("credit_card", data_root="data")
print(bundle.metadata["source"])   # ADBench
print(bundle.metadata["n_samples"])
```

**可用 `name`（与 `exp1` / `eda_summary` 一致）**：

| 形态 | 名称 |
|------|------|
| 表格 | `cardio`, `thyroid`, `satellite`, `shuttle`, `credit_card`/`fraud`, `pima`, `annthyroid`, `mammography`, `pendigits` |
| CV | `cifar10_0`, `cifar10_1`, `fashionmnist_0`, `fashionmnist_1` |
| NLP | `20news_0`, `20news_1`, `agnews_0`, `amazon` |

约定：分层 7:3 划分 · 训练集 z-score（`credit_card`/`fraud` 默认跳过）· 特征 `float64`

---

## 3. TSB-AD（时序，8 条计划子集）

实验脚本使用**短名**（与 CSV 文件名前缀一致）：

```python
for ds in [
    "006_NAB_id_6_Traffic",
    "149_Stock_id_1_Finance",
    "171_MITDB_id_2_Medical",
    "225_MGAB_id_1_Synthetic",
    "276_IOPS_id_17_WebService",
    "331_UCR_id_29_Facility",
    "337_UCR_id_35_HumanActivity",
    "550_SWaT_id_1_Sensor",
]:
    bundle = load_dataset(ds, modality="timeseries", data_root="data")
```

也可传入完整 CSV 文件名；适配器在 `data/raw/TSB-AD-U` 或 `data/timeseries/` 下 `rglob` 匹配。

约定：文件名 `tr_<n>` 为训练截止 · 训练段 z-score · 默认 `window_size=100, stride=10`

列出代表性文件：

```python
from adapters.timeseries_adapter import select_representative_tsb_files
files = select_representative_tsb_files(data_root="data")
```

清单文件：`data/timeseries/selected_files.txt`、`data/selected_files.json` → `"timeseries"`

---

## 4. GADBench（图，4 个数据集）

```python
bundle = load_dataset("reddit", modality="graph", data_root="data")
graph = bundle.extras["graph"]      # DGL 图对象（需 DGL 环境）
masks = bundle.extras["masks"]      # train / val / test mask
X_train, X_test, y_train, y_test = bundle.as_tuple()  # 节点特征矩阵
```

**计划子集**：`tfinance`, `reddit`, `amazon`, `weibo`（见 `data/selected_files.json` → `"graph"`）

| 数据集 | mask 来源 |
|--------|-----------|
| `amazon` | 数据自带 `predefined_masks` |
| `tfinance`, `reddit`, `weibo` | 适配器 `seed=42` 分层生成 |

**无 DGL 时的 NPZ 回退**（通用算法 / Streamlit PCA）：

```bash
# 在服务器执行
python -m scripts.export_graph_npz --out-dir data/graph_npz
```

本地将 `data/graph_npz/{name}.npz` 就位后，`load_dataset(name, modality="graph")` 自动读取节点特征四元组。

---

## 5. 污染注入（Exp-2）

```python
from data.contaminate import contaminate_unsupervised, contaminate_supervised

Xc, yc, meta = contaminate_unsupervised(X_train, y_train, contamination_rate=0.05, seed=42)
Xc, yc, meta = contaminate_supervised(X_train, y_train, flip_rate=0.10, seed=42)
```

无监督 → 控制训练集异常保留比例；有监督 → 对称标签翻转。默认 `seed=42`。

---

## 6. EDA 与前端契约

生成 / 刷新统计（在**有原始数据**的机器上）：

```bash
python scripts/generate_eda_summary.py \
  --data-root /root/autodl-tmp/final_project/data \
  --output data/eda_summary.json
```

输出：

- `data/eda_summary.json` — Streamlit 数据集浏览、报告引用
- `data/eda_summary.csv` — 人工查阅
- `data/timeseries/selected_files.txt` — 时序代表性清单

**说明**：仓库中的 `eda_summary.json` 已包含 ADBench 17 项 + 图 4 项的完整 `ok` 统计；时序 8 项在服务器跑实验前建议重跑上述脚本以补齐 `status=ok` 行（当前部分为 `selected` 占位）。

---

## 7. 算法 / 实验侧接入

```python
X_train, X_test, y_train, y_test = bundle.as_tuple()
detector.fit(X_train, y_train)          # 监督算法
scores = detector.decision_function(X_test)
```

图专用 / 时序专用算法从 `bundle.extras` 取图对象或原始序列，不要把模态特有字段塞进统一四元组。

**回归检查**（服务器）：

```bash
AD_DATA_ROOT=/root/autodl-tmp/final_project/data python -m pytest tests/test_data_adapters_integration.py -vv
python -m scripts.check_data_readiness   # 对照 selected_files 与本地文件
```

---

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| `doc/data/data_lineage.md` | 来源、服务器路径、预处理步骤 |
| `doc/data/test_report.md` | 集成测试范围与结果 |
| `doc/data/data_download_and_processing_requirements.md` | 成员 A 原始需求 |
| `data/selected_files.json` | 计划子集文件清单（29 项） |
| `doc/streamlit_test_guide.md` | 前端演示与本地缺数据时的回退 |
