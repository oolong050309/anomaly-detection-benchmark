# 异常检测算法系统性对比研究

跨 **5 类数据形态**（表格 / CV / NLP / 时序 / 图），在统一框架下系统对比 **26 种**异常检测算法，以**训练集污染鲁棒性**为核心问题，辅以跨模态泛化、精度-效率权衡与标签翻转防御消融。数据挖掘课程期末项目（选项 B：算法系统性对比研究）。

**小组成员**：徐云鹏（2353583）· 杨景翔（2351576）· 陈艺龙（2352359）· 李雪菲（2354093）

---

## 项目概览

| 维度 | 规模 |
|------|------|
| 算法 | 26 种（统计 / 近邻 / 树 / 边界 / 监督 / 深度 / 时序 / 图） |
| 数据形态 | 5 类 |
| 数据集 | 28+（ADBench 17 + TSB-AD 8 + GADBench 4） |
| 随机种子 | {41, 42, 43} |
| 核心实验 | Exp-1 基线 · Exp-2 污染 · Exp-3 跨模态 · Exp-4 防御 |
| 主推指标 | **AUC-PR**（极度不平衡场景）；辅以 AUC-ROC、F1@best |

### 三个核心研究问题

1. **Exp-1 基线对比**：干净数据下，各算法在其适配模态内精度如何排名？深度方法是否优于经典方法？
2. **Exp-2 污染鲁棒性**：污染率 {0, 1, 5, 10, 20}% 时，无监督（保留异常比例）与有监督（对称标签翻转）的退化规律有何差异？
3. **Exp-3 跨模态泛化**：同一通用算法在 5 类形态上的排名是否稳定？是否存在「跨模态通吃」的单一算法？

**扩展 · Exp-4**：6 基模型 × 8 去噪器 × Trim/Flip × 4 档翻转率，评估对称标签翻转下的主动防御策略（2000+ 条记录）。

### 工程决策摘要

| 场景 | 推荐方向 |
|------|----------|
| 干净表格 + 有标签 | XGBoost / LightGBM |
| 干净表格 + 无标签 | IForest / ECOD / COPOD |
| 训练集可能被污染 | 优先无监督浅层法；有监督需配合防御 |
| 跨模态部署 | 按模态分别选型，勿迷信单一 SOTA |
| 高噪标签翻转（Exp-4） | IQR_Trim / IForest_Trim + 树模型；Trim 普遍优于 Flip |

---

## 数据来源

三大 NeurIPS 级 benchmark，经 `adapters.load_dataset()` 统一为 `(X_train, X_test, y_train, y_test)`：

| Benchmark | 会议 | 覆盖模态 | 本项目子集 |
|-----------|------|----------|------------|
| [ADBench](https://github.com/Minqi824/ADBench) | NeurIPS 2022 | 表格 + CV + NLP | 17 个 |
| [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) | NeurIPS 2024 | 时序 | 8 条 |
| [GADBench](https://github.com/squareRoot3/GADBench) | NeurIPS 2023 | 图 | 4 个 |

- 计划子集清单：`data/selected_files.json`、`data/timeseries/selected_files.txt`
- EDA 摘要：`data/eda_summary.json` / `data/eda_summary.csv`
- 数据工程说明：[`data/README.md`](data/README.md)、[`doc/data/data_integration_guide.md`](doc/data/data_integration_guide.md)

---

## 算法清单（26）

| 类别 | 算法 | 适配模态 |
|------|------|----------|
| 统计 / 浅层集成 | IQR、ECOD、COPOD | 通用 |
| 近邻 / 树 | KNN、LOF、Isolation Forest | 通用 |
| 边界 | OCSVM | 通用 |
| 监督 | LR、RF、MLP、XGBoost、LightGBM、TabPFN | 通用（有标签） |
| 深度 | AutoEncoder、Deep SVDD | 表格 / CV / NLP / 图 |
| 时序专用 | MatrixProfile、MiniRocket、LSTM-AE、LSTM-Sup、DADA | 时序 |
| 图专用 | DOMINANT、CoLA、GCN、BWGNN、XGBGraph、UNPrompt | 图 |

统一接口：`models.base.BaseDetector`，实验侧通过 `fit` / `decision_function` 调用，日志写入 `results/exp*_results.csv`。

---

## 仓库结构

```text
anomaly-detection-benchmark/
├── adapters/              # 数据适配层（ADBench / TSB-AD / GADBench）
│   ├── load_dataset.py    # 统一入口 load_dataset()
│   ├── adbench_adapter.py
│   ├── timeseries_adapter.py
│   └── graph_adapter.py
├── data/
│   ├── contaminate.py     # Exp-2 污染注入（无监督保留 / 有监督翻转）
│   ├── eda_summary.json     # 数据集 EDA（Streamlit / 报告）
│   ├── selected_files.json
│   └── tabular|cv|nlp|timeseries|graph/   # 本地可选原始数据
├── models/                # 26 算法封装 + device / defense
├── experiments/           # exp1_baseline / exp2_contamination / exp3_cross_modal / exp4_defense
├── eval/                  # metrics / visualize / significance / analysis_utils
├── results/               # exp*_results.csv + artifacts/
├── figures/               # analyze_results.py 生成的图表
├── app/
│   ├── streamlit_app.py   # 七页交互演示
│   └── utils.py
├── scripts/               # 数据拉取、冒烟测试、环境检查
├── doc/                   # 答辩大纲、Streamlit 指南、数据文档
├── report/                # LaTeX 报告
├── run_all.py             # 一键跑实验（支持 --analyze）
├── analyze_results.py     # 从 CSV 生成全部图表
├── 项目计划书.md
└── 团队分工文档.md
```

---

## 快速开始

### 1. 仅浏览实验结果（推荐本地）

无需全量原始数据；仓库已含 `results/*.csv` 与 `figures/`，可直接启动 Streamlit：

```bash
pip install -r requirements-streamlit.txt   # 若环境已有 pandas/sklearn/matplotlib 可跳过
streamlit run app/streamlit_app.py
```

浏览器打开后可在侧边栏配置 `results/`、`figures/`、`data/` 路径。

### 2. 拉取演示数据（可选）

为「数据集浏览」页的 PCA / 特征分布提供本地 ADBench 子集：

```bash
python -m scripts.fetch_demo_data --all-selected
python -m scripts.check_data_readiness
```

### 3. 服务器全量复现

完整实验依赖 PyOD、XGBoost、DGL/PyG 等，建议在 Linux GPU 服务器安装：

```bash
bash scripts/setup_server.sh          # PyTorch + DGL + PyG（CUDA 11.8）
pip install -r requirements.txt -r requirements-dev.txt
```

原始数据置于 `AD_DATA_ROOT`（不入 git），典型布局见 [`data/README.md`](data/README.md)。

---

## 统一数据加载

算法与实验侧**只应**通过适配层取数：

```python
from adapters import load_dataset

bundle = load_dataset("cardio", data_root="data")
X_train, X_test, y_train, y_test = bundle.as_tuple()
```

图数据如需 DGL 图对象：

```python
bundle = load_dataset("reddit", modality="graph", data_root="data")
graph = bundle.extras["graph"]
masks = bundle.extras["masks"]
```

### 污染注入（Exp-2）

```python
from data.contaminate import contaminate_unsupervised, contaminate_supervised

Xc, yc, meta = contaminate_unsupervised(X_train, y_train, contamination_rate=0.05, seed=42)
Xc, yc, meta = contaminate_supervised(X_train, y_train, flip_rate=0.10, seed=42)
```

---

## 运行实验

```bash
# 四组实验 × 三 seed，结束后自动生成图表
python run_all.py --exp all --data-root data --output-dir results --seeds 41 42 43 --analyze

# 单独运行
python run_all.py --exp exp1 --modality all
python run_all.py --exp exp2 --modalities tabular timeseries graph
python run_all.py --exp exp3 --modalities tabular cv nlp timeseries graph
python run_all.py --exp exp4
```

每组实验产出：

- `results/exp*_results.csv`：指标、耗时、模态、污染率、参数量、失败原因、artifact 路径
- `results/artifacts/<exp>/*.npz`：`y_true`、`scores` 等，供 ROC/PR 与错误分析
- `results/artifacts/<exp>/*.json`：run 元数据

### 冒烟测试（小样本）

```bash
python -m scripts.smoke_test_experiments \
  --data-root data \
  --output-dir results/smoke_cpu \
  --modalities tabular timeseries graph \
  --max-train 160 --max-test 120 --run-analysis
```


---

## 生成评估图表

```bash
python analyze_results.py --results-dir results --figures-dir figures --metric auc_pr
```

| 目录 | 内容 |
|------|------|
| `figures/exp1/` | 热力图、平均排名、ROC、精度-效率权衡、Friedman/Nemenyi |
| `figures/exp2/` | 退化曲线、鲁棒性排名、跨模态 AUC 下降热力图 |
| `figures/exp3/` | 算法×形态热力图、雷达图、形态难度 |
| `figures/exp4/` | 防御对比（需先跑 Exp-4；`python analyze_results.py --section exp4`） |
| `figures/error_analysis/` | 分数分布、FP/FN 案例 |

---

## Streamlit 交互界面

七页导航，与四组实验及计划书指标对齐：

| 页面 | 功能 |
|------|------|
| 首页 | 四实验状态、核心研究问题、数据覆盖总览 |
| 数据集浏览 | EDA、异常率、特征分布与 PCA（需本地原始数据） |
| 基准对比 (Exp-1) | 按模态/数据集/指标的柱状图与 ROC |
| 污染率分析 (Exp-2) | 污染率滑块 {0,1,5,10,20}%，无监督/有监督机制切换 |
| 跨模态对比 (Exp-3) | 热力图、雷达图、形态难度 |
| 鲁棒防御 (Exp-4) | Trim vs Flip 曲线、20% 消融柱图 |
| 精度-效率 | AUC vs 耗时/参数量散点图 |

**环境变量**

| 变量 | 含义 | 默认 |
|------|------|------|
| `AD_DATA_ROOT` | 原始数据根目录 | `./data` |
| `AD_RESULTS_DIR` | 实验 CSV 目录 | `./results` |
| `AD_FIGURES_DIR` | 预生成图表目录 | `./figures` |
| `AD_DEVICE` | `auto` / `cpu` / `cuda` | `auto` |

---

## 测试

```bash
python -m pytest tests -q
```

本地无原始数据时，集成测试会自动跳过。服务器全量数据：

```bash
AD_DATA_ROOT=/path/to/data python -m pytest tests -vv --tb=short
```

详见 [`doc/data/test_report.md`](doc/data/test_report.md)。

---

## 团队分工

| 角色 | 负责人 | 核心职责 | 主要产出 |
|------|--------|----------|----------|
| 数据工程 | 徐云鹏 | Benchmark 获取、统一适配层、污染注入、EDA | `adapters/`、`data/contaminate.py`、`eda_summary.*` |
| 算法建模 | 杨景翔 | 26 算法封装、Exp-1~4 脚本、多 seed 调度 | `models/`、`experiments/`、`results/*.csv` |
| 评估分析 | 陈艺龙 | 指标、显著性检验、可视化、错误分析 | `eval/`、`figures/`、`analyze_results.py` |
| 工程交付 | 李雪菲 | Pipeline、Streamlit、文档与答辩材料 | `run_all.py`、`app/`、`README`、[`doc/ppt/`](doc/ppt/) |


---

## 引用

若在报告中引用本 benchmark 组合，请同时引用原始数据集论文：

- ADBench: Han et al., NeurIPS 2022
- TSB-AD: Liu et al., NeurIPS 2024
- GADBench: Tang et al., NeurIPS 2023

---

## License

见 [LICENSE](LICENSE)。
