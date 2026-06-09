# 数据目录说明

本目录是项目**数据工程产物 + 可选本地原始文件**的根路径，与服务器 `AD_DATA_ROOT` 布局一致。

## 快速使用（推荐）

原始数据已在**服务器**就位并完成 adapter，算法 / 实验侧**直接 import**：

```python
from adapters import load_dataset

bundle = load_dataset("cardio", data_root="data")  # 或 AD_DATA_ROOT 指向服务器 data/
X_train, X_test, y_train, y_test = bundle.as_tuple()
```

完整对接说明：`doc/data/data_integration_guide.md`

## 已提交 git 的文件

| 文件 | 用途 |
|------|------|
| `eda_summary.json` / `eda_summary.csv` | 数据集统计（Streamlit、报告） |
| `selected_files.json` | 计划子集清单（29 项，对齐项目计划书 §2） |
| `timeseries/selected_files.txt` | TSB-AD 代表性时序路径列表 |
| `contaminate.py` | Exp-2 训练集污染注入 |

## 原始大文件（不入 git）

服务器完整路径：

```text
/root/autodl-tmp/final_project/data/raw/ADBench
/root/autodl-tmp/final_project/data/raw/TSB-AD-U
/root/autodl-tmp/final_project/data/raw/GADBench
```

本地可选镜像（适配器均支持）：

```text
data/tabular/   data/cv/   data/nlp/     ← ADBench .npz
data/timeseries/                            ← TSB-AD .csv（8 条见 selected_files.json）
data/graph/                                 ← GADBench 原始图（需 DGL）
data/graph_npz/                             ← 节点特征 NPZ（无 DGL 时 PCA / 通用算法）
```

本地仅演示 Streamlit 时：ADBench 可 `python -m scripts.fetch_demo_data --all-selected`；时序 / 图从服务器 `scp` 子集即可。

## 检查就位情况

```bash
python -m scripts.check_data_readiness
```
