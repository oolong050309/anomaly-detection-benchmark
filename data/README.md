# 数据目录说明

本目录只提交轻量级项目产物，不提交原始 benchmark 大数据。

已提交文件：

- `contaminate.py`：Exp-2 训练集污染注入工具
- `eda_summary.json`：供 Streamlit 和报告使用的数据集统计摘要
- `eda_summary.csv`：便于人工查看的数据集统计表
- `timeseries/selected_files.txt`：选定的 TSB-AD 代表性时序文件

原始 benchmark 数据存放在远程服务器，禁止提交到 git：

```text
/root/autodl-tmp/final_project/data/raw/ADBench
/root/autodl-tmp/final_project/data/raw/TSB-AD-U
/root/autodl-tmp/final_project/data/raw/GADBench
```
