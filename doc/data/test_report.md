# 数据模块测试报告

本文档记录成员 A 数据工程模块的测试范围、运行方式和最近一次测试结果。

## 1. 测试文件

```text
tests/
  smoke_data_pipeline.py
  test_contamination_and_eda.py
  test_data_adapters_integration.py
```

## 2. 测试覆盖范围

### 本地轻量测试

`tests/test_contamination_and_eda.py`

覆盖：

- 无监督污染率为 `0` 时只保留正常样本
- 无监督污染率接近目标比例
- 有监督标签翻转数量准确
- 同一 `seed` 下标签翻转可复现
- `data/eda_summary.json` 可解析
- EDA 行数为 30
- EDA 中不存在 `missing_or_error`
- GADBench 4 个图的划分策略记录正确

### 真实数据集成测试

`tests/test_data_adapters_integration.py`

需要设置：

```bash
AD_DATA_ROOT=/root/autodl-tmp/final_project/data
```

覆盖：

- ADBench 计划使用的 17 个数据集全部可加载
- TSB-AD 至少 1 条代表性时序可完成标准化和滑窗切分
- GADBench 4 个计划图均可被 DGL 读取
- 统一四元组 `(X_train, X_test, y_train, y_test)` 形状一致
- 标签只包含 `0/1`
- GADBench mask 来源符合预期：
  - `amazon`：`predefined_masks`
  - `tfinance`、`reddit`、`weibo`：`generated_stratified_node_masks`

## 3. 运行命令

本地运行：

```bash
python -m pytest tests -q
```

本地没有原始数据时，真实数据集成测试会自动跳过。

服务器运行：

```bash
cd /root/autodl-tmp/final_project
AD_DATA_ROOT=/root/autodl-tmp/final_project/data python -m pytest tests -vv --tb=short
```

## 4. 最近测试结果

本地：

```text
4 passed, 6 skipped
```

服务器：

```text
10 passed in 20.17s
```

服务器测试时间：2026-05-27。

## 5. 结论

当前测试已经覆盖成员 A 数据模块的主要交付风险：

- 数据可加载
- 划分可复现
- 污染注入可用
- EDA 输出满足前端和报告契约
- 图数据 DGL 读取可用

后续成员 B 编写算法和实验脚本时，可以把这些测试作为数据接口是否被破坏的回归检查。
