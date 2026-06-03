"""Detector parameter-count introspection.

每个 detector 训练完后调用 ``count_parameters(detector)`` 拿到精确参数量。
用于汇总耗时与"模型大小"对照，以及散点图点大小映射。

支持矩阵
--------
- PyTorch nn.Module（自家深度模型 + DADA + GNN）：``sum(p.numel() for p in m.parameters())``
- sklearn Linear（LogisticRegression / MLPClassifier）：``coef_/intercept_`` size 求和
- sklearn 树集成（RandomForestClassifier / IsolationForest）：每棵树的 ``node_count`` 求和
- XGBoost：``get_booster().trees_to_dataframe()`` 的行数（叶子+分裂节点）
- LightGBM：``booster_.dump_model()`` 里所有 tree 的 ``num_leaves`` 求和
- OCSVM (PyOD)：``support_vectors_.size``
- TabPFN：内部 ``model_`` / ``model`` 是 nn.Module，按 PyTorch 路径
- MiniRocket：``num_kernels * 9``（每核 9 维 ±1 卷积权重 + dilation 索引）
- 无可训练参数（IQR / ECOD / COPOD / KNN / LOF / MatrixProfile）：返回 ``0``

返回值约定
----------
``count_parameters`` 返回 ``int`` 或 ``None``：
- 拿到精确数 → ``int``（可能是 0，例如 IQR）
- 推断不出来 → ``None``（汇总脚本会回退到 heuristic proxy）
"""

from __future__ import annotations

from typing import Any

# 已知"无可训练参数"算法名（小写匹配子串）
_PARAMETERLESS_ALGOS = (
    "iqr", "ecod", "copod", "knn", "lof", "matrixprofile", "matrix_profile",
)


def _torch_param_count(module: Any) -> int | None:
    """对一个 PyTorch ``nn.Module`` 求总参数量。"""
    try:
        params = list(module.parameters())
    except Exception:
        return None
    try:
        return int(sum(p.numel() for p in params))
    except Exception:
        return None


def _sklearn_linear_count(model: Any) -> int | None:
    """LogisticRegression / linear models with ``coef_`` & ``intercept_``."""
    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if coef is None:
        return None
    try:
        n = int(coef.size)
        if intercept is not None:
            n += int(getattr(intercept, "size", len(intercept)))
        return n
    except Exception:
        return None


def _sklearn_mlp_count(model: Any) -> int | None:
    """MLPClassifier: 各层 weight + bias size 求和。"""
    coefs = getattr(model, "coefs_", None)
    biases = getattr(model, "intercepts_", None)
    if coefs is None:
        return None
    try:
        total = sum(int(c.size) for c in coefs)
        if biases is not None:
            total += sum(int(b.size) for b in biases)
        return total
    except Exception:
        return None


def _sklearn_tree_ensemble_count(model: Any) -> int | None:
    """RandomForestClassifier / IsolationForest / ExtraTreesClassifier：
    所有树 node_count 之和（一个 split-node 大致等价 1 个 "参数"）。"""
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        return None
    try:
        total = 0
        for est in estimators:
            tree = getattr(est, "tree_", None)
            if tree is not None and hasattr(tree, "node_count"):
                total += int(tree.node_count)
        return int(total) if total > 0 else None
    except Exception:
        return None


def _xgboost_count(model: Any) -> int | None:
    """XGBClassifier: booster trees_to_dataframe 行数（叶子+分裂节点总数）。"""
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is None:
            return None
        df = booster.trees_to_dataframe()
        return int(df.shape[0])
    except Exception:
        return None


def _lightgbm_count(model: Any) -> int | None:
    """LGBMClassifier: 每棵树的 num_leaves 之和（叶子节点总数；split 数 = 叶子-1）。"""
    try:
        booster = getattr(model, "booster_", None)
        if booster is None:
            return None
        dump = booster.dump_model()
        trees = dump.get("tree_info", [])
        return int(sum(int(t.get("num_leaves", 0)) for t in trees))
    except Exception:
        return None


def _ocsvm_count(model: Any) -> int | None:
    """OCSVM: support_vectors_.size = n_support × n_features。

    PyOD 包裹了 sklearn ``OneClassSVM``：分支：
    - PyOD ``OCSVM`` → 真正的 estimator 在 ``model.detector_``
    - sklearn ``OneClassSVM`` → 直接在 ``model``
    """
    inner = getattr(model, "detector_", model)
    sv = getattr(inner, "support_vectors_", None)
    if sv is None:
        return None
    try:
        return int(sv.size)
    except Exception:
        return None


def _iforest_pyod_count(model: Any) -> int | None:
    """PyOD IForest 包了 sklearn IsolationForest，真正 estimator 在 ``detector_``。"""
    inner = getattr(model, "detector_", None) or model
    return _sklearn_tree_ensemble_count(inner)


def _tabpfn_count(model: Any) -> int | None:
    """TabPFN: 内部 nn.Module 通常在 ``model_`` 或 ``model``。"""
    for attr in ("model_", "model", "_model"):
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        if hasattr(inner, "parameters"):
            n = _torch_param_count(inner)
            if n is not None:
                return n
    return None


def _minirocket_count(detector: Any, fit_params: dict[str, Any] | None) -> int | None:
    """MiniRocket: num_kernels * 9（每核 9 维卷积）+ num_kernels (bias)。"""
    n_kernels = None
    if fit_params and "num_kernels" in fit_params:
        n_kernels = fit_params["num_kernels"]
    elif hasattr(detector, "num_kernels"):
        n_kernels = getattr(detector, "num_kernels", None)
    if n_kernels is None:
        return None
    try:
        return int(int(n_kernels) * 9 + int(n_kernels))
    except Exception:
        return None


def _algorithm_name_hint(detector: Any) -> str:
    """detector 类名的小写形式，用于关键字匹配。"""
    return type(detector).__name__.lower()


def count_parameters(
    detector: Any,
    *,
    fit_params: dict[str, Any] | None = None,
) -> int | None:
    """返回 ``detector`` 训练后的真实参数量；推断不出来返回 ``None``。

    Parameters
    ----------
    detector
        任何已 ``fit`` 过的 detector 实例。
    fit_params
        可选；该 detector 训练时使用的超参字典。仅 MiniRocket 走这条路（它没
        保留模型对象，只能从 ``num_kernels`` 反推）。
    """

    name = _algorithm_name_hint(detector)

    # 1) 已知无可训练参数的算法 → 0
    if any(token in name for token in _PARAMETERLESS_ALGOS):
        return 0

    # 2) MiniRocket 走超参反推
    if "minirocket" in name:
        return _minirocket_count(detector, fit_params)

    # 3) UNPrompt 不保存 nn.Module，只缓存 scores → 推断不出来
    if "unprompt" in name:
        return None

    # 4) 拿到底层模型对象（约定 _model；vendored DADA 也兜住）
    model = getattr(detector, "_model", None)
    if model is None:
        model = getattr(detector, "model", None)
    if model is None:
        return None

    # 5) 类型分派
    cls_name = type(model).__name__.lower()

    # OCSVM (PyOD wrapper or raw sklearn)
    if "ocsvm" in cls_name or "oneclasssvm" in cls_name:
        n = _ocsvm_count(model)
        if n is not None:
            return n

    # PyOD IForest wrapper
    if cls_name == "iforest":
        return _iforest_pyod_count(model)

    # XGBoost
    if "xgb" in cls_name:
        return _xgboost_count(model)

    # LightGBM
    if "lgbm" in cls_name or "lightgbm" in cls_name:
        return _lightgbm_count(model)

    # TabPFN
    if "tabpfn" in cls_name or "tabpfn" in name:
        n = _tabpfn_count(model)
        if n is not None:
            return n

    # sklearn 树集成（RandomForestClassifier / IsolationForest / ExtraTreesClassifier）
    if hasattr(model, "estimators_"):
        n = _sklearn_tree_ensemble_count(model)
        if n is not None:
            return n

    # sklearn MLPClassifier
    if hasattr(model, "coefs_"):
        n = _sklearn_mlp_count(model)
        if n is not None:
            return n

    # sklearn 线性模型（LR, etc.）
    if hasattr(model, "coef_"):
        n = _sklearn_linear_count(model)
        if n is not None:
            return n

    # PyTorch nn.Module（深度时序 / 图模型 / DADA / DeepSVDD / AutoEncoder / LSTM-AE/Sup）
    if hasattr(model, "parameters") and callable(model.parameters):
        n = _torch_param_count(model)
        if n is not None:
            return n

    # 兜底：PyOD-style 嵌套 ``detector._model.model``（detector → wrapper → 真模型）
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        if hasattr(nested, "parameters") and callable(nested.parameters):
            n = _torch_param_count(nested)
            if n is not None:
                return n

    return None
