"""异常检测算法系统性对比 — Streamlit 交互界面。

与项目计划书对齐：26 算法 × 5 模态 × 3 seed，Exp-1/2/3/4 + 精度-效率评估。

启动：streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.utils import (
    CONTAMINATION_RATES,
    EXP4_BASE_MODELS,
    EXP4_DENOISERS,
    EXP4_FLIP_RATES,
    EXP4_STRATEGIES,
    METRIC_HINTS,
    METRIC_LABELS,
    METRIC_RECOMMENDED,
    MODALITY_LABELS,
    MODALITY_DATA_SUBDIRS,
    N_ALGORITHMS,
    N_MODALITIES,
    PROJECT_TITLE,
    RESEARCH_QUESTIONS,
    aggregate_metric,
    check_experiment_files,
    contamination_rate_column,
    dataset_load_user_message,
    dedupe_results,
    eda_detail_options,
    eda_field_float,
    eda_field_int,
    fetch_adbench_demo_file,
    fetch_tsb_demo_file,
    default_data_root,
    default_figures_dir,
    default_results_dir,
    filter_by_seeds,
    format_metric_value,
    get_project_root,
    list_algorithms_from_exp,
    list_available_seeds,
    list_datasets_from_exp,
    load_eda_summary,
    load_experiment_results,
    plot_contamination_curves,
    plot_efficiency_tradeoff,
    plot_exp4_ablation_bars,
    plot_exp4_defense_curves,
    plot_metric_bars,
    plot_modality_heatmap,
    plot_radar,
    render_feature_plots,
    clear_dataset_load_cache,
    resolve_raw_dataset_path,
    render_sidebar_metric_guide,
    pollution_mode_label,
    render_advanced_figure_gallery,
    show_roc_fallback_figure,
    render_fallback_figures,
    summarize_project_coverage,
    build_eda_browser_frame,
    modality_status_display,
    summarize_modality_coverage,
    try_load_dataset,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_SEEDS = [41, 42, 43]


def _init_session_defaults() -> None:
    if "metric" not in st.session_state:
        st.session_state["metric"] = "auc_pr"
    if "selected_seeds" not in st.session_state:
        st.session_state["selected_seeds"] = DEFAULT_SEEDS.copy()


def _sidebar_config() -> None:
    st.sidebar.title("异常检测 Benchmark")
    st.sidebar.caption("交互式实验结果浏览")

    results_dir = default_results_dir()
    figures_dir = default_figures_dir()
    data_root = default_data_root()
    eda_path = get_project_root() / "data" / "eda_summary.json"

    with st.sidebar.expander("路径配置", expanded=False):
        results_dir = Path(st.text_input("结果目录", value=str(results_dir)))
        figures_dir = Path(st.text_input("图表目录", value=str(figures_dir)))
        data_root = Path(st.text_input("数据根目录 (data/)", value=str(data_root)))
        eda_path = Path(st.text_input("EDA 摘要", value=str(eda_path)))

    exp1_probe = load_experiment_results(str(results_dir), "exp1")
    available_seeds = list_available_seeds(exp1_probe) if not exp1_probe.empty else DEFAULT_SEEDS

    metric = st.sidebar.selectbox(
        "主指标",
        options=["auc_pr", "auc_roc", "f1_best"],
        index=["auc_pr", "auc_roc", "f1_best"].index(st.session_state.get("metric", "auc_pr")),
        format_func=lambda x: (
            f"{METRIC_LABELS[x]}（主推）" if x == METRIC_RECOMMENDED else METRIC_LABELS[x]
        ),
        help="三指标互补：AUC-ROC 看整体排序，AUC-PR 关注异常类（主推），F1@best 看最优工作点。详见下方说明。",
    )
    render_sidebar_metric_guide(metric)

    selected_seeds = st.sidebar.multiselect(
        "随机种子（多选取均值±标准差）",
        options=available_seeds,
        default=available_seeds,
        format_func=lambda s: f"seed={s}",
    )

    st.session_state["metric"] = metric
    st.session_state["selected_seeds"] = selected_seeds or available_seeds
    st.session_state["results_dir"] = results_dir
    st.session_state["figures_dir"] = figures_dir
    data_root_key = str(data_root.expanduser().resolve())
    if st.session_state.get("_data_root_key") not in (None, data_root_key):
        clear_dataset_load_cache()
    st.session_state["_data_root_key"] = data_root_key
    st.session_state["data_root"] = data_root
    st.session_state["eda_path"] = eda_path


def _metric() -> str:
    return st.session_state.get("metric", "auc_pr")


def _seeds() -> list[int]:
    return st.session_state.get("selected_seeds", DEFAULT_SEEDS)


def _prepare_exp(df: pd.DataFrame, dedupe_keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = filter_by_seeds(df, _seeds())
    return dedupe_results(filtered, dedupe_keys)


def page_home() -> None:
    st.title(PROJECT_TITLE)
    st.markdown(
        f"{N_ALGORITHMS} 种算法 · {N_MODALITIES} 类数据形态 · "
        f"seed ∈ {{41, 42, 43}}"
    )

    st.subheader("核心研究问题")
    for title, desc in RESEARCH_QUESTIONS:
        st.markdown(f"- **{title}**：{desc}")

    results_dir = st.session_state["results_dir"]
    status = check_experiment_files(results_dir)
    c1, c2, c3, c4 = st.columns(4)
    labels = [("exp1", "Exp-1 基线"), ("exp2", "Exp-2 污染"), ("exp3", "Exp-3 跨模态"), ("exp4", "Exp-4 防御")]
    for col, (key, label) in zip([c1, c2, c3, c4], labels):
        col.metric(label, "✓ 已就绪" if status.get(key) else "✗ 缺失")

    eda = load_eda_summary(str(st.session_state["eda_path"]))
    exp1 = load_experiment_results(str(results_dir), "exp1")
    coverage = summarize_project_coverage(eda, _prepare_exp(exp1, ["dataset_name", "algorithm_name", "modality", "seed"]))
    if not coverage.empty:
        st.subheader("数据覆盖")
        st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.subheader("评估指标说明")
    st.caption("每个「算法 × 数据集」组合统一报告以下三个互补指标，交叉印证，避免单一指标误导。")
    for key in ["auc_roc", "auc_pr", "f1_best"]:
        badge = " **【主推】**" if key == METRIC_RECOMMENDED else ""
        st.markdown(f"- **{METRIC_LABELS[key]}**{badge}：{METRIC_HINTS[key]}")

    with st.expander("一键复现命令"):
        st.code(
            "python run_all.py --exp all --data-root data --output-dir results --seeds 41 42 43 --analyze\n"
            "streamlit run app/streamlit_app.py",
            language="bash",
        )


def page_dataset_browser() -> None:
    st.header("数据集浏览")
    data_root = Path(st.session_state["data_root"])
    st.caption(
        f"数据来源：`{data_root}` · 统计摘要 `eda_summary.json` · "
        "文件清单 `selected_files.json` · 经 `adapters.load_dataset` 统一加载"
    )

    eda = load_eda_summary(str(st.session_state["eda_path"]))
    if eda.empty:
        st.warning("未找到 EDA 摘要。请确认 `data/eda_summary.json` 存在。")
        return

    coverage = summarize_modality_coverage(data_root, eda)
    status_cols = st.columns(len(MODALITY_DATA_SUBDIRS))
    for col, (mod, _) in zip(status_cols, MODALITY_DATA_SUBDIRS.items()):
        label, delta = modality_status_display(coverage[mod])
        col.metric(MODALITY_LABELS.get(mod, mod), label, delta=delta)

    ok = build_eda_browser_frame(eda, data_root)

    c1, c2 = st.columns(2)
    modality_filter = c1.multiselect(
        "数据形态",
        options=sorted(ok["modality"].dropna().unique()),
        default=sorted(ok["modality"].dropna().unique()),
        format_func=lambda x: MODALITY_LABELS.get(x, x),
    )
    if "source" in ok.columns:
        source_filter = c2.multiselect(
            "Benchmark 来源",
            options=sorted(ok["source"].dropna().unique()),
            default=sorted(ok["source"].dropna().unique()),
            format_func=lambda x: x,
        )
    else:
        source_filter = []

    filtered = ok.copy()
    if modality_filter:
        filtered = filtered[filtered["modality"].isin(modality_filter)]
    if source_filter:
        filtered = filtered[filtered["source"].isin(source_filter)]

    st.dataframe(
        filtered[[c for c in [
            "name", "modality", "source", "n_samples", "n_features",
            "n_nodes", "anomaly_rate",
        ] if c in filtered.columns]].rename(columns={
            "name": "数据集", "modality": "形态", "source": "来源",
            "n_samples": "样本数", "n_features": "特征维度",
            "n_nodes": "节点数", "anomaly_rate": "异常率",
        }),
        use_container_width=True,
        hide_index=True,
    )

    detail_df = eda_detail_options(filtered)
    if detail_df.empty:
        return

    sort_col = st.selectbox(
        "表格排序",
        ["name", "anomaly_rate", "n_samples", "n_features"],
        format_func=lambda c: {
            "name": "数据集名称", "anomaly_rate": "异常率",
            "n_samples": "样本数", "n_features": "特征维度",
        }.get(c, c),
    )
    ascending = st.checkbox("升序排列", value=sort_col == "name")
    if sort_col in detail_df.columns:
        detail_df = detail_df.sort_values(sort_col, ascending=ascending, na_position="last")

    max_hist = st.slider("直方图展示特征数", 3, 12, 6, key="ds_hist_n")

    selected_label = st.selectbox("查看详情", detail_df["_label"].tolist())
    row = detail_df[detail_df["_label"].eq(selected_label)].iloc[0]
    selected = str(row["name"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("样本/节点", f"{eda_field_int(row, 'n_samples', 'n_nodes'):,}")
    m2.metric("特征维度", eda_field_int(row, "n_features"))
    m3.metric("异常率", f"{eda_field_float(row, 'anomaly_rate') * 100:.2f}%")
    m4.metric("形态", MODALITY_LABELS.get(str(row.get("modality", "")), ""))

    if "preprocessing" in row and pd.notna(row["preprocessing"]):
        st.json({"preprocessing": row["preprocessing"], "split": row.get("split"), "seed": row.get("seed")})

    modality = str(row.get("modality", "tabular"))
    st.subheader("特征分布")
    loaded = try_load_dataset(selected, modality, str(data_root))
    if loaded and "error" not in loaded:
        if modality == "graph":
            st.caption("图数据：训练集节点特征矩阵直方图。")
        render_feature_plots(loaded["X_train"], max_hist_features=max_hist)
    else:
        err_msg = (loaded or {}).get("error", "")
        raw_path = resolve_raw_dataset_path(selected, modality, data_root)
        if raw_path is not None and raw_path.exists():
            st.warning(
                f"已找到本地文件 `{raw_path}`，但加载未成功。"
                + (f"\n\n原因：{err_msg}" if err_msg else "")
            )
            if st.button("清除缓存并重新加载", key=f"reload_{selected}_{modality}"):
                clear_dataset_load_cache()
                st.rerun()
        else:
            st.info(dataset_load_user_message(modality, data_root))
            if modality in {"tabular", "cv", "nlp"}:
                if st.button("下载 ADBench 演示文件", key=f"fetch_{selected}_{modality}"):
                    ok, msg = fetch_adbench_demo_file(selected, data_root)
                    if ok:
                        clear_dataset_load_cache()
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            elif modality == "timeseries":
                if st.button("下载 TSB-AD 演示文件", key=f"fetch_{selected}_{modality}"):
                    with st.spinner("首次下载 TSB-AD-U.zip（约 70MB）并解压，请稍候…"):
                        ok, msg = fetch_tsb_demo_file(selected, data_root)
                    if ok:
                        clear_dataset_load_cache()
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


def page_baseline() -> None:
    st.header("基准对比 · Exp-1")
    st.caption("干净数据下的算法 × 数据集性能；侧边栏 seed 多选可查看均值±标准差")

    results_dir = st.session_state["results_dir"]
    metric = _metric()
    raw = load_experiment_results(str(results_dir), "exp1")
    if raw.empty:
        render_fallback_figures(
            st.session_state["figures_dir"],
            [("全局热力图", "exp1_*heatmap*"), ("平均排名", "exp1_average_rank*")],
            key="exp1_fallback",
            warning=f"缺少 {results_dir / 'exp1_results.csv'}。",
        )
        return

    deduped = _prepare_exp(raw, ["dataset_name", "algorithm_name", "modality", "seed"])

    c1, c2, c3 = st.columns(3)
    modalities = sorted(deduped["modality"].dropna().unique())
    modality = c1.selectbox("数据形态", ["全部"] + modalities, format_func=lambda x: MODALITY_LABELS.get(x, x) if x != "全部" else x)
    part = deduped if modality == "全部" else deduped[deduped["modality"].astype(str).eq(modality)]
    datasets = list_datasets_from_exp(part)
    dataset = c2.selectbox("数据集", datasets)
    algos = sorted(part[part["dataset_name"].astype(str).eq(dataset)]["algorithm_name"].unique())
    algo_filter = c3.multiselect("算法筛选（空=全部）", algos, default=[])

    subset_raw = part[part["dataset_name"].astype(str).eq(dataset)]
    if algo_filter:
        subset_raw = subset_raw[subset_raw["algorithm_name"].isin(algo_filter)]

    st.subheader("算法性能柱状图")
    bc1, bc2 = st.columns(2)
    bar_metrics = bc1.multiselect(
        "柱状图指标",
        ["auc_roc", "auc_pr", "f1_best"],
        default=[metric] if metric in ["auc_roc", "auc_pr", "f1_best"] else [metric, "auc_roc"],
        format_func=lambda x: METRIC_LABELS.get(x, x),
        key="exp1_bar_metrics",
    )
    show_std_bars = bc2.checkbox("显示多种子误差棒", value=len(_seeds()) > 1, key="exp1_show_std")

    agg = aggregate_metric(subset_raw, ["algorithm_name"], bar_metrics[0] if bar_metrics else metric)
    if agg.empty or not bar_metrics:
        st.warning("当前筛选无结果。")
        return
    agg[f"{bar_metrics[0]}_std"] = agg["std"]
    for extra_m in bar_metrics[1:]:
        extra_agg = aggregate_metric(subset_raw, ["algorithm_name"], extra_m)
        if not extra_agg.empty:
            agg = agg.merge(extra_agg[["algorithm_name", extra_m]], on="algorithm_name", how="left")

    st.caption(
        f"数据集 **{dataset}** · 算法 {len(agg)} 个 · seed={_seeds()} · "
        f"Top-1: **{agg.iloc[agg[bar_metrics[0]].argmax()]['algorithm_name']}** "
        f"({format_metric_value(agg[bar_metrics[0]].max(), agg.loc[agg[bar_metrics[0]].idxmax(), 'std'])})"
    )

    fig = plot_metric_bars(agg, bar_metrics, f"Exp-1 · {dataset}", show_std=show_std_bars)
    st.pyplot(fig, clear_figure=True)

    figures_dir = st.session_state["figures_dir"]

    st.subheader("ROC 曲线")
    show_roc_fallback_figure(figures_dir, dataset)

    render_advanced_figure_gallery(
        figures_dir,
        [
            ("全局热力图", "exp1_*heatmap*"),
            ("平均排名", "exp1_average_rank*"),
            ("精度-效率", "exp1_efficiency*"),
            ("Nemenyi 检验", "nemenyi*"),
            ("Friedman 排名", "friedman*"),
        ],
        key="exp1_gallery",
        default_label="heatmap",
    )

    with st.expander("原始结果（含多种子）"):
        cols = [c for c in ["algorithm_name", "seed", "auc_roc", "auc_pr", "f1_best", "fit_time_sec", "predict_time_sec"] if c in subset_raw.columns]
        st.dataframe(subset_raw[cols].sort_values([metric, "seed"], ascending=[False, True]), use_container_width=True, hide_index=True)


def page_contamination() -> None:
    st.header("污染鲁棒性 · Exp-2")
    st.caption("污染率 {0, 1, 5, 10, 20}% · 无监督=保留异常比例 · 有监督=对称标签翻转")

    results_dir = st.session_state["results_dir"]
    metric = _metric()
    raw = load_experiment_results(str(results_dir), "exp2")
    if raw.empty:
        render_fallback_figures(
            st.session_state["figures_dir"],
            [
                ("退化曲线", "exp2_degradation_*"),
                ("跨模态 AUC 下降", "exp2_cross_modal_auc_drop*"),
            ],
            key="exp2_fallback",
            warning=f"缺少 {results_dir / 'exp2_results.csv'}。",
        )
        return

    deduped = _prepare_exp(
        raw,
        ["modality", "dataset_name", "algorithm_name", "contamination_mode", "contamination_rate", "label_flip_rate", "seed"],
    )
    deduped = deduped.copy()
    deduped["rate"] = contamination_rate_column(deduped)

    c1, c2, c3, c4 = st.columns(4)
    modalities = sorted(deduped["modality"].dropna().unique())
    modality = c1.selectbox("形态", modalities, format_func=lambda x: MODALITY_LABELS.get(x, x), key="exp2_mod")
    pollution_mode = c2.selectbox("污染机制", ["全部", "无监督", "有监督"])
    ds_list = list_datasets_from_exp(deduped, modality)
    dataset = c3.selectbox("数据集", ["全部（形态内汇总）"] + ds_list, key="exp2_ds")
    rate_pct = c4.select_slider(
        "快照污染率 (%)",
        options=[int(r * 100) for r in CONTAMINATION_RATES],
        value=20,
    )

    view = deduped[deduped["modality"].astype(str).eq(modality)]
    if dataset != "全部（形态内汇总）":
        view = view[view["dataset_name"].astype(str).eq(dataset)]

    all_algos = list_algorithms_from_exp(view)
    c5, c6 = st.columns(2)
    curve_algos = c5.multiselect(
        "退化曲线 · 算法筛选（空=全部）",
        all_algos,
        default=[],
        key="exp2_curve_algos",
    )
    rank_top_n = c6.slider("排名柱状图 · 展示 Top-N", 5, 30, 15, key="exp2_rank_n")

    mode_key = None if pollution_mode == "全部" else ("unsupervised" if pollution_mode == "无监督" else "supervised")
    fig = plot_contamination_curves(
        view, metric=metric, pollution_filter=mode_key,
        algorithms=curve_algos or None,
    )
    st.pyplot(fig, clear_figure=True)

    snapshot = view[np.isclose(view["rate"] * 100, rate_pct)]
    if mode_key == "unsupervised":
        snapshot = snapshot[snapshot.apply(lambda r: "无监督" in pollution_mode_label(r), axis=1)]
    elif mode_key == "supervised":
        snapshot = snapshot[snapshot.apply(lambda r: "有监督" in pollution_mode_label(r), axis=1)]

    st.subheader(f"污染率 = {rate_pct}% 时的算法排名")
    if snapshot.empty:
        st.info("该条件下无数据。")
    else:
        ranking = aggregate_metric(snapshot, ["algorithm_name"], metric).sort_values(metric, ascending=False).head(rank_top_n)
        st.dataframe(
            ranking.assign(显示=ranking.apply(lambda r: format_metric_value(r[metric], r["std"]), axis=1))[
                ["algorithm_name", metric, "std", "n_runs", "显示"]
            ].rename(columns={"algorithm_name": "算法", metric: METRIC_LABELS.get(metric, metric)}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(ranking, x="algorithm_name", y=metric, use_container_width=True)

    render_advanced_figure_gallery(
        st.session_state["figures_dir"],
        [
            (f"{MODALITY_LABELS.get(modality, modality)} 退化曲线", f"exp2_degradation_{modality}*"),
            ("跨模态 AUC 下降", "exp2_cross_modal_auc_drop*"),
            ("鲁棒性排名", "exp2_robustness_ranking*"),
            ("其他形态退化", "exp2_degradation_*"),
        ],
        key="exp2_gallery",
        default_label=f"degradation_{modality}",
    )


def page_cross_modal() -> None:
    st.header("跨模态泛化 · Exp-3")
    st.caption("11 个通用算法横跨 5 类形态；观察排名稳定性与全能性")

    metric = _metric()
    raw = load_experiment_results(str(st.session_state["results_dir"]), "exp3")
    if raw.empty:
        render_fallback_figures(
            st.session_state["figures_dir"],
            [("跨模态热力图", "exp3_*heatmap*"), ("雷达图", "exp3_radar*")],
            key="exp3_fallback",
            warning="缺少 exp3_results.csv。",
        )
        return

    deduped = _prepare_exp(raw, ["modality", "dataset_name", "algorithm_name", "seed"])

    fc1, fc2, fc3 = st.columns(3)
    datasets = sorted(deduped["dataset_name"].astype(str).unique())
    ds_filter = fc1.selectbox("数据聚合", ["全部数据集均值", *datasets], key="exp3_ds")
    modalities_avail = sorted(deduped["modality"].dropna().unique())
    mod_pick = fc2.multiselect(
        "热力图 · 形态列",
        modalities_avail,
        default=modalities_avail,
        format_func=lambda x: MODALITY_LABELS.get(x, x),
        key="exp3_mods",
    )
    view = deduped if ds_filter == "全部数据集均值" else deduped[deduped["dataset_name"].astype(str).eq(ds_filter)]

    mod_df = view.pivot_table(index="algorithm_name", columns="modality", values=metric, aggfunc="mean")
    if mod_pick:
        mod_df = mod_df[[c for c in mod_pick if c in mod_df.columns]]
    mod_df = mod_df.loc[mod_df.mean(axis=1).sort_values(ascending=False).index]

    algo_pick_heat = fc3.multiselect(
        "热力图 · 算法行（空=全部）",
        mod_df.index.tolist(),
        default=[],
        key="exp3_heat_algos",
    )
    if algo_pick_heat:
        mod_df = mod_df.loc[[a for a in algo_pick_heat if a in mod_df.index]]

    st.subheader("算法 × 形态热力图")
    st.pyplot(plot_modality_heatmap(mod_df, metric=metric), clear_figure=True)

    difficulty = mod_df.mean(axis=0).sort_values(ascending=False)
    diff_df = difficulty.reset_index()
    diff_df.columns = ["modality", metric]
    diff_df["形态"] = diff_df["modality"].map(lambda m: MODALITY_LABELS.get(m, m))
    st.subheader("形态难度（均值越低越难）")
    diff_order = st.multiselect(
        "难度图 · 形态顺序",
        diff_df["形态"].tolist(),
        default=diff_df["形态"].tolist(),
        key="exp3_diff_order",
    )
    if diff_order:
        diff_df = diff_df.set_index("形态").loc[diff_order].reset_index()
    st.bar_chart(diff_df, x="形态", y=metric, use_container_width=True)

    st.subheader("跨模态雷达图")
    default_algos = mod_df.mean(axis=1).sort_values(ascending=False).head(5).index.tolist()
    selected = st.multiselect("雷达图 · 选择算法", mod_df.index.tolist(), default=default_algos, key="exp3_radar")
    radar_fig = plot_radar(mod_df, selected, metric=metric)
    if radar_fig:
        st.pyplot(radar_fig, clear_figure=True)
    else:
        st.info("请至少选择 3 个在已选形态上都有结果的算法。")

    render_advanced_figure_gallery(
        st.session_state["figures_dir"],
        [
            ("算法×形态热力图", "exp3_*heatmap*"),
            ("雷达图", "exp3_radar*"),
            ("跨模态折线", "exp3_cross_modal_lines*"),
            ("形态难度", "exp3_modality_difficulty*"),
        ],
        key="exp3_gallery",
    )


def page_defense() -> None:
    st.header("鲁棒防御 · Exp-4")
    st.caption(
        "扩展实验：对称标签翻转 {0, 5, 10, 20}% 下，"
        "6 种有监督基模型 × 8 种无监督去噪器 × Trim/Flip 策略的防御消融"
    )

    metric = _metric()
    raw = load_experiment_results(str(st.session_state["results_dir"]), "exp4")
    if raw.empty:
        render_fallback_figures(
            st.session_state["figures_dir"],
            [("防御分析", "exp4_*")],
            key="exp4_fallback",
            warning="缺少 exp4_results.csv。",
        )
        return

    deduped = _prepare_exp(
        raw,
        ["modality", "dataset_name", "algorithm_name", "label_flip_rate", "seed"],
    )

    c1, c2, c3, c4 = st.columns(4)
    modalities = sorted(deduped["modality"].dropna().unique())
    modality = c1.selectbox("形态", modalities, format_func=lambda x: MODALITY_LABELS.get(x, x), key="exp4_mod")
    ds_list = list_datasets_from_exp(deduped, modality)
    dataset = c2.selectbox("数据集", ds_list, key="exp4_ds")
    base_model = c3.selectbox("有监督基模型", EXP4_BASE_MODELS, index=0, key="exp4_base")
    denoiser_focus = c4.multiselect(
        "曲线展示的去噪器（空=全部）",
        EXP4_DENOISERS,
        default=["IQR", "IForest", "KNN"],
        key="exp4_denoisers",
    )

    view = deduped[
        (deduped["modality"].astype(str).eq(modality))
        & (deduped["dataset_name"].astype(str).eq(dataset))
    ]
    if view.empty:
        st.warning("当前数据集无 Exp-4 记录。")
        return

    ec1, ec2, ec3 = st.columns(3)
    curve_strategies = ec1.multiselect(
        "防御曲线 · 策略",
        EXP4_STRATEGIES,
        default=EXP4_STRATEGIES,
        key="exp4_curve_strat",
    )
    ablation_denoisers = ec2.multiselect(
        "消融柱图 · 去噪器（空=全部）",
        EXP4_DENOISERS,
        default=[],
        key="exp4_ablation_denoisers",
    )
    ablation_strategies = ec3.multiselect(
        "消融柱图 · 策略（空=全部）",
        EXP4_STRATEGIES,
        default=[],
        key="exp4_ablation_strat",
    )

    st.subheader("防御退化曲线")
    st.pyplot(
        plot_exp4_defense_curves(
            view,
            base_model=base_model,
            denoisers=denoiser_focus or None,
            strategies=curve_strategies or None,
            metric=metric,
        ),
        clear_figure=True,
    )
    st.markdown(
        "**图例**：红色实线 = 无防御基线；绿色系 **Trim** = 剪裁可疑样本；棕色系 **Flip** = 翻转可疑标签"
    )

    flip_pct = st.select_slider(
        "消融快照 · 标签翻转率 (%)",
        options=[int(r * 100) for r in EXP4_FLIP_RATES],
        value=20,
        key="exp4_flip",
    )
    ab1, ab2, ab3 = st.columns(3)
    ablation_top_n = ab1.slider("消融柱图 · Top-N", 5, 20, 12, key="exp4_ab_top")
    include_std = ab2.checkbox("消融含无防御基线", value=True, key="exp4_inc_std")
    ablation_metric = ab3.selectbox(
        "消融柱图指标",
        ["auc_roc", "auc_pr", "f1_best"],
        index=["auc_roc", "auc_pr", "f1_best"].index(metric),
        format_func=lambda x: METRIC_LABELS.get(x, x),
        key="exp4_ab_metric",
    )

    st.subheader("消融对比柱图")
    st.pyplot(
        plot_exp4_ablation_bars(
            view,
            base_model=base_model,
            flip_rate=flip_pct / 100.0,
            metric=ablation_metric,
            top_n=ablation_top_n,
            denoisers=ablation_denoisers or None,
            strategies=ablation_strategies or None,
            include_standard=include_std,
        ),
        clear_figure=True,
    )

    st.subheader("核心结论速览")
    st.markdown(
        """
        - **Trim 普遍优于 Flip**：剪裁策略不引入额外标签噪声
        - **IQR_Trim / IForest_Trim**：高噪（20%）环境下常见最优防御组合
        - **TabPFN / LR**：防御可能破坏边界样本，出现反效果（No Free Lunch）
        """
    )

    with st.expander("原始结果抽样"):
        cols = [c for c in ["algorithm_name", "seed", "label_flip_rate", "auc_roc", "auc_pr", "f1_best", "status"] if c in view.columns]
        st.dataframe(
            view[cols].sort_values(["label_flip_rate", metric], ascending=[True, False]).head(40),
            use_container_width=True,
            hide_index=True,
        )

    render_advanced_figure_gallery(
        st.session_state["figures_dir"],
        [
            ("动态防御包络", "exp4_dynamic*"),
            ("消融对比", "exp4_ablation*"),
            ("防御对比", "exp4_defense*"),
        ],
        key="exp4_gallery",
    )


def page_efficiency() -> None:
    st.header("精度-效率权衡")
    st.caption("除精度外同步评估训练/推理耗时与参数量代理")

    metric = _metric()
    raw = load_experiment_results(str(st.session_state["results_dir"]), "exp1")
    if raw.empty:
        render_fallback_figures(
            st.session_state["figures_dir"],
            [("精度-效率权衡", "exp1_efficiency*")],
            key="eff_fallback",
            warning="缺少 exp1_results.csv。",
        )
        return

    deduped = _prepare_exp(raw, ["dataset_name", "algorithm_name", "modality", "seed"])

    e1, e2, e3, e4, e5 = st.columns(5)
    modalities = sorted(deduped["modality"].dropna().unique())
    modality = e1.selectbox(
        "数据形态",
        ["全部"] + modalities,
        format_func=lambda x: MODALITY_LABELS.get(x, x) if x != "全部" else x,
        key="eff_mod",
    )
    part = deduped if modality == "全部" else deduped[deduped["modality"].astype(str).eq(modality)]
    ds_list = list_datasets_from_exp(part)
    dataset = e2.selectbox("数据集", ["全部"] + ds_list, key="eff_ds")
    if dataset != "全部":
        part = part[part["dataset_name"].astype(str).eq(dataset)]
    algo_type = e3.selectbox("算法类型", ["全部", "监督", "无监督"], key="eff_type")
    all_algos = list_algorithms_from_exp(part)
    eff_algos = e4.multiselect("算法筛选（空=全部）", all_algos, default=[], key="eff_algos")
    eff_top_n = e5.slider("散点图标注数", 5, 25, 15, key="eff_top_n")

    st.subheader("精度-效率散点图")
    st.pyplot(
        plot_efficiency_tradeoff(
            part, metric=metric,
            algorithms=eff_algos or None,
            algo_type=algo_type,
            top_n=eff_top_n,
        ),
        clear_figure=True,
    )
    st.markdown("气泡越大表示参数量/模型复杂度代理越高；**橙色=监督**，**蓝色=无监督**")

    render_advanced_figure_gallery(
        st.session_state["figures_dir"],
        [("精度-效率权衡", "exp1_efficiency*")],
        key="eff_gallery",
    )

    top_n_table = st.slider("耗时明细表 Top-N", 5, 30, 15, key="eff_table_n")
    top = (
        part.groupby("algorithm_name", as_index=False)
        .agg({metric: "mean", "fit_time_sec": "median", "predict_time_sec": "median"})
        .assign(total_time=lambda d: d["fit_time_sec"] + d["predict_time_sec"])
        .sort_values(metric, ascending=False)
        .head(top_n_table)
    )
    if eff_algos:
        top = top[top["algorithm_name"].isin(eff_algos)]
    with st.expander(f"耗时明细 Top-{top_n_table}"):
        st.dataframe(top, use_container_width=True, hide_index=True)


PAGES = {
    "首页": page_home,
    "数据集浏览": page_dataset_browser,
    "基准对比 (Exp-1)": page_baseline,
    "污染率分析 (Exp-2)": page_contamination,
    "跨模态对比 (Exp-3)": page_cross_modal,
    "鲁棒防御 (Exp-4)": page_defense,
    "精度-效率": page_efficiency,
}


def main() -> None:
    st.set_page_config(
        page_title="异常检测 Benchmark",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_defaults()
    _sidebar_config()
    choice = st.sidebar.radio("页面导航", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
