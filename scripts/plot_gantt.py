"""生成项目甘特图 PNG（放进报告/PPT 用）。

用法:
    python scripts/plot_gantt.py
输出:
    results/gantt_chart.png
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

# 中文字体（按系统可用字体回退）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def d(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()


# (任务名, 开始, 结束, 阶段)  —— 结束日为闭区间，绘图时 +1 天表示当天完成
TASKS = [
    ("数据下载与适配层接口",       "2026-05-26", "2026-05-27", "框架搭建"),
    ("算法基类与实验脚本骨架",     "2026-05-26", "2026-05-27", "框架搭建"),
    ("评估指标与日志 schema",      "2026-05-27", "2026-05-28", "框架搭建"),
    ("表格数据适配与预处理",       "2026-05-27", "2026-05-28", "基线 Exp-1"),
    ("经典算法封装",               "2026-05-27", "2026-05-28", "基线 Exp-1"),
    ("深度算法封装(AE/DeepSVDD)",  "2026-05-29", "2026-05-30", "基线 Exp-1"),
    ("时序切窗与图特征适配",       "2026-05-29", "2026-05-31", "基线 Exp-1"),
    ("Exp-1 表格全量跑通",         "2026-05-29", "2026-06-01", "基线 Exp-1"),
    ("时序专用算法封装",           "2026-06-02", "2026-06-03", "污染/跨模态 Exp-2/3"),
    ("图专用算法封装",             "2026-06-04", "2026-06-05", "污染/跨模态 Exp-2/3"),
    ("Exp-2 污染实验(3 seed)",     "2026-06-02", "2026-06-06", "污染/跨模态 Exp-2/3"),
    ("Exp-3 跨模态实验(3 seed)",   "2026-06-04", "2026-06-07", "污染/跨模态 Exp-2/3"),
    ("跨环境缺陷修复",             "2026-06-06", "2026-06-08", "污染/跨模态 Exp-2/3"),
    ("Exp-1 图表",                 "2026-06-02", "2026-06-04", "可视化与界面"),
    ("Exp-2/3 退化曲线与雷达图",   "2026-06-06", "2026-06-08", "可视化与界面"),
    ("Streamlit 交互界面",         "2026-05-29", "2026-06-08", "可视化与界面"),
    ("报告各章撰写",               "2026-06-09", "2026-06-12", "报告与答辩"),
    ("答辩 PPT 与终稿",            "2026-06-13", "2026-06-16", "报告与答辩"),
]

# 里程碑 (名称, 日期)
MILESTONES = [
    ("接口冻结",         "2026-05-28"),
    ("M1 pipeline跑通",  "2026-06-01"),
    ("M2 实验完成",      "2026-06-08"),
    ("M3 报告初稿",      "2026-06-12"),
    ("M4 终稿+答辩",     "2026-06-16"),
]

# 柔和现代配色
PHASE_COLORS = {
    "框架搭建":              "#5B8FF9",
    "基线 Exp-1":            "#5AD8A6",
    "污染/跨模态 Exp-2/3":   "#F6BD16",
    "可视化与界面":          "#9270CA",
    "报告与答辩":            "#FF9D4D",
}

CHART_START = d("2026-05-26")
CHART_END = d("2026-06-18")  # 留出右边界，避免 M4 标注被裁切


def add_round_bar(ax, x0, width, y, height, color):
    """画带圆角的任务条。"""
    box = FancyBboxPatch(
        (mdates.date2num(x0), y - height / 2),
        width, height,
        boxstyle="round,pad=0,rounding_size=0.18",
        mutation_aspect=0.6,
        linewidth=0,
        facecolor=color,
        alpha=0.95,
        zorder=3,
    )
    ax.add_patch(box)


def main() -> None:
    n = len(TASKS)
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFBFD")

    # 交替行背景带
    for i in range(n):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F0F2F5", zorder=0)

    # 任务条 + 左侧任务名
    for i, (name, start, end, phase) in enumerate(TASKS):
        y = n - 1 - i
        s = d(start)
        e = d(end) + dt.timedelta(days=1)
        width = mdates.date2num(e) - mdates.date2num(s)
        add_round_bar(ax, s, width, y, 0.62, PHASE_COLORS[phase])
        # 任务名放在条内左侧
        ax.text(
            mdates.date2num(s) + 0.12, y, name,
            va="center", ha="left", fontsize=9.5,
            color="white", fontweight="bold", zorder=4,
        )

    # 里程碑：菱形 + 标注。表示"当天结束时达成"，画在该日格子右边界
    # （即该日与次日之间的分界线上）
    for name, date_str in MILESTONES:
        x = mdates.date2num(d(date_str)) + 1.0
        ax.scatter(x, n - 0.3, marker="D", s=85,
                   color="#E8684A", edgecolor="white", linewidth=1.2,
                   zorder=6, clip_on=False)
        ax.annotate(
            name, (x, n - 0.3), xytext=(0, 14), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color="#444", rotation=0,
        )

    # 轴设置
    ax.set_yticks([])
    ax.set_xlim(mdates.date2num(CHART_START) - 0.4,
                mdates.date2num(CHART_END) + 0.4)
    ax.set_ylim(-0.7, n + 0.6)

    # 一天一格：日期标签居中到每个格子正中央（避免"标签在格线上"造成的歧义）
    day0 = mdates.date2num(CHART_START)
    day1 = mdates.date2num(CHART_END)
    n_days = int(day1 - day0)
    # 标签刻度放在每格中心（整数日 + 0.5）
    label_ticks = [day0 + i + 0.5 for i in range(n_days)]
    label_text = [
        (CHART_START + dt.timedelta(days=i)).strftime("%m/%d")
        for i in range(n_days)
    ]
    ax.set_xticks(label_ticks)
    ax.set_xticklabels(label_text)
    # 网格线画在格子边界（整数日），单独用竖线绘制
    for i in range(n_days + 1):
        ax.axvline(day0 + i, color="#E3E6EB", lw=0.6, zorder=1)

    # 日期刻度移到顶部
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="major", length=0, pad=6)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8.5, color="#555")

    # 去掉边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 标题（放最上方，给里程碑标注留空间）
    ax.set_title(
        "异常检测算法系统性对比研究 · 项目甘特图",
        fontsize=15, fontweight="bold", pad=46, color="#222",
    )

    # 图例（阶段 + 里程碑）
    legend_handles = [Patch(color=c, label=p) for p, c in PHASE_COLORS.items()]
    legend_handles.append(
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#E8684A",
                   markeredgecolor="white", markersize=9, label="里程碑")
    )
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=6, fontsize=9,
              frameon=False, handlelength=1.2, columnspacing=1.4)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = Path(__file__).resolve().parent.parent / "results" / "gantt_chart.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"甘特图已保存 → {out}")


if __name__ == "__main__":
    main()
