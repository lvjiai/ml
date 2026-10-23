"""Streamlit demo: one sensor trajectory in, RUL + interval out.

Labels avoid adjectives and parenthetical asides: every element says what it is
(noun) or what it does (verb), and the range is explained by its computation in
the "90% 范围的计算方法" expander.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chinese labels need a CJK font; without this every glyph renders as a box.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.demo import coerce_uploaded, observable_window, settings_defaults  # noqa: E402
from src.pipeline import load_bundle, predict_one, predict_last, with_targets  # noqa: E402

# The four NASA settings, named by what varies in them.
SCENARIOS = {
    "场景 1：一种运转状态": "FD001",
    "场景 2：六种运转状态": "FD002",
    "场景 3：两种损坏方式": "FD003",
    "场景 4：六种运转状态、两种损坏方式": "FD004",
}
SHOWN_SENSORS = ("s_4", "s_11", "s_12", "s_7")

st.set_page_config(page_title="剩余周期数", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle(subset: str):
    return load_bundle(subset)


@st.cache_data(show_spinner=False)
def get_dataset(subset: str, show_truth: bool) -> pd.DataFrame:
    ds = load_subset(subset, Config().data_dir)
    if show_truth:
        return with_targets(ds.train, Config().rul_clip)
    return ds.test


@st.cache_data(show_spinner=False)
def rul_curve(subset: str, show_truth: bool, unit: int, max_cycle: int, n_points: int = 18):
    """Predicted RUL as the observed window grows."""
    bundle = get_bundle(subset)
    df = get_dataset(subset, show_truth)
    cutoffs = np.unique(
        np.linspace(30, max_cycle, num=min(n_points, max(2, max_cycle // 5))).astype(int)
    )
    cutoffs = cutoffs[cutoffs <= max_cycle]
    out = []
    for cut in cutoffs:
        rows = df[(df["unit"] == unit) & (df["cycle"] <= cut)]
        if len(rows) < 30:
            continue
        pred = predict_one(bundle, rows)
        pred["cutoff"] = int(cut)
        out.append(pred)
    return pd.DataFrame(out)


def sensor_figure(rows: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=130)
    for name, col in zip(SHOWN_SENSORS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")):
        s = rows[name]
        z = (s - s.mean()) / (s.std() or 1.0)
        ax.plot(rows["cycle"], z, label=name, color=col, linewidth=1.2)
    ax.set_xlabel("周期")
    ax.set_ylabel("读数（标准化）")
    ax.set_title("图 1：s_4、s_11、s_12、s_7 的读数", fontsize=10)
    ax.legend(fontsize=7, ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def rul_figure(
    curve: pd.DataFrame, max_cycle: int, point: float, cut: int
) -> plt.Figure:
    """Predicted vs true RUL along the observation window.

    The true RUL of a train unit at cutoff t is ``max_cycle - t``.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=130)
    ax.plot(curve["cutoff"], curve["rul_point"], color="#1f77b4", label="点预测")
    if "rul_lo" in curve.columns:
        ax.fill_between(
            curve["cutoff"], curve["rul_lo"], curve["rul_hi"], color="#1f77b4", alpha=0.18,
            label="90% 范围",
        )
    if max_cycle > 0:
        ax.plot(
            curve["cutoff"], max_cycle - curve["cutoff"], color="#d62728",
            linestyle="--", linewidth=1.0, label="真实值",
        )
    ax.axvline(cut, color="grey", linestyle=":", linewidth=1.0)
    ax.scatter([cut], [point], color="black", zorder=5, s=28)
    ax.set_xlabel("观测到的周期数")
    ax.set_ylabel("剩余周期数")
    ax.set_title("图 2：点预测与 90% 范围随观测周期数变化", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def page_map() -> None:
    st.markdown(
        "| 位置 | 内容 | 操作 |\n"
        "|---|---|---|\n"
        "| 左侧边栏 | 场景、机器编号、观测周期数 | 选场景，选机器，拖动滑杆 |\n"
        "| 上方四个数 | 点预测、90% 范围、范围宽度、真实值 | 读数字 |\n"
        "| 下方两张图 | 传感器读数、点预测随观测周期数的变化 | 读图 |\n"
        "| 底部展开项 | 计算方法、数据边界、CSV 上传 | 点开 |"
    )


def range_method() -> None:
    """How the 90% range is produced, in the order the code executes it."""
    st.markdown(
        "这个范围叫 **CQR（conformalized quantile regression）** 区间，α = 0.10，"
        "计算顺序如下：\n"
        "1. 在 80% 的训练单元上训练两个 LightGBM 分位数模型：一个预测 10% 分位，"
        "一个预测 90% 分位，得到初始区间 `[低, 高]`。\n"
        "2. 剩下 20% 的训练单元做校准集。对每个校准点计算它被区间漏掉多少："
        "`E = max(低 − 真实值, 真实值 − 高)`；盖住时 E ≤ 0。\n"
        "3. 把 E 从小到大排序，取第 `k = ceil((n + 1) × 0.9)` 个，记为 `q`。\n"
        "4. 最终范围 = `[低 − q, 高 + q]`，再截断到 `[0, 125]`。\n\n"
        "第 3 步保证：在“和校准单元同分布”的新机器上，至少 90% 的区间会盖住真实值"
        "（有限样本保证，依赖样本可交换）。这条保证**不覆盖**换场景的情况——"
        "把场景 1 的模型套到场景 2 的机器上，覆盖率会从 0.86 掉到 0.37–0.39，"
        "详见 `docs/reports/W5.md`。\n\n"
        "实测（α = 0.10，场景 1 到 4）：覆盖 0.86–0.96，宽度 42–54 个周期，"
        "见 `reports/tables/w4_intervals.csv`。"
    )


def main() -> None:
    st.title("这台机器还能用多久？")
    st.caption(
        "输入机器的传感器读数，输出剩余周期数和一个范围。数据是 NASA 的发动机仿真数据。"
    )

    with st.expander("页面说明", expanded=True):
        st.markdown(
            "给模型看一台发动机最近几十个周期的传感器读数（温度、压力、转速），"
            "它估计这台机器还能再运行多少个周期。一个周期是它跑完一趟。"
        )
        page_map()
        st.markdown(
            "**场景 1–4** 是 NASA 数据里的四组设置：运转状态一种或六种、损坏方式一种或两种。"
            "原始代号是 `FD001`–`FD004`，与上面的场景编号一一对应。"
        )

    scenario = st.sidebar.selectbox(
        "场景",
        list(SCENARIOS),
        index=0,
        help="编号对应原始数据里的 FD001–FD004。",
    )
    subset = SCENARIOS[scenario]

    show_truth = (
        st.sidebar.radio(
            "真实值",
            ["显示", "不显示"],
            index=0,
            help="显示：用训练单元，页面上给出真实剩余周期数，可与预测对照。"
            "不显示：用测试单元，真实值不出现在页面。",
        )
        == "显示"
    )

    bundle = get_bundle(subset)
    df = get_dataset(subset, show_truth)
    units = np.sort(df["unit"].unique())
    unit = st.sidebar.selectbox(
        "机器编号",
        units.tolist(),
        index=0,
        format_func=lambda u: f"第 {u} 号机器",
        help="每台机器的总周期数不同。",
    )

    cycles = df.loc[df["unit"] == unit, "cycle"].to_numpy()
    # Start at 70% of life: at the last cycle the true RUL is 0 and the first
    # screen would look like a failed prediction.
    default_cut = int(round(int(cycles.max()) * 0.7))
    default_cut = max(int(cycles.min()) + 30, min(default_cut, int(cycles.max())))
    cut = st.sidebar.slider(
        "观测到的周期数",
        min_value=int(cycles.min()) + 29,
        max_value=int(cycles.max()),
        value=default_cut,
        step=1,
        help="从第 1 个周期到第 N 个周期。N 越小，输入模型的读数越少。",
    )
    st.sidebar.caption(f"第 {unit} 号机器的总周期数：{int(cycles.max())}。")

    view = observable_window(df, int(unit), cut)
    pred = predict_one(bundle, view.rows)
    truth = view.rul_true

    st.markdown("#### 结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "剩余周期数",
        f"{pred['rul_point']:.1f}",
        help="模型给出的一个数，单位是周期。",
    )
    if "rul_lo" in pred:
        c2.metric(
            "90% 范围",
            f"{pred['rul_lo']:.0f} – {pred['rul_hi']:.0f}",
            help="按 90% 覆盖率校准的区间，算法见底部“90% 范围的计算方法”。",
        )
        c3.metric(
            "范围宽度",
            f"{pred['rul_hi'] - pred['rul_lo']:.0f}",
            help="上界减下界，单位是周期。",
        )
    else:
        c2.metric("90% 范围", "无", help="这个模型没有区间校准。")
        c3.metric("范围宽度", "无")
    c4.metric(
        "真实值",
        "不显示" if truth is None else f"{truth:.0f}",
        help="只在“真实值：显示”时给出。",
    )

    st.caption(
        "剩余周期数：模型给出的一个数。90% 范围：按 90% 覆盖率校准的区间。"
        "范围宽度：上界减下界。真实值：仅“真实值：显示”时可用。"
    )

    if truth is not None:
        err = pred["rul_point"] - truth
        covered = pred["rul_lo"] <= truth <= pred["rul_hi"]
        st.caption(
            f"误差 {err:+.1f} 个周期（正数=预测值高于真实值）。"
            f"真实值{'在' if covered else '不在'} 90% 范围内。"
        )
    else:
        st.caption("真实值不显示，页面只给出预测。")

    left, right = st.columns(2)
    with left:
        st.pyplot(sensor_figure(view.rows))
        st.caption(
            "图 1：s_4、s_11、s_12、s_7 在观测窗口内的读数。四条曲线各自标准化，"
            "比较的是走势，不是绝对数值。"
        )
    with right:
        curve = rul_curve(subset, show_truth, int(unit), int(cycles.max()))
        if not curve.empty:
            true_life = int(cycles.max()) if view.full_life else 0
            st.pyplot(rul_figure(curve, true_life, pred["rul_point"], view.cycle))
            st.caption(
                "图 2：蓝线是点预测，蓝色带是 90% 范围，红虚线是真实值（仅“真实值：显示”时出现），"
                "黑点是当前观测周期数。"
            )
        else:
            st.info("观测窗口不足 30 个周期，无法作图。把滑杆右移。")

    with st.expander("90% 范围的计算方法"):
        range_method()

    with st.expander("这个模型在什么情况下不能用"):
        st.markdown(
            "- 数据是仿真数据，不是真机。换到真实设备要重新校准。\n"
            "- 换场景后覆盖率从 0.86 掉到 0.37–0.39，范围宽度不变——"
            "范围看起来没变，实际已经盖不住真实值。见 `docs/reports/W5.md`。\n"
            "- 预测截断在 125 个周期：超过 125 只能说“至少 125”。\n"
            "- 上传的 CSV 缺列时用中位数补齐，结果只作参考。"
        )

    with st.expander("上传传感器数据（CSV）"):
        st.markdown("需要 21 个传感器列 `s_1`–`s_21`，`cycle` 可选。用最后一个周期的数据计算。")
        up = st.file_uploader("选择 CSV 文件", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            try:
                frame, patched = coerce_uploaded(raw, settings_defaults(df))
            except ValueError as exc:
                st.error(str(exc))
            else:
                if patched:
                    st.warning(f"缺少设置列，已用中位数补齐：{', '.join(patched)}")
                st.write(f"共 {len(frame)} 行，用最后一个周期计算。")
                p = predict_one(bundle, frame)
                st.metric("剩余周期数", f"{p['rul_point']:.1f}")
                if "rul_lo" in p:
                    st.metric("90% 范围", f"{p['rul_lo']:.0f} – {p['rul_hi']:.0f}")


main()
