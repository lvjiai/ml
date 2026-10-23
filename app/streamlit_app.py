"""Streamlit demo: one sensor trajectory in, RUL + interval out.

Run with:  streamlit run app/streamlit_app.py

Everything inference-related goes through ``src.pipeline`` — the app is a thin
shell, so the numbers here are exactly the ones the report quotes.
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

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
KEY_SENSORS = ("s_4", "s_11", "s_12", "s_7")

st.set_page_config(page_title="RUL 预测与区间", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle(subset: str):
    return load_bundle(subset)


@st.cache_data(show_spinner=False)
def get_dataset(subset: str, source: str) -> pd.DataFrame:
    ds = load_subset(subset, Config().data_dir)
    if source == "训练单元（有真实 RUL）":
        return with_targets(ds.train, Config().rul_clip)
    return ds.test


@st.cache_data(show_spinner=False)
def rul_curve(subset: str, source: str, unit: int, max_cycle: int, n_points: int = 18):
    """Predicted RUL as the observed window grows (shows the interval narrowing)."""
    bundle = get_bundle(subset)
    df = get_dataset(subset, source)
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
    for name, col in zip(KEY_SENSORS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")):
        s = rows[name]
        z = (s - s.mean()) / (s.std() or 1.0)
        ax.plot(rows["cycle"], z, label=name, color=col, linewidth=1.2)
    ax.set_xlabel("cycle")
    ax.set_ylabel("z-score")
    ax.set_title("观测窗口内的传感器（按自身标准化）", fontsize=10)
    ax.legend(fontsize=7, ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def rul_figure(
    curve: pd.DataFrame, max_cycle: int, point: float, cut: int
) -> plt.Figure:
    """Predicted vs true RUL along the observation window.

    The true RUL of a train unit at cutoff t is ``max_cycle - t`` — a line that
    falls with time, unlike the prediction which rises as evidence accumulates.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=130)
    ax.plot(curve["cutoff"], curve["rul_point"], color="#1f77b4", label="点预测")
    if "rul_lo" in curve.columns:
        ax.fill_between(
            curve["cutoff"], curve["rul_lo"], curve["rul_hi"], color="#1f77b4", alpha=0.18,
            label="90% 区间",
        )
    if max_cycle > 0:
        ax.plot(
            curve["cutoff"], max_cycle - curve["cutoff"], color="#d62728",
            linestyle="--", linewidth=1.0, label="真实 RUL",
        )
    ax.axvline(cut, color="grey", linestyle=":", linewidth=1.0)
    ax.scatter([cut], [point], color="black", zorder=5, s=28)
    ax.set_xlabel("观测到第几个周期")
    ax.set_ylabel("剩余寿命（周期）")
    ax.set_title("随着观测变多，预测怎么变", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def intro() -> None:
    """Plain-language explanation for a first-time visitor."""
    st.markdown(
        "**一句话**：给模型看一台发动机最近 N 个周期的传感器读数（温度、压力、转速等），"
        "它估计这台机器还能再运行多少个周期，并给出一个范围。"
        "\n\n"
        "这里的周期可以理解成“又跑完一趟”；数据是 NASA 公开的 **C-MAPSS 仿真数据**（不是真机），"
        "用来验证方法。换成真实设备，必须用真实数据重新校准，不能直接搬。"
    )

    st.markdown("#### 这个页面怎么用（三步）")
    st.markdown(
        "1. **左边选一台机器**，再拖动“只看前多少个周期”——相当于告诉模型“我只知道它跑了这么久，"
        "后面的读数还没看到”。\n"
        "2. **看上面四个数字**：模型猜它还能跑多久、这个猜测的范围有多宽、以及（仅训练单元）真实答案。\n"
        "3. **看下面两张图**：左边是传感器随时间的变化；右边是“证据变多时，预测怎么变”。"
    )


def read_metrics() -> None:
    """Explain what the four numbers mean, right where they are shown."""
    st.markdown("#### 四个数字怎么读")
    st.markdown(
        "- **剩余寿命（点预测）**：模型认为最可能的剩余周期数，单位是周期。\n"
        "- **90% 区间**：真实剩余寿命有约 90% 的概率落在这个范围里。它是一个统计承诺，"
        "不是“绝对安全”；模型被验证过才可信。\n"
        "- **区间宽度**：模型有多不确定。越窄越“敢下结论”，但如果模型没校准好，"
        "窄区间可能是过度自信——这正是本项目要用真实数据检验的东西。\n"
        "- **真实剩余寿命**：只有“训练单元”才有答案，用来和预测对照；"
        "测试单元的答案被藏起来了。"
    )


def legend_notes() -> None:
    """One-line legends for the two figures."""
    st.markdown(
        "**左图（传感器）**：只看被观测到的那一段，每条曲线是一种传感器（这里只画了最关键的 4 个，"
        "并按各自的波动幅度标准化，方便放在一张图上看趋势）。晚期通常能看到曲线明显抬头/下沉，"
        "那就是退化的信号。\n\n"
        "**右图（预测怎么变）**：横轴是“看到了第几个周期”，纵轴是剩余寿命。蓝线=模型的点预测，"
        "蓝色带=90% 区间；**红色虚线**=真实剩余寿命（只有训练单元有），它随时间线性下降；"
        "黑色圆点=你现在拖到的位置。理想情况：蓝线越来越贴近红色虚线，蓝带越来越窄。"
    )


def main() -> None:
    st.title("这台机器还能用多久？——剩余寿命预测 demo")
    st.caption(
        "C-MAPSS 仿真数据｜输入一段传感器读数，输出剩余寿命（RUL）+ 90% 区间。"
        "所有数字都由仓库里的模型实时算出，不是写死的。"
    )

    with st.expander("这是什么？怎么看？（30 秒读完）", expanded=True):
        intro()
        read_metrics()

    with st.sidebar:
        st.header("输入")
        st.caption("改这里的选项，右侧的结果会立刻重新计算。")
        subset = st.selectbox("子集", SUBSETS, index=0, help="C-MAPSS 的四种工况组合，FD001 最简单，FD004 最难。")
        source = st.radio(
            "数据来源",
            ["训练单元（有真实 RUL）", "测试单元（真实 RUL 隐藏）"],
            index=0,
            help="训练单元能看到真实剩余寿命，适合对照；测试单元只有末周期标签，和不看答案的真实使用更接近。",
        )
        bundle = get_bundle(subset)
        df = get_dataset(subset, source)
        units = np.sort(df["unit"].unique())
        unit = st.selectbox("发动机编号", units.tolist(), index=0, help="每台发动机的寿命长短不同，可以多换几台看。")
        cycles = df.loc[df["unit"] == unit, "cycle"].to_numpy()
        default_cut = int(cycles.max())
        cut = st.slider(
            "只看前多少个周期",
            min_value=int(cycles.min()) + 29,
            max_value=int(cycles.max()),
            value=default_cut,
            step=1,
            help="相当于“我只看到这里”。往前拖=证据更少、区间更宽；往后拖=证据更多。30 个周期以下构不成特征窗口。",
        )
        st.caption(f"这台机器完整寿命 {int(cycles.max())} 个周期")

    view = observable_window(df, int(unit), cut)
    pred = predict_one(bundle, view.rows)
    truth = view.rul_true

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "剩余寿命（点预测）", f"{pred['rul_point']:.1f}", help="模型认为最可能的剩余周期数（单位：周期）。"
    )
    if "rul_lo" in pred:
        c2.metric(
            "90% 区间",
            f"{pred['rul_lo']:.0f} – {pred['rul_hi']:.0f}",
            help="真实剩余寿命约有 90% 的概率落在这个范围；它是统计承诺，不是保证。",
        )
        c3.metric(
            "区间宽度",
            f"{pred['rul_hi'] - pred['rul_lo']:.0f}",
            help="模型有多不确定：越窄越敢下结论，但过窄也可能是过度自信。",
        )
    else:
        c2.metric("90% 区间", "未校准", help="该模型没有校准工件，只有点预测。")
        c3.metric("区间宽度", "—")
    c4.metric(
        "真实剩余寿命",
        "—" if truth is None else f"{truth:.0f}",
        help="只有训练单元能显示；测试单元的答案是隐藏的。",
    )

    if truth is not None:
        err = pred["rul_point"] - truth
        st.caption(
            f"点预测误差 {err:+.1f} 周期（负=猜少了，正=猜多了）；"
            f"区间{'覆盖' if pred['rul_lo'] <= truth <= pred['rul_hi'] else '未覆盖'}真值。"
        )
    else:
        st.caption("测试单元的真实剩余寿命不在数据里（只在 RUL_FD00x 中给出末周期标签），故不显示。")

    with st.expander("两张图怎么看", expanded=False):
        legend_notes()

    left, right = st.columns(2)
    with left:
        st.pyplot(sensor_figure(view.rows))
    with right:
        curve = rul_curve(subset, source, int(unit), int(cycles.max()))
        if not curve.empty:
            true_life = int(cycles.max()) if view.full_life else 0
            st.pyplot(rul_figure(curve, true_life, pred["rul_point"], view.cycle))
        else:
            st.info("观测窗口太短，画不出曲线。")

    with st.expander("上传自己的曲线（CSV）"):
        up = st.file_uploader("需要 21 个传感器列；cycle 可选", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            try:
                frame, patched = coerce_uploaded(raw, settings_defaults(df))
            except ValueError as exc:
                st.error(str(exc))
            else:
                if patched:
                    st.warning(f"CSV 缺少设置列，已用本子集的中位数补齐：{', '.join(patched)}")
                st.write(f"共 {len(frame)} 行；用最后一个周期的特征推理。")
                p = predict_one(bundle, frame)
                st.metric("剩余寿命（点预测）", f"{p['rul_point']:.1f}")
                if "rul_lo" in p:
                    st.metric("90% 区间", f"{p['rul_lo']:.0f} – {p['rul_hi']:.0f}")

    with st.expander("这个 demo 的边界"):
        st.markdown(
            "- 区间是 **CQR** 校准的 90% 区间，校准集来自训练单元，覆盖率见报告 W4。\n"
            "- 换工况（例如用 FD001 的模型看 FD002 的机器）区间会失效，见报告 W5："
            "名义 90%，实际只有 37–39%。\n"
            "- 预测值在 125 处截断：超过 125 个周期只能说\"至少还有 125\"。\n"
            "- 上传的曲线若没有设置列，用中位数补齐，工况归一化会失真，结果仅供参考。"
        )


main()
