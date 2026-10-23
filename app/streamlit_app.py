"""Streamlit demo: one sensor trajectory in, RUL + interval out.

Written for a first-time visitor with no machine-learning background: every
label is plain language, ``FD001``-style codes are translated into "场景 1-4",
and each part of the page is introduced at the top.

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

# The four NASA scenarios, translated so a non-expert can pick one.
SCENARIOS = {
    "场景 1：只有一种运转状态（最好懂，推荐先玩这个）": "FD001",
    "场景 2：六种运转状态混在一起（更难）": "FD002",
    "场景 3：两种损坏方式（较难）": "FD003",
    "场景 4：六种运转状态 + 两种损坏方式（最难）": "FD004",
}
KEY_SENSORS = ("s_4", "s_11", "s_12", "s_7")

st.set_page_config(page_title="这台机器还能用多久？", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle(subset: str):
    return load_bundle(subset)


@st.cache_data(show_spinner=False)
def get_dataset(subset: str, source: str) -> pd.DataFrame:
    ds = load_subset(subset, Config().data_dir)
    if source == "看得到答案（用来检查模型准不准）":
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
    ax.set_xlabel("跑了多少个周期")
    ax.set_ylabel("读数高低（已缩放）")
    ax.set_title("图 1：机器最近这段路是怎么过的", fontsize=10)
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
    ax.plot(curve["cutoff"], curve["rul_point"], color="#1f77b4", label="模型猜的剩余寿命")
    if "rul_lo" in curve.columns:
        ax.fill_between(
            curve["cutoff"], curve["rul_lo"], curve["rul_hi"], color="#1f77b4", alpha=0.18,
            label="它给的合理范围",
        )
    if max_cycle > 0:
        ax.plot(
            curve["cutoff"], max_cycle - curve["cutoff"], color="#d62728",
            linestyle="--", linewidth=1.0, label="真实答案",
        )
    ax.axvline(cut, color="grey", linestyle=":", linewidth=1.0)
    ax.scatter([cut], [point], color="black", zorder=5, s=28)
    ax.set_xlabel("已经看到第几个周期")
    ax.set_ylabel("还能再跑多少个周期")
    ax.set_title("图 2：看到的信息越多，判断怎么变", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def page_map() -> None:
    """Tell the visitor what every region of the page does, before they scroll."""
    st.markdown(
        "| 页面上的位置 | 它是干什么的 | 你要做什么 |\n"
        "|---|---|---|\n"
        "| 左侧边栏 | 设置区 | 选一台机器，拖动滑杆决定“我看到了它多久” |\n"
        "| 上方的四个数 | 结果区 | 看模型猜它还能撑多久、范围有多宽 |\n"
        "| 下方两张图 | 证据区 | 左图看传感器怎么变，右图看判断怎么变 |\n"
        "| 底部的展开项 | 说明书 | 不懂就点开；也能上传自己的数据 |"
    )


def main() -> None:
    st.title("这台机器还能用多久？")
    st.caption(
        "你告诉它机器最近是怎么运转的，它猜这台机器还能再撑多久，并诚实地说出自己有多不确定。"
        "数据是 NASA 公开的发动机仿真数据（不是真飞机），用来演示方法。"
    )

    with st.expander("第一次用？30 秒看懂这个页面", expanded=True):
        st.markdown(
            "**一句话**：给模型看一台发动机最近几十个周期的传感器读数（温度、压力、转速等），"
            "它估计这台机器还能再运行多少个周期。一个“周期”可以理解成它又跑完一趟。"
        )
        page_map()
        st.markdown(
            "**“场景 1–4”是什么？** NASA 把这批数据按难度分成了四种情况："
            "运转状态是一种还是六种、损坏方式是两种还是一种。场景 1 最简单，"
            "场景 4 最难。原始数据里的代号是 FD001–FD004，页面已把它们翻译成场景名，"
            "你不需要记代号。"
        )

    scenario = st.sidebar.selectbox(
        "先选一个场景",
        list(SCENARIOS),
        index=0,
        help="场景越靠后，机器运转状态越复杂，模型越难猜准。建议先玩场景 1。",
    )
    subset = SCENARIOS[scenario]

    source = st.sidebar.radio(
        "这台机器的答案，你看得到吗？",
        ["看得到答案（用来检查模型准不准）", "看不到答案（更接近真实使用）"],
        index=0,
        help="看得到答案时，页面会显示真实剩余寿命，方便你对比；看不到答案时，页面只能给你模型的猜测。",
    )

    bundle = get_bundle(subset)
    df = get_dataset(subset, source)
    units = np.sort(df["unit"].unique())
    unit = st.sidebar.selectbox(
        "选一台机器",
        units.tolist(),
        index=0,
        format_func=lambda u: f"第 {u} 号机器",
        help="每台机器的寿命长短不同，可以多换几台，看看模型在任何时候是不是都靠谱。",
    )

    cycles = df.loc[df["unit"] == unit, "cycle"].to_numpy()
    cut = st.sidebar.slider(
        "假设我只看到了第几个周期",
        min_value=int(cycles.min()) + 29,
        max_value=int(cycles.max()),
        value=int(cycles.max()),
        step=1,
        help="往前拖=只看到早期（信息少，模型应该更没把握）；往后拖=看到更多（信息多，范围应该更窄）。",
    )
    st.sidebar.caption(f"这台机器完整活了 {int(cycles.max())} 个周期。")

    view = observable_window(df, int(unit), cut)
    pred = predict_one(bundle, view.rows)
    truth = view.rul_true

    st.markdown("#### 模型给出的答案")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "还能再跑多少个周期",
        f"{pred['rul_point']:.1f}",
        help="模型认为最可能的剩余周期数，是一个数。",
    )
    if "rul_lo" in pred:
        c2.metric(
            "合理范围（约 90%）",
            f"{pred['rul_lo']:.0f} – {pred['rul_hi']:.0f}",
            help="真实答案约有 90% 的概率落在这个范围里。这是统计承诺，不是保证。",
        )
        c3.metric(
            "范围有多宽",
            f"{pred['rul_hi'] - pred['rul_lo']:.0f}",
            help="范围越小，说明模型越有把握；但太小也可能是“太自信”，所以还要看它准不准。",
        )
    else:
        c2.metric("合理范围（约 90%）", "没有校准", help="这个模型没有做区间校准，只能给一个数。")
        c3.metric("范围有多宽", "—")
    c4.metric(
        "真实答案",
        "看不到" if truth is None else f"{truth:.0f}",
        help="只有“看得到答案”时才显示；看不到时这一栏是藏起来的。",
    )

    st.caption(
        "**怎么读**：第一个数是模型最可能的一个猜测；第二个数是它给的合理范围；"
        "第三个数是范围有多宽（越小越有把握，但也可能是过度自信）；第四个数是真实答案，只用来对照。"
    )

    if truth is not None:
        err = pred["rul_point"] - truth
        covered = pred["rul_lo"] <= truth <= pred["rul_hi"]
        st.caption(
            f"这一次：模型猜{f'多了 {err:.1f}' if err > 0 else f'少了 {-err:.1f}'} 个周期；"
            f"真实答案{'落在' if covered else '没落在'}它给的范围里。"
        )
    else:
        st.caption("这一次看不到真实答案，所以只能给你模型的猜测——这也是真实使用时的样子。")

    left, right = st.columns(2)
    with left:
        st.pyplot(sensor_figure(view.rows))
        st.caption("图 1：机器这段时间的传感器读数。这里只画了最关键的 4 个，并把波动缩放到同一尺度，"
                   "所以看的是“形状”（有没有明显抬头或下沉），不是绝对数值。")
    with right:
        curve = rul_curve(subset, source, int(unit), int(cycles.max()))
        if not curve.empty:
            true_life = int(cycles.max()) if view.full_life else 0
            st.pyplot(rul_figure(curve, true_life, pred["rul_point"], view.cycle))
            st.caption("图 2：看到的信息越多，模型的判断怎么变。蓝线=它的猜测，蓝色带=它给的合理范围，"
                       "红虚线=真实答案（只在“看得到答案”时出现），黑点=你现在停的位置。"
                       "理想情况：蓝线贴近红虚线，蓝带越来越窄。")
        else:
            st.info("观测窗口太短，画不出图。把滑杆往右拖一点。")

    with st.expander("我什么时候不能信它？"):
        st.markdown(
            "- **这是仿真数据，不是真飞机。** 想用到真实设备上，必须用真实数据重新校准，不能直接搬。\n"
            "- **换场景会失效。** 用场景 1 的模型去看场景 2 的机器，范围会“假装很准”："
            "名义 90%，实际只有 37–39% 能罩住真实答案。这是本项目最重要的发现，详见报告 W5。\n"
            "- **它最多只说 125 个周期。** 超过 125 就只会告诉你“至少还有 125”，因为再往后预测不可信。\n"
            "- **上传的数据可能被凑合处理。** 缺的列会用中位数补齐，结果只能参考。"
        )

    with st.expander("上传我自己的传感器数据（CSV）"):
        st.markdown(
            "需要 21 个传感器列（`s_1`–`s_21`），`cycle` 可选。上传后模型用最后一个周期做判断。"
        )
        up = st.file_uploader("选择 CSV 文件", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            try:
                frame, patched = coerce_uploaded(raw, settings_defaults(df))
            except ValueError as exc:
                st.error(str(exc))
            else:
                if patched:
                    st.warning(f"缺少这些设置列，已用中位数补齐：{', '.join(patched)}")
                st.write(f"共 {len(frame)} 行；用最后一个周期的数据判断。")
                p = predict_one(bundle, frame)
                st.metric("还能再跑多少个周期", f"{p['rul_point']:.1f}")
                if "rul_lo" in p:
                    st.metric("合理范围（约 90%）", f"{p['rul_lo']:.0f} – {p['rul_hi']:.0f}")


main()
