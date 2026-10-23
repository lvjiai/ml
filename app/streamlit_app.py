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


def main() -> None:
    st.title("发动机剩余寿命：点预测 + 置信区间")
    st.caption(
        "C-MAPSS 仿真数据。选一台发动机、决定只看前多少个周期，模型给出剩余寿命和 90% 区间。"
    )

    with st.sidebar:
        st.header("输入")
        subset = st.selectbox("子集", SUBSETS, index=0)
        source = st.radio(
            "数据来源", ["训练单元（有真实 RUL）", "测试单元（真实 RUL 隐藏）"], index=0
        )
        bundle = get_bundle(subset)
        df = get_dataset(subset, source)
        units = np.sort(df["unit"].unique())
        unit = st.selectbox("发动机编号", units.tolist(), index=0)
        cycles = df.loc[df["unit"] == unit, "cycle"].to_numpy()
        default_cut = int(cycles.max())
        cut = st.slider(
            "只看前多少个周期",
            min_value=int(cycles.min()) + 29,
            max_value=int(cycles.max()),
            value=default_cut,
            step=1,
            help="30 个周期以下特征窗口构不成，所以下限是 30。",
        )
        st.caption(f"这台机器完整寿命 {int(cycles.max())} 个周期")

    view = observable_window(df, int(unit), cut)
    pred = predict_one(bundle, view.rows)
    truth = view.rul_true

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("剩余寿命（点预测）", f"{pred['rul_point']:.1f}")
    if "rul_lo" in pred:
        c2.metric("90% 区间", f"{pred['rul_lo']:.0f} – {pred['rul_hi']:.0f}")
        c3.metric("区间宽度", f"{pred['rul_hi'] - pred['rul_lo']:.0f}")
    else:
        c2.metric("90% 区间", "未校准")
        c3.metric("区间宽度", "—")
    c4.metric("真实剩余寿命", "—" if truth is None else f"{truth:.0f}")

    if truth is not None:
        err = pred["rul_point"] - truth
        st.caption(f"点预测误差 {err:+.1f} 周期；区间{'覆盖' if pred['rul_lo'] <= truth <= pred['rul_hi'] else '未覆盖'}真值。")
    else:
        st.caption("测试单元的真实剩余寿命不在数据里（只在 RUL_FD00x 中给出末周期标签），故不显示。")

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
