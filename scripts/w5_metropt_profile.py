"""Profile the MetroPT-3 real-world dataset (W5).

Reads the 218 MB CSV once and writes a JSON profile plus a small markdown
summary. The profile is the factual basis for the W5 transfer protocol: how
many rows, what time span, how regular the sampling is, and how much signal
is available before each documented air-leak event.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV = Path(r"E:\新建文件夹\rul-cmapss\metropt\MetroPT3.csv")
OUT_JSON = PROJECT_ROOT / "reports" / "tables" / "w5_metropt_profile.json"
OUT_MD = PROJECT_ROOT / "reports" / "w5_metropt_profile.md"

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = [
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses",
]

# Failure table transcribed from the shipped PDF (Data Description_Metro.pdf).
FAILURES = [
    ("#1", "2020-04-18 00:00", "2020-04-18 23:59"),
    ("#2", "2020-05-29 23:30", "2020-05-30 06:00"),
    ("#3", "2020-06-05 10:00", "2020-06-07 14:30"),
    ("#4", "2020-07-15 14:30", "2020-07-15 19:00"),
]


def main() -> None:
    dtypes = {c: "float32" for c in ANALOG}
    dtypes.update({c: "int8" for c in DIGITAL})
    df = pd.read_csv(CSV, dtype=dtypes, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    dt = df["timestamp"].diff().dt.total_seconds().dropna()
    gaps = dt[dt > 60]

    profile: dict = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "t_min": str(df["timestamp"].iloc[0]),
        "t_max": str(df["timestamp"].iloc[-1]),
        "span_days": float((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400),
        "na": {c: int(df[c].isna().sum()) for c in df.columns},
        "interval_seconds": {
            "mode": float(dt.mode().iloc[0]),
            "mode_share": float((dt == dt.mode().iloc[0]).mean()),
            "median": float(dt.median()),
            "p99": float(dt.quantile(0.99)),
            "max": float(dt.max()),
            "share_gt_60s": float((dt > 60).mean()),
        },
        "gaps_gt_60s": {
            "count": int(len(gaps)),
            "gt_1h": int((gaps > 3600).sum()),
            "max_seconds": float(gaps.max()) if len(gaps) else 0.0,
        },
        "analog_stats": {
            c: {
                "min": float(df[c].min()),
                "mean": float(df[c].mean()),
                "max": float(df[c].max()),
                "std": float(df[c].std()),
                "n_unique": int(df[c].nunique()),
            }
            for c in ANALOG
        },
        "digital_share_on": {c: float(df[c].mean()) for c in DIGITAL},
        "rows_per_month": {
            str(k): int(v) for k, v in df["timestamp"].dt.to_period("M").value_counts().sort_index().items()
        },
        "failures": [],
    }

    for name, start, end in FAILURES:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        in_win = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)]
        pre = df[(df["timestamp"] >= s - pd.Timedelta(days=7)) & (df["timestamp"] < s)]
        entry = {
            "id": name,
            "start": start,
            "end": end,
            "duration_h": float((e - s).total_seconds() / 3600),
            "rows_in_window": int(len(in_win)),
            "pre_7d_rows": int(len(pre)),
        }
        for c in ANALOG:
            entry[f"{c}_pre_mean"] = float(pre[c].mean())
            entry[f"{c}_win_mean"] = float(in_win[c].mean()) if len(in_win) else float("nan")
        profile["failures"].append(entry)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# MetroPT-3 数据侧写（自动生成）",
        "",
        f"- 行数：{profile['rows']:,}；列数：{len(profile['columns'])}",
        f"- 时间范围：{profile['t_min']} → {profile['t_max']}（{profile['span_days']:.1f} 天）",
        f"- 采样间隔：众数 {profile['interval_seconds']['mode']:.0f} s（占 {profile['interval_seconds']['mode_share']:.1%}），"
        f"中位 {profile['interval_seconds']['median']:.0f} s，p99 {profile['interval_seconds']['p99']:.0f} s，"
        f"最大 {profile['interval_seconds']['max']:.0f} s",
        f"- 缺失：{'无' if not any(profile['na'].values()) else profile['na']}",
        f"- >60 s 空隙：{profile['gaps_gt_60s']['count']} 个，其中 >1 h {profile['gaps_gt_60s']['gt_1h']} 个，"
        f"最长 {profile['gaps_gt_60s']['max_seconds']:.0f} s",
        "",
        "| 信号 | min | mean | max | std | 唯一值 |",
        "|---|---|---|---|---|---|",
    ]
    for c, s in profile["analog_stats"].items():
        lines.append(f"| {c} | {s['min']:.3f} | {s['mean']:.3f} | {s['max']:.3f} | {s['std']:.3f} | {s['n_unique']} |")
    lines += ["", "| 事件 | 时长 (h) | 窗口行数 | 前 7 天行数 | 前 7 天 Oil_temperature 均值 | 窗口内均值 |", "|---|---|---|---|---|---|"]
    for f in profile["failures"]:
        lines.append(
            f"| {f['id']} | {f['duration_h']:.1f} | {f['rows_in_window']:,} | {f['pre_7d_rows']:,} | "
            f"{f['Oil_temperature_pre_mean']:.2f} | {f['Oil_temperature_win_mean']:.2f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: profile[k] for k in ("rows", "t_min", "t_max", "span_days", "interval_seconds", "gaps_gt_60s")}, indent=2))


if __name__ == "__main__":
    main()
