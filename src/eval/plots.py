"""EDA plots for the C-MAPSS subsets (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.schema import SENSORS  # noqa: E402

_TRAJ_SENSORS = ["s_2", "s_4", "s_7", "s_11", "s_12"]


def plot_trajectories(
    train: pd.DataFrame, name: str, out_dir: Path, n_units: int = 5
) -> Path:
    """Sensor evolution over the life of a few engines (z-scored per sensor).

    Sensors live on wildly different scales (s_4 ~1400 vs s_11 ~47), so raw
    units hide the trends; each sensor is normalised by its train mean/std.
    """
    units = sorted(train["unit"].unique())[:n_units]
    stats = {c: (float(train[c].mean()), float(train[c].std()) or 1.0) for c in _TRAJ_SENSORS}
    fig, axes = plt.subplots(len(units), 1, figsize=(9, 2.0 * len(units)), sharex=True)
    if len(units) == 1:
        axes = [axes]
    for ax, unit in zip(axes, units):
        sub = train[train["unit"] == unit]
        for col in _TRAJ_SENSORS:
            mu, sd = stats[col]
            ax.plot(sub["cycle"], (sub[col] - mu) / sd, label=col, linewidth=1.0)
        ax.set_ylabel(f"unit {unit}", fontsize=8)
        ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, ncol=5, loc="upper center", frameon=False)
    axes[-1].set_xlabel("cycle")
    fig.suptitle(f"{name}: sensor trajectories, z-scored per sensor (first {len(units)} units)", fontsize=10)
    fig.tight_layout()
    out = out_dir / f"eda_trajectories_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_trajectory_lengths(train: pd.DataFrame, name: str, out_dir: Path) -> Path:
    """Distribution of engine lifetimes: how much of the run-to-failure is seen."""
    lengths = train.groupby("unit")["cycle"].max()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lengths, bins=25, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(lengths.median(), color="#C44E52", linestyle="--", linewidth=1,
               label=f"median = {lengths.median():.0f}")
    ax.set_xlabel("life length (cycles)")
    ax.set_ylabel("engines")
    ax.set_title(f"{name}: trajectory lengths (n={len(lengths)})", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f"eda_lengths_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_sensor_std(train: pd.DataFrame, name: str, out_dir: Path) -> Path:
    """Which sensors carry signal: standard deviation of every sensor."""
    stds = train[SENSORS].std().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#C44E52" if v < 1e-6 else "#4C72B0" for v in stds.values]
    bars = ax.bar(range(len(stds)), stds.values, color=colors)
    ax.set_xticks(range(len(stds)))
    ax.set_xticklabels(stds.index, rotation=90, fontsize=7)
    for label, v in zip(ax.get_xticklabels(), stds.values):
        if v < 1e-6:
            label.set_color("#C44E52")
    ax.set_ylabel("std (raw units)")
    ax.set_title(f"{name}: sensor standard deviation (red = constant, zero height)", fontsize=9)
    fig.tight_layout()
    out = out_dir / f"eda_sensor_std_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
