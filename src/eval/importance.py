"""Feature-importance bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_importance(imp: pd.Series, name: str, out_dir: Path, top: int = 20) -> Path:
    top_imp = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * top + 1.0))
    ax.barh(range(len(top_imp)), top_imp.values, color="#4C72B0")
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp.index, fontsize=7)
    ax.set_xlabel("gain")
    ax.set_title(f"{name}: top {len(top_imp)} features", fontsize=9)
    fig.tight_layout()
    out = out_dir / f"importance_{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
