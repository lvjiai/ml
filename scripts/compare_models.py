"""Merge W1/W2/W3 result tables into one comparison table.

    python scripts/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402

ORDER = [
    "constant",
    "cycle_linear",
    "ridge_raw",
    "gbdt_engineered",
    "ridge_engineered",
    "blend_50",
    "lstm",
    "lstm_blend50_avg",
    "lstm0.4_blend0.6",
]


def main() -> int:
    cfg = Config()
    tab_dir = cfg.reports_dir / "tables"
    frames = []
    for name in (
        "w1_baseline.csv",
        "w2_results.csv",
        "w3_results.csv",
        "w3_ensemble.csv",
    ):
        path = tab_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        print("no result tables found")
        return 1

    all_res = pd.concat(frames, ignore_index=True)
    all_res.to_csv(tab_dir / "all_results.csv", index=False)

    for metric in ("rmse", "nasa_score"):
        pivot = all_res.pivot_table(index="model", columns="subset", values=metric)
        pivot = pivot.reindex([m for m in ORDER if m in pivot.index])
        print(f"\n=== {metric} ===")
        print(pivot.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
