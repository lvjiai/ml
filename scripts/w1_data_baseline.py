"""W1 deliverable: data pipeline + EDA + naive baselines.

Run from the project root:

    python scripts/w1_data_baseline.py
    python scripts/w1_data_baseline.py --subsets FD001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_all  # noqa: E402
from src.data.schema import SENSORS, SETTINGS  # noqa: E402
from src.eval import plots  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.features.rul import add_rul  # noqa: E402
from src.models.base import (  # noqa: E402
    constant_value,
    fit_cycle_linear,
    fit_ridge,
    last_cycle_rows,
)

FEATURE_COLS = ["cycle", *SETTINGS, *SENSORS]


def dataset_table(datasets: dict) -> pd.DataFrame:
    rows = []
    for name, ds in datasets.items():
        lengths = ds.train.groupby("unit")["cycle"].max()
        const = [c for c in SENSORS if ds.train[c].std() < 1e-6]
        rows.append(
            {
                "subset": name,
                "train_rows": len(ds.train),
                "test_rows": len(ds.test),
                "train_units": ds.train["unit"].nunique(),
                "test_units": ds.test["unit"].nunique(),
                "median_life": float(lengths.median()),
                "max_life": int(lengths.max()),
                "const_sensors": ",".join(const) if const else "-",
            }
        )
    return pd.DataFrame(rows)


def baseline_table(datasets: dict, clip: int = 125) -> pd.DataFrame:
    rows = []
    for name, ds in datasets.items():
        train = add_rul(ds.train)
        test_last = last_cycle_rows(ds.test)
        y = ds.rul.astype(np.float64)

        def score(model_name: str, p: np.ndarray) -> None:
            # RUL cannot be negative and never exceeds the training clip; an
            # unclipped linear extrapolation would otherwise blow up NASA score.
            p = np.clip(p, 0.0, float(clip))
            rows.append({"subset": name, "model": model_name, **summarize(y, p)})

        # 1) constant: the median training RUL
        score("constant", np.full(len(test_last), constant_value(train)))

        # 2) cycle-only linear regression
        m_cycle = fit_cycle_linear(train)
        score("cycle_linear", m_cycle.predict(test_last[["cycle"]].to_numpy()))

        # 3) ridge on raw features (no window engineering)
        m_ridge = fit_ridge(train, FEATURE_COLS)
        score("ridge_raw", m_ridge.predict(test_last[FEATURE_COLS].to_numpy()))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W1: data pipeline, EDA, naive baselines")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--subsets", nargs="+", default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    if args.data_dir:
        cfg = cfg.replace(data_dir=args.data_dir)
    if args.subsets:
        cfg = cfg.replace(subsets=tuple(args.subsets))

    print(f"[data] reading from {cfg.data_dir}")
    datasets = load_all(cfg)
    print("[data] validation passed for: " + ", ".join(datasets))

    fig_dir = cfg.reports_dir / "figures"
    tab_dir = cfg.reports_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    ds_tab = dataset_table(datasets)
    ds_tab.to_csv(tab_dir / "w1_dataset.csv", index=False)
    print("\n=== dataset ===")
    print(ds_tab.to_string(index=False))

    for name, ds in datasets.items():
        plots.plot_trajectories(ds.train, name, fig_dir)
        plots.plot_trajectory_lengths(ds.train, name, fig_dir)
        plots.plot_sensor_std(ds.train, name, fig_dir)
    print(f"\n[eda] figures written to {fig_dir}")

    base_tab = baseline_table(datasets, clip=cfg.rul_clip)
    base_tab.to_csv(tab_dir / "w1_baseline.csv", index=False)
    print("\n=== naive baselines (evaluated at last cycle of each test unit) ===")
    print(base_tab.round(2).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
