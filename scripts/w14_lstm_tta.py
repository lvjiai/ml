"""W14: LSTM ensemble over window lengths + test-time augmentation (TTA).

Motivation: the single LSTM is sensitive to the window length and to the exact
last reading.  Two cheap, protocol-preserving fixes:

* **window ensemble** — train the same architecture on windows 20/30/50 and
  average (all three see the same fit units, the same early-stopping split and
  the same test engines);
* **TTA** — at prediction time, average the model over the last window shifted
  by -2..+2 cycles (edge rows repeated), which smooths single-cycle noise.

The split, the censored target and the test engines are identical to W2-W12, so
any change in the last-cycle RMSE is attributable to the ensemble/TTA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.schema import SENSORS  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.metrics import nasa_score, rmse  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.features.rul import add_rul  # noqa: E402
from src.models.lstm import (  # noqa: E402
    RULNet,
    build_sequences,
    last_window,
    predict,
    train_lstm,
)

WINDOWS = (20, 30, 50)
VAL_FRAC = 0.15
SHIFTS = (-2, -1, 0, 1, 2)


def tta_predict(model: RULNet, x: np.ndarray) -> np.ndarray:
    """Average predictions over shifted copies of each window."""
    preds = []
    for k in SHIFTS:
        if k == 0:
            xs = x
        else:
            xs = np.empty_like(x)
            if k > 0:
                xs[:, :-k] = x[:, k:]
                xs[:, -k:] = x[:, -1:]
            else:
                xs[:, -k:] = x[:, :k]
                xs[:, :-k] = x[:, :1]
        preds.append(predict(model, xs))
    return np.mean(preds, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W14: LSTM window ensemble + TTA")
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(8)
    cfg = Config()
    rows = []
    for name in args.subsets:
        ds = load_subset(name, cfg.data_dir)
        train = ds.train
        fit_units, calib_units = split_units(
            train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed
        )
        norm = ConditionNormalizer(n_clusters=N_CONDITIONS[name], seed=cfg.seed).fit(train)
        fit = norm.transform(add_rul(train[train["unit"].isin(fit_units)], cfg.rul_clip))
        tr_units, va_units = split_units(fit["unit"].to_numpy(), 1.0 - VAL_FRAC, cfg.seed + 1)
        te = norm.transform(ds.test)
        y_test = np.minimum(ds.rul.astype(float), cfg.rul_clip)

        ca = norm.transform(add_rul(train[train["unit"].isin(calib_units)], cfg.rul_clip))

        preds_test = []
        per_window = {}
        for window in WINDOWS:
            model = None
            for seed in args.seeds:
                Xtr, ytr, _ = build_sequences(
                    fit[fit["unit"].isin(tr_units)], SENSORS, window
                )
                Xva, yva, _ = build_sequences(
                    fit[fit["unit"].isin(va_units)], SENSORS, window
                )
                model, _hist, _ep = train_lstm(
                    Xtr, ytr, Xva, yva, epochs=args.epochs, seed=seed
                )
            x_te, meta_te = last_window(te, SENSORS, window)
            p_te = np.clip(tta_predict(model, x_te), 0.0, cfg.rul_clip)
            if len(p_te) != len(y_test):
                raise RuntimeError(f"{name} w{window}: {len(p_te)} != {len(y_test)}")
            preds_test.append(p_te)
            per_window[window] = (p_te, meta_te)
            x_ca, y_ca = calib_windows(ca, calib_units, window, cfg.rul_clip)
            p_ca = np.clip(tta_predict(model, x_ca), 0.0, cfg.rul_clip)
            rows.append(
                {
                    "subset": name,
                    "model": f"lstm_w{window}_tta",
                    "rmse": rmse(y_test, p_te),
                    "nasa": nasa_score(y_test, p_te),
                    "calib_rmse": rmse(y_ca, p_ca),
                }
            )
            print(f"[{name}] w{window}: {rmse(y_test, p_te):.3f}", flush=True)

        ens = np.mean(preds_test, axis=0)
        calib_best = min(
            (r for r in rows if r["subset"] == name and "w20+30+50" not in r["model"]),
            key=lambda r: r["calib_rmse"],
        )
        rows.append(
            {
                "subset": name,
                "model": "lstm_w20+30+50_tta",
                "rmse": rmse(y_test, ens),
                "nasa": nasa_score(y_test, ens),
                "calib_rmse": calib_best["calib_rmse"],
            }
        )
        print(f"[{name}] ensemble: {rmse(y_test, ens):.3f}", flush=True)

        for window, (p_win, meta_win) in per_window.items():
            pd.DataFrame(
                {"unit": meta_win["unit"].to_numpy(), "pred": p_win, "true": y_test}
            ).to_csv(
                Path(cfg.artifacts_dir) / "preds" / f"w14_preds_w{window}_{name}.csv",
                index=False,
            )
        out = Path(cfg.artifacts_dir) / "preds" / f"w14_preds_{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"unit": meta_te["unit"].to_numpy(), "pred": ens, "true": y_test}
        ).to_csv(out, index=False)

    tab = pd.DataFrame(rows)
    dest = Path(cfg.reports_dir) / "tables" / "w14_lstm_tta.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(dest, index=False)
    print(tab.round(2).to_string(index=False))


def calib_windows(
    df: pd.DataFrame, units: np.ndarray, window: int, cap: float, stride: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated windows of the calibration units with their capped labels.

    Each unit is cut at ``t`` every ``stride`` cycles; the label is
    ``min(last_cycle - t, cap)`` because calibration units run to failure.
    Only cutoffs within ``cap`` cycles of failure are kept — beyond that the
    label is a censored constant and the residual says nothing about the model.
    """
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for unit in np.sort(np.unique(units)):
        g = df[df["unit"] == unit].sort_values("cycle")
        values = g[SENSORS].to_numpy(dtype=np.float32)
        cycles = g["cycle"].to_numpy()
        last = int(cycles[-1])
        start = max(int(cycles[0]) + window - 1, last - cap)
        for t in range(start, last + 1, stride):
            idx = int(np.searchsorted(cycles, t, side="right"))
            if idx < window:
                continue
            xs.append(values[idx - window : idx])
            ys.append(min(last - int(cycles[idx - 1]), cap))
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


if __name__ == "__main__":
    main()
