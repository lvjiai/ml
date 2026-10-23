"""W3 deliverable: LSTM sequence model, same split and test protocol as W2.

    python scripts/w3_lstm.py --subsets FD001
    python scripts/w3_lstm.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.schema import SENSORS  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.models.lstm import build_sequences, last_window, predict, train_lstm  # noqa: E402

WINDOW = 30
VAL_FRAC = 0.15  # of the fit units, used only for early stopping


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W3: LSTM RUL model")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args(argv)

    cfg = Config()
    if args.data_dir:
        cfg = cfg.replace(data_dir=args.data_dir)
    if args.subsets:
        cfg = cfg.replace(subsets=tuple(args.subsets))

    tab_dir = cfg.reports_dir / "tables"
    art_dir = cfg.artifacts_dir
    pred_dir = art_dir / "preds"
    for d in (tab_dir, art_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for subset in cfg.subsets:
        t0 = time.time()
        ds = load_subset(subset, cfg.data_dir)
        train = ds.train.copy()
        train["rul"] = (
            train.groupby("unit")["cycle"].transform("max") - train["cycle"]
        ).clip(upper=cfg.rul_clip).astype("int32")

        norm = ConditionNormalizer(n_clusters=N_CONDITIONS[subset], seed=cfg.seed).fit(train)
        tr = norm.transform(train)
        te = norm.transform(ds.test)

        fit_units, _calib_units = split_units(
            train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed
        )
        fit = tr[tr["unit"].isin(fit_units)]
        inner_units, val_units = split_units(
            fit["unit"].to_numpy(), 1.0 - VAL_FRAC, cfg.seed
        )

        Xtr, ytr, meta_tr = build_sequences(fit[fit["unit"].isin(inner_units)], SENSORS, WINDOW)
        Xva, yva, _ = build_sequences(fit[fit["unit"].isin(val_units)], SENSORS, WINDOW)
        Xte, meta_te = last_window(te, SENSORS, WINDOW)

        print(f"[{subset}] train {Xtr.shape} val {Xva.shape} test {Xte.shape}", flush=True)
        model, history, best_epoch = train_lstm(
            Xtr, ytr, Xva, yva, epochs=args.epochs, seed=cfg.seed
        )
        pred = np.clip(predict(model, Xte), 0.0, float(cfg.rul_clip))
        y = ds.rul.astype(np.float64)

        rows.append({"subset": subset, "model": "lstm", **summarize(y, pred)})
        pd.DataFrame(
            {
                "unit": meta_te["unit"].to_numpy(),
                "cycle": meta_te["cycle"].to_numpy(),
                "y_true": y,
                "pred_lstm": pred,
            }
        ).to_csv(pred_dir / f"w3_preds_{subset}.csv", index=False)

        torch.save(model.state_dict(), art_dir / f"lstm_{subset}.pt")
        (art_dir / f"lstm_{subset}.json").write_text(
            json.dumps(
                {
                    "subset": subset,
                    "window": WINDOW,
                    "hidden": 64,
                    "epochs_run": len(history),
                    "best_epoch": best_epoch,
                    "val_rmse_history": [round(h, 3) for h in history],
                    "seed": cfg.seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{subset}] done in {time.time() - t0:.1f}s, epochs={len(history)} "
            f"(best {best_epoch}), first/last val RMSE "
            f"{history[0]:.2f}/{min(history):.2f}",
            flush=True,
        )

    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "w3_results.csv", index=False)
    print("\n=== W3 LSTM results (test last cycle) ===")
    print(tab.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
