"""W13: per-cycle evaluation of the test engines.

The last-cycle protocol (W2-W12) scores one number per engine.  Published work
often reports every observed cycle instead, and a model that is good at the last
cycle is not automatically good two hundred cycles earlier.  This script scores
*all* observed cycles of the official test engines:

    failure_time(unit)  = last observed cycle + RUL_FD00x
    true RUL(t)         = min(failure_time - t, 125)

so the label is the same censored target the models were trained on.  Two views
are reported:

* ``late``  — cycles where the uncapped RUL is <= 125 (the regime the labels
  describe, and the only one where the metric is meaningful);
* ``all``   — every cycle after the first full window (kept for honesty: early
  cycles are almost pure noise and the models are not trained for them).

GBDT features are causal per row, so the feature frame built on the whole test
trajectory is valid at every cycle.  The LSTM uses a stride-1 window ending at
each cycle, and the deep model (when its checkpoints exist) the same.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.schema import SENSORS as LSTM_SENSORS  # noqa: E402
from src.eval.metrics import nasa_score, rmse  # noqa: E402
from src.models.lstm import RULNet, build_sequences, predict as lstm_predict  # noqa: E402
from src.pipeline import build_features, load_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W13: per-cycle evaluation")
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    return p.parse_args()


def failure_times(ds) -> pd.DataFrame:
    """unit -> last observed cycle and the implied failure time."""
    last = ds.test.groupby("unit")["cycle"].max().sort_index()
    units = last.index.to_numpy()
    assert len(units) == len(ds.rul), (len(units), len(ds.rul))
    return pd.DataFrame(
        {
            "unit": units,
            "last_cycle": last.to_numpy(),
            "failure_time": last.to_numpy() + np.asarray(ds.rul, dtype=float),
        }
    )


def score_curve(
    df: pd.DataFrame, preds: np.ndarray, cap: float, window: int
) -> list[dict]:
    """Metrics for one model's per-cycle predictions."""
    rows = df.assign(pred=np.clip(preds, 0.0, cap))
    rows["true_raw"] = rows["failure_time"] - rows["cycle"]
    rows["true"] = np.clip(rows["true_raw"], 0.0, cap)
    late = rows[rows["true_raw"] <= cap]
    allrows = rows[rows["cycle"] >= window]
    out = [
        {
            "view": "late",
            "rmse": rmse(late["true"], late["pred"]),
            "nasa": nasa_score(late["true"], late["pred"]),
            "n": len(late),
        },
        {
            "view": "all",
            "rmse": rmse(allrows["true"], allrows["pred"]),
            "nasa": nasa_score(allrows["true"], allrows["pred"]),
            "n": len(allrows),
        },
    ]
    return out


def gbdt_curve(ds, bundle, ft: pd.DataFrame, cap: float) -> tuple[np.ndarray, pd.DataFrame]:
    feat = build_features(ds.test, bundle)
    x = feat[list(bundle.features)].to_numpy()
    pred = bundle.booster.predict(x)
    merged = feat[["unit", "cycle"]].merge(ft, on="unit", how="left")
    merged["pred"] = pred
    return merged


def lstm_curve(ds, norm, subset: str, cap: float) -> tuple[np.ndarray, pd.DataFrame, int]:
    art = Path(Config().artifacts_dir)
    meta = json.loads((art / f"lstm_{subset}.json").read_text(encoding="utf-8"))
    window = int(meta["window"])
    model = RULNet(len(LSTM_SENSORS), hidden=int(meta.get("hidden", 64)))
    state = torch.load(art / f"lstm_{subset}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    te = norm.transform(ds.test).sort_values(["unit", "cycle"])
    x, _y, meta_df = build_sequences(te, LSTM_SENSORS, window, with_target=False)
    pred = lstm_predict(model, x)
    ft = failure_times(ds)
    merged = meta_df.merge(ft, on="unit", how="left")
    merged["pred"] = pred
    return merged, window


def main() -> None:
    args = parse_args()
    torch.set_num_threads(8)
    cfg = Config()
    rows: list[dict] = []
    for name in args.subsets:
        ds = load_subset(name, cfg.data_dir)
        ft = failure_times(ds)
        bundle = load_bundle(name)

        merged = gbdt_curve(ds, bundle, ft, cfg.rul_clip)
        for rec in score_curve(merged, merged["pred"].to_numpy(), cfg.rul_clip, window=1):
            rows.append({"subset": name, "model": "gbdt", **rec})

        merged_l, window = lstm_curve(ds, bundle.norm, name, cfg.rul_clip)
        for rec in score_curve(merged_l, merged_l["pred"].to_numpy(), cfg.rul_clip, window=window):
            rows.append({"subset": name, "model": "lstm", **rec})

        # equal-weight blend on the cycles both models cover
        b = merged.merge(
            merged_l[["unit", "cycle", "pred"]], on=["unit", "cycle"], suffixes=("_g", "_l")
        )
        if len(b):
            b["pred"] = (b["pred_g"] + b["pred_l"]) / 2.0
            for rec in score_curve(b, b["pred"].to_numpy(), cfg.rul_clip, window=window):
                rows.append({"subset": name, "model": "gbdt+lstm", **rec})
        print(f"[{name}] done", flush=True)

    tab = pd.DataFrame(rows)
    dest = Path(cfg.reports_dir) / "tables" / "w13_per_cycle.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(dest, index=False)
    print(tab.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
