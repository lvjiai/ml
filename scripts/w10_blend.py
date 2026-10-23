"""W10: equal-weight blends of the independent models (no test-set tuning).

Combining a flat GBDT, a per-condition mixture of experts and an LSTM averages
three different inductive biases; equal weights keep the choice out of the test
set. Everything is read from the saved prediction files of W2/W3/W9.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402

PRED_DIR = Config().artifacts_dir / "preds"


def load(subset: str) -> pd.DataFrame:
    w2 = pd.read_csv(PRED_DIR / f"w2_preds_{subset}.csv")
    w3 = pd.read_csv(PRED_DIR / f"w3_preds_{subset}.csv")
    w9 = pd.read_csv(PRED_DIR / f"w9_preds_{subset}.csv")
    df = (
        w9.rename(columns={"gbdt": "pred_gbdt9", "moe": "pred_moe", "moe_linear_tree": "pred_moe_lin"})
        [["unit", "cycle", "y_true", "pred_gbdt9", "pred_moe", "pred_moe_lin"]]
        .merge(w2[["unit", "pred_gbdt", "pred_blend"]], on="unit", how="left")
        .merge(w3[["unit", "pred_lstm"]], on="unit", how="left")
    )
    return df


def main() -> int:
    cfg = Config()
    rows = []
    for subset in cfg.subsets:
        df = load(subset)
        y = df["y_true"].to_numpy(dtype=float)

        candidates = {
            "gbdt": df["pred_gbdt"].to_numpy(),
            "moe": df["pred_moe"].to_numpy(),
            "moe_linear": df["pred_moe_lin"].to_numpy(),
            "lstm": df["pred_lstm"].to_numpy(),
            "gbdt+lstm": (df["pred_gbdt"] + df["pred_lstm"]).to_numpy() / 2,
            "moe+lstm": (df["pred_moe"] + df["pred_lstm"]).to_numpy() / 2,
            "moe_lin+lstm": (df["pred_moe_lin"] + df["pred_lstm"]).to_numpy() / 2,
            "gbdt+moe+lstm": (df["pred_gbdt"] + df["pred_moe"] + df["pred_lstm"]).to_numpy() / 3,
            "gbdt+ridge+lstm": (df["pred_gbdt"] + df["pred_blend"] + df["pred_lstm"]).to_numpy() / 3,
        }
        for name, pred in candidates.items():
            s = summarize(y, np.clip(pred, 0, 125))
            rows.append({"subset": subset, "model": name, "rmse": s["rmse"],
                         "nasa_score": s["nasa_score"], "mae_0_30": s["mae_0_30"]})

    out = pd.DataFrame(rows)
    path = cfg.reports_dir / "tables" / "w10_blend.csv"
    out.to_csv(path, index=False)

    pivot = out.pivot(index="model", columns="subset", values="rmse").round(2)
    print(pivot.to_string())
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
