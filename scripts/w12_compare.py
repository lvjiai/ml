"""W12: one canonical comparison table with the *capped* test target.

Why this exists: the point-prediction tables in W2/W3/W9/W10 scored predictions
against the raw RUL label, which reaches 145 (FD001/FD003) and 195 (FD002/
FD004).  Every model here is trained on a target truncated at 125, so a unit
with 195 cycles left is penalised by 70 cycles no matter how good the model is.
The interval experiments in W4 already used min(RUL, 125); this script brings
the point tables to the same definition so that all numbers are comparable.

Usage: python scripts/w12_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.eval.metrics import nasa_score, rmse  # noqa: E402

CAP = 125.0


def load(path: Path, cols: dict[str, str], index: pd.Index) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("unit").sort_index()
    return df.loc[index, list(cols)].rename(columns=cols)


def main() -> None:
    cfg = Config()
    rows = []
    for name in cfg.subsets:
        preds = Path(cfg.artifacts_dir) / "preds"
        w2 = pd.read_csv(preds / f"w2_preds_{name}.csv").set_index("unit").sort_index()
        y = np.minimum(w2["y_true"].to_numpy(), CAP)
        idx = w2.index

        gbdt = w2.loc[idx, "pred_gbdt"].to_numpy()
        blend = w2.loc[idx, "pred_blend"].to_numpy()
        lstm = (
            pd.read_csv(preds / f"w3_preds_{name}.csv")
            .set_index("unit")
            .sort_index()
            .loc[idx, "pred_lstm"]
            .to_numpy()
        )
        deep = (
            pd.read_csv(preds / f"w11_preds_{name}.csv")
            .set_index("unit")
            .sort_index()
            .loc[idx, "pred"]
            .to_numpy()
        )
        candidates = {
            "gbdt": gbdt,
            "gbdt+ridge": blend,
            "lstm": lstm,
            "deep": deep,
            "gbdt+lstm": 0.5 * gbdt + 0.5 * lstm,
            "deep+gbdt": 0.5 * deep + 0.5 * gbdt,
            "deep+lstm": 0.5 * deep + 0.5 * lstm,
            "deep+gbdt+lstm": (deep + gbdt + lstm) / 3.0,
        }
        for model, pred in candidates.items():
            rows.append(
                {
                    "subset": name,
                    "model": model,
                    "rmse": round(rmse(y, pred), 3),
                    "nasa_score": round(nasa_score(y, pred), 1),
                    "n_units": len(y),
                }
            )

    table = pd.DataFrame(rows)
    dest = Path(cfg.reports_dir) / "tables" / "w12_final.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(table.pivot(index="model", columns="subset", values="rmse").to_string())
    print("\nmean RMSE across subsets:")
    print(
        table.groupby("model")["rmse"].mean().sort_values().round(3).to_string()
    )
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
