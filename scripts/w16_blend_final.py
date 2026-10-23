"""W16: final comparison — LSTM window ensemble/TTA vs deep+TTA vs GBDT, blends.

All candidates are scored on the same test engines with the same censored target
(min(RUL, 125)) at the last observed cycle.  Differences are tested by a paired
bootstrap over engines (3000 resamples), because the subset-to-subset ranking
alone does not say whether a gap is real.

Inputs (all saved by earlier scripts):

* ``w2_preds_*.csv``        GBDT point model (W2)
* ``w14_preds_w{20,30,50}_*.csv``  LSTM + TTA per window (W14)
* ``w14_preds_*.csv``       LSTM window ensemble (W14)
* ``w15_preds_*.csv``       deep model + TTA (W15)

Outputs ``reports/tables/w16_final.csv`` and ``w16_bootstrap.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.eval.metrics import rmse  # noqa: E402

N_BOOT = 3000
SEED = 7


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"pred": "pred_" + path.stem.split("_")[-1]})
    return df


def main() -> None:
    cfg = Config()
    art = Path(cfg.artifacts_dir) / "preds"
    rng = np.random.default_rng(SEED)
    rows, boots = [], []
    all_cands: dict[str, dict[str, np.ndarray]] = {}
    all_truth: dict[str, np.ndarray] = {}
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        ds = load_subset(subset, cfg.data_dir)
        truth = np.minimum(ds.rul.astype(float), cfg.rul_clip)
        units = np.sort(ds.test["unit"].unique())

        cands: dict[str, np.ndarray] = {}
        g = pd.read_csv(art / f"w2_preds_{subset}.csv").set_index("unit")["pred_gbdt"]
        cands["gbdt"] = g.reindex(units).to_numpy()
        for window in (20, 30, 50):
            p = pd.read_csv(art / f"w14_preds_w{window}_{subset}.csv").set_index("unit")["pred"]
            cands[f"lstm_w{window}_tta"] = p.reindex(units).to_numpy()
        p = pd.read_csv(art / f"w14_preds_{subset}.csv").set_index("unit")["pred"]
        cands["lstm_ens_tta"] = p.reindex(units).to_numpy()
        p = pd.read_csv(art / f"w15_preds_{subset}.csv").set_index("unit")["pred"]
        cands["deep_tta"] = p.reindex(units).to_numpy()

        cands["lstm_w50+deep"] = (cands["lstm_w50_tta"] + cands["deep_tta"]) / 2.0
        cands["lstm_w50+gbdt"] = (cands["lstm_w50_tta"] + cands["gbdt"]) / 2.0
        cands["lstm_w50+deep+gbdt"] = (
            cands["lstm_w50_tta"] + cands["deep_tta"] + cands["gbdt"]
        ) / 3.0
        cands["lstm_w50+ens+deep"] = (
            cands["lstm_w50_tta"] + cands["lstm_ens_tta"] + cands["deep_tta"]
        ) / 3.0

        all_cands[subset] = {k: v.copy() for k, v in cands.items()}
        all_truth[subset] = truth.copy()
        for name, p in cands.items():
            rows.append({"subset": subset, "model": name, "rmse": rmse(truth, p)})

        base = rmse(truth, cands["lstm_w50_tta"])
        for name, p in cands.items():
            if name == "lstm_w50_tta":
                continue
            diffs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, len(units), len(units))
                r_a = rmse(truth[idx], p[idx])
                r_b = rmse(truth[idx], cands["lstm_w50_tta"][idx])
                diffs[b] = r_a - r_b
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            boots.append(
                {
                    "subset": subset,
                    "model": name,
                    "rmse": rmse(truth, p),
                    "delta_vs_w50": rmse(truth, p) - base,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

    # pooled bootstrap across the four subsets: resample engines within each
    # subset, then pool the squared errors (a single CI for the average gap).
    pooled = []
    for name in ["lstm_w50+deep", "lstm_w50+deep+gbdt", "lstm_w50+gbdt", "gbdt",
                 "lstm_w30_tta", "lstm_ens_tta", "deep_tta"]:
        diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            se_a, se_b, n = 0.0, 0.0, 0
            for subset in all_cands:
                truth = all_truth[subset]
                a = all_cands[subset][name]
                bb = all_cands[subset]["lstm_w50_tta"]
                idx = rng.integers(0, len(truth), len(truth))
                se_a += float(np.sum((truth[idx] - a[idx]) ** 2))
                se_b += float(np.sum((truth[idx] - bb[idx]) ** 2))
                n += len(idx)
            diffs[b] = np.sqrt(se_a / n) - np.sqrt(se_b / n)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pooled.append(
            {
                "model": name,
                "pooled_rmse": np.sqrt(
                    np.mean(
                        [
                            np.mean((all_truth[s] - all_cands[s][name]) ** 2)
                            for s in all_cands
                        ]
                    )
                ),
                "delta_vs_w50": np.sqrt(
                    np.mean(
                        [
                            np.mean((all_truth[s] - all_cands[s][name]) ** 2)
                            for s in all_cands
                        ]
                    )
                )
                - np.sqrt(
                    np.mean(
                        [
                            np.mean((all_truth[s] - all_cands[s]["lstm_w50_tta"]) ** 2)
                            for s in all_cands
                        ]
                    )
                ),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    pt = pd.DataFrame(pooled)
    pt.to_csv(Path(cfg.reports_dir) / "tables" / "w16_pooled.csv", index=False)
    print()
    print(pt.round(2).to_string(index=False))

    tab = pd.DataFrame(rows)
    out = Path(cfg.reports_dir) / "tables" / "w16_final.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    bt = pd.DataFrame(boots)
    bt.to_csv(Path(cfg.reports_dir) / "tables" / "w16_bootstrap.csv", index=False)
    print(tab.pivot(index="model", columns="subset", values="rmse").round(2).to_string())
    print()
    print(bt.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
