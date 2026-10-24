"""W17: mid-life comparison on *all* engines (60-80% of life).

The demo made a real weakness visible: on some engines the prediction stays on
the "healthy plateau" (>=100 cycles) long after the true RUL has left it.  This
script measures that across every training engine, at five cutoffs per engine,
for the four models that exist: GBDT, LSTM(w30), LSTM(w50, retrained here and
saved) and the deep ConvTransformer with TTA.

No sampling: every engine of every subset is scored at 60/65/70/75/80% of its
life.  Reported per subset and model: mean bias (pred - truth), MAE, p90 of
|error|, and the number of "stuck on the plateau" cases (pred >= 100 while the
true RUL is <= 80).
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
from src.data.schema import SENSORS  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.features.rul import add_rul  # noqa: E402
from src.models.deep import ConvTransformer, last_windows  # noqa: E402
from src.models.lstm import RULNet, build_sequences, last_window, predict, train_lstm  # noqa: E402
from src.pipeline import load_bundle, predict_one  # noqa: E402

WINDOW50 = 50
VAL_FRAC = 0.15
FRACS = (0.60, 0.65, 0.70, 0.75, 0.80)
SHIFTS = (-2, -1, 0, 1, 2)


def shift_windows(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x
    xs = np.empty_like(x)
    if k > 0:
        xs[:, :-k] = x[:, k:]
        xs[:, -k:] = x[:, -1:]
    else:
        xs[:, -k:] = x[:, :k]
        xs[:, :-k] = x[:, :1]
    return xs


def tta_lstm(model: RULNet, x: np.ndarray) -> np.ndarray:
    return np.mean([predict(model, shift_windows(x, k)) for k in SHIFTS], axis=0)


def tta_deep(model: ConvTransformer, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for k in SHIFTS:
            xs = shift_windows(x, k)
            out.append(model(torch.from_numpy(xs), torch.from_numpy(cond)).numpy().ravel())
    return np.mean(out, axis=0)


def load_lstm(path: Path, window: int, hidden: int = 64) -> RULNet:
    model = RULNet(n_features=len(SENSORS), hidden=hidden)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_deep(subset: str, cfg: Config) -> tuple[ConvTransformer, int, list[str]]:
    meta = json.loads((Path(cfg.artifacts_dir) / f"deep_{subset}.json").read_text(encoding="utf-8"))
    model = ConvTransformer(
        n_sensors=len(meta["sensors"]),
        n_cond=N_CONDITIONS[subset],
        d_model=meta["d_model"],
        layers=meta["layers"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(
        torch.load(Path(cfg.artifacts_dir) / f"deep_{subset}.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, int(meta["window"]), list(meta["sensors"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W17: mid-life comparison, all engines")
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--epochs", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(8)
    cfg = Config()

    rows: list[dict] = []
    for name in args.subsets:
        ds = load_subset(name, cfg.data_dir)
        train = ds.train
        fit_units, _ = split_units(train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed)
        norm = ConditionNormalizer(n_clusters=N_CONDITIONS[name], seed=cfg.seed).fit(train)
        fit_df = norm.transform(add_rul(train[train["unit"].isin(fit_units)], cfg.rul_clip))
        tr_units, va_units = split_units(fit_df["unit"].to_numpy(), 1.0 - VAL_FRAC, cfg.seed + 1)

        # Train the w50 LSTM once per subset and persist it (W14 did not save).
        ckpt = Path(cfg.artifacts_dir) / f"lstm_w50_{name}.pt"
        model50 = None
        if ckpt.exists():
            model50 = load_lstm(ckpt, WINDOW50)
        else:
            Xtr, ytr, _ = build_sequences(fit_df[fit_df["unit"].isin(tr_units)], SENSORS, WINDOW50)
            Xva, yva, _ = build_sequences(fit_df[fit_df["unit"].isin(va_units)], SENSORS, WINDOW50)
            model50, _h, _e = train_lstm(Xtr, ytr, Xva, yva, epochs=args.epochs, seed=0)
            torch.save(model50.state_dict(), ckpt)
            print(f"[{name}] trained w50 LSTM", flush=True)
        model30 = load_lstm(Path(cfg.artifacts_dir) / f"lstm_{name}.pt", 30)
        deep, deep_window, deep_sensors = load_deep(name, cfg)
        # W15 fitted the normaliser on the 80% fit units only; feeding the deep
        # model windows normalised with different statistics puts it off its
        # training distribution, so keep a separate normaliser for it.
        deep_norm = ConditionNormalizer(n_clusters=N_CONDITIONS[name], seed=cfg.seed).fit(
            train[train["unit"].isin(fit_units)]
        )
        bundle = load_bundle(name)

        units = np.sort(train["unit"].unique())
        for frac in FRACS:
            for unit in units:
                g = train[train["unit"] == unit]
                life = int(g["cycle"].max())
                cut = int(round(life * frac))
                if cut < 30:
                    continue
                sub = g[g["cycle"] <= cut]
                truth = float(min(life - cut, cfg.rul_clip))
                raw = sub

                gbdt = float(predict_one(bundle, raw)["rul_point"])

                norm_sub = norm.transform(sub)
                x30, _ = last_window(norm_sub, SENSORS, 30)
                lstm30 = float(tta_lstm(model30, x30)[0])
                x50, _ = last_window(norm_sub, SENSORS, WINDOW50)
                if len(sub) < WINDOW50:
                    continue
                lstm50 = float(tta_lstm(model50, x50)[0])
                xd, _u, cd = last_windows(
                    deep_norm.transform(sub), deep_sensors, deep_window
                )
                deepp = float(tta_deep(deep, xd, cd)[0])

                for model_name, pred in (
                    ("gbdt", gbdt),
                    ("lstm_w30", lstm30),
                    ("lstm_w50", lstm50),
                    ("deep_tta", deepp),
                ):
                    rows.append(
                        {
                            "subset": name,
                            "unit": int(unit),
                            "cycle": int(cut),
                            "life": life,
                            "frac": frac,
                            "model": model_name,
                            "pred": pred,
                            "truth": truth,
                            "err": pred - truth,
                        }
                    )
        print(f"[{name}] done ({len(rows)} rows)", flush=True)

    tab = pd.DataFrame(rows)
    dest = Path(cfg.reports_dir) / "tables" / "w17_midlife.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(dest, index=False)

    out = []
    for (subset, model), g in tab.groupby(["subset", "model"]):
        err = g["err"].to_numpy()
        out.append(
            {
                "subset": subset,
                "model": model,
                "n": len(g),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "p90": float(np.percentile(np.abs(err), 90)),
                "stuck": int(((g["pred"] >= 100) & (g["truth"] <= 80)).sum()),
            }
        )
    summary = pd.DataFrame(out).sort_values(["subset", "mae"])
    summary.to_csv(Path(cfg.reports_dir) / "tables" / "w17_midlife_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
