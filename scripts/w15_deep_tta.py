"""W15: deep model with saved checkpoints and test-time augmentation (TTA).

W11 trained the conv-transformer but never saved it, so the checkpoint could not
be reused and the model could not be TTA'd.  This script:

1. trains the same ConvTransformer under the same protocol (fit 80% / 15% inner
   validation / official test engines), one seed by default;
2. **saves** every checkpoint as ``artifacts/deep_{subset}.pt`` plus a small
   ``deep_{subset}.json`` (sensors, window, condition levels);
3. predicts the test last cycles with TTA: the last window shifted by
   -2..+2 cycles (edge rows repeated), averaged;
4. reports RMSE/NASA and writes ``reports/tables/w15_deep_tta.csv``.

Artifacts make the deep model usable from ``src.pipeline`` later; TTA is the
cheap accuracy knob that does not touch the protocol.
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.metrics import nasa_score, rmse  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.models.deep import (  # noqa: E402
    SENSORS,
    ConvTransformer,
    last_windows,
    make_windows,
)

SHIFTS = (-2, -1, 0, 1, 2)


def tta_predict(model: ConvTransformer, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """Average predictions over shifted copies of each window."""
    preds = []
    c = torch.from_numpy(cond.astype(np.int64))
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
        with torch.no_grad():
            preds.append(model(torch.from_numpy(xs), c).numpy())
    return np.mean(preds, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W15: deep model + TTA, checkpoints saved")
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def train_one(tr, va, n_cond: int, args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvTransformer(len(SENSORS), n_cond, d_model=64, layers=2, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr / 100
    )
    loss_fn = nn.HuberLoss(delta=10.0)
    xv = torch.from_numpy(va[0])
    yv = np.asarray(va[1], dtype=np.float32)
    cv = torch.from_numpy(np.asarray(va[4], dtype=np.int64))

    best_rmse, best_state = float("inf"), None
    n = len(tr[0])
    order = np.arange(n)
    for _epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        for start in range(0, n, args.batch):
            b = order[start : start + args.batch]
            opt.zero_grad()
            loss = loss_fn(
                model(torch.from_numpy(tr[0][b]), torch.from_numpy(np.asarray(tr[4][b], dtype=np.int64))),
                torch.from_numpy(np.asarray(tr[1][b], dtype=np.float32)),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            score = rmse(yv, model(xv, cv).numpy())
        if score < best_rmse:
            best_rmse, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_rmse


def main() -> None:
    args = parse_args()
    torch.set_num_threads(8)
    cfg = Config()
    rows = []
    for name in args.subsets:
        t0 = time.time()
        ds = load_subset(name, cfg.data_dir)
        fit_units, _ = split_units(ds.train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed)
        fit = ds.train[ds.train["unit"].isin(fit_units)]
        tr_units, va_units = split_units(fit["unit"].to_numpy(), 1.0 - args.val_frac, cfg.seed + 1)
        norm = ConditionNormalizer(n_clusters=N_CONDITIONS[name], seed=cfg.seed).fit(fit)
        tr_df = norm.transform(fit[fit["unit"].isin(tr_units)])
        va_df = norm.transform(fit[fit["unit"].isin(va_units)])
        test_df = norm.transform(ds.test)

        tr = make_windows(tr_df, SENSORS, args.window, cap=cfg.rul_clip)
        va = make_windows(va_df, SENSORS, args.window, cap=cfg.rul_clip)
        x_te, te_units, te_cond = last_windows(test_df, SENSORS, args.window)
        y_te = np.minimum(ds.rul.astype(float), cfg.rul_clip)

        preds = []
        for seed in args.seeds:
            model, val_rmse = train_one(tr, va, N_CONDITIONS[name], args, seed)
            model.eval()
            p = np.clip(tta_predict(model, x_te, te_cond), 0.0, cfg.rul_clip)
            preds.append(p)
            torch.save(model.state_dict(), Path(cfg.artifacts_dir) / f"deep_{name}.pt")
            (Path(cfg.artifacts_dir) / f"deep_{name}.json").write_text(
                json.dumps(
                    {
                        "subset": name,
                        "window": args.window,
                        "sensors": SENSORS,
                        "d_model": 64,
                        "layers": 2,
                        "dropout": 0.1,
                        "val_rmse": val_rmse,
                        "seed": seed,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[{name}] seed={seed} val={val_rmse:.3f} test_tta={rmse(y_te, p):.3f}", flush=True)
        p_ens = np.mean(preds, axis=0)
        rows.append(
            {
                "subset": name,
                "model": "deep_tta",
                "rmse": rmse(y_te, p_ens),
                "nasa": nasa_score(y_te, p_ens),
                "seconds": round(time.time() - t0, 1),
            }
        )
        out = Path(cfg.artifacts_dir) / "preds" / f"w15_preds_{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unit": te_units, "pred": p_ens, "true": y_te}).to_csv(out, index=False)

    tab = pd.DataFrame(rows)
    dest = Path(cfg.reports_dir) / "tables" / "w15_deep_tta.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(dest, index=False)
    print(tab.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
