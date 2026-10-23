"""W11: strong deep model (dilated conv + transformer + attention pooling).

Protocol (unchanged from W2 so the numbers are comparable):
- fit on 80% of the training engines, test on the official test engines;
- inside fit, 15% of engines are held out for early stopping;
- the condition normaliser is fit on fit engines only (no test leakage);
- test RUL is the truncated target at the last observed cycle.

The only thing that changes is the model class, so any gain is attributable to
the architecture and the condition embedding, not to a looser protocol.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset
from src.data.split import split_units
from src.eval.metrics import nasa_score, rmse
from src.features.conditions import N_CONDITIONS, ConditionNormalizer
from src.models.deep import (
    SENSORS,
    ConvTransformer,
    last_windows,
    make_windows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def train_one(
    tr: tuple[np.ndarray, np.ndarray, np.ndarray],
    va: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_cond: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[ConvTransformer, float, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvTransformer(
        len(SENSORS), n_cond, d_model=args.d_model, layers=args.layers,
        dropout=args.dropout,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr / 100
    )
    loss_fn = nn.HuberLoss(delta=10.0)

    xv = torch.from_numpy(va[0])
    yv = np.asarray(va[1], dtype=np.float32)
    cv = torch.from_numpy(np.asarray(va[4], dtype=np.int64))

    best_rmse, best_state, best_epoch = float("inf"), None, 0
    n = len(tr[0])
    order = np.arange(n)
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        for start in range(0, n, args.batch):
            b = order[start : start + args.batch]
            xb = torch.from_numpy(tr[0][b])
            yb = torch.from_numpy(np.asarray(tr[1][b], dtype=np.float32))
            cb = torch.from_numpy(np.asarray(tr[4][b], dtype=np.int64))
            opt.zero_grad()
            loss = loss_fn(model(xb, cb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv, cv).numpy()
        score = rmse(yv, pred)
        if score < best_rmse:
            best_rmse = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_rmse, best_epoch


def main() -> None:
    args = parse_args()
    torch.set_num_threads(16)
    cfg = Config()
    rows = []
    for name in args.subsets:
        t0 = time.time()
        ds = load_subset(name, cfg.data_dir)
        train = ds.train
        fit_units, _ = split_units(
            train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed
        )
        fit = train[train["unit"].isin(fit_units)]
        tr_units, val_units = split_units(
            fit["unit"].to_numpy(), 1.0 - args.val_frac, cfg.seed + 1
        )
        tr_df = fit[fit["unit"].isin(tr_units)]
        va_df = fit[fit["unit"].isin(val_units)]

        norm = ConditionNormalizer(n_clusters=N_CONDITIONS[name]).fit(fit)
        tr_df, va_df = norm.transform(tr_df), norm.transform(va_df)
        test_df = norm.transform(ds.test)

        tr = make_windows(tr_df, SENSORS, args.window, cap=cfg.rul_clip)
        va = make_windows(va_df, SENSORS, args.window, cap=cfg.rul_clip)
        xte, te_units, te_cond = last_windows(test_df, SENSORS, args.window)
        yte = np.minimum(ds.rul.astype(np.float64), cfg.rul_clip)

        preds = []
        for seed in args.seeds:
            model, val_rmse, epoch = train_one(tr, va, N_CONDITIONS[name], args, seed)
            model.eval()
            with torch.no_grad():
                pred = model(
                    torch.from_numpy(xte),
                    torch.from_numpy(te_cond.astype(np.int64)),
                ).numpy()
            preds.append(np.clip(pred, 0, cfg.rul_clip))
            print(
                f"{name} seed={seed} val_rmse={val_rmse:.3f} epoch={epoch} "
                f"test_rmse={rmse(yte[: len(pred)], pred[: len(yte)]):.3f}"
            )
        p = np.mean(preds, axis=0)
        rows.append(
            {
                "subset": name,
                "model": "deep_conv_transformer",
                "rmse": rmse(yte, p),
                "nasa_score": nasa_score(yte, p),
                "n_test_units": len(yte),
                "seconds": round(time.time() - t0, 1),
            }
        )
        out = Path(cfg.artifacts_dir) / "preds" / f"w11_preds_{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unit": te_units[: len(yte)], "pred": p, "true": yte}).to_csv(
            out, index=False
        )
        print(rows[-1])

    table = pd.DataFrame(rows)
    dest = Path(cfg.reports_dir) / "tables" / "w11_deep.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        old = pd.read_csv(dest)
        table = pd.concat([old, table], ignore_index=True).drop_duplicates(
            ["subset", "model"], keep="last"
        )
    table.to_csv(dest, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

