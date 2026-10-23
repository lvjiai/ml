"""W5: how far conformal RUL intervals survive a domain shift (simulation only).

No cross-device transfer: every experiment stays inside C-MAPSS, so each step
of the ladder adds exactly one factor and the degradation can be attributed.

    E1  in-distribution   FD001 fit -> FD001 calib -> FD001 test
    E2  naive transfer    FD001 model + FD001 normaliser + FD001 calibration
                          -> FD002 / FD004 test (6 operating conditions)
    E3  adapted transfer  same model, but the target is normalised with a
                          normaliser fitted *unsupervised* on the target
                          training sensors (transductive; declared in the ADR)
    E4  few-shot          E3 + k labelled target units to recalibrate the
                          conformal quantile
    E5  perturbation      E3 features with additive noise / sensor dropout

Run from the project root:

    python scripts/w5_domain_shift.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.schema import SENSORS  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.metrics import DEFAULT_BINS  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.features.rolling import add_rolling_features  # noqa: E402
from src.intervals.calibration import sample_truncations  # noqa: E402
from src.intervals.conformal import SplitConformal, conformal_quantile, interval_metrics  # noqa: E402
from src.models.base import last_cycle_rows  # noqa: E402
from src.models.gbdt import select_model, train_gbdt  # noqa: E402

PAIRS = (("FD001", "FD002"), ("FD001", "FD004"))
K_GRID = (1, 2, 3, 5, 10, 20, 50)
NOISE_LEVELS = (0.1, 0.3, 0.5)
WINDOW = 30
CAP = 125.0
CALIB_HORIZON = 150.0
ALPHA = 0.1
EXCLUDE = {"unit", "cycle", "setting_1", "setting_2", "setting_3", "rul", "rul_true", "condition"}


def with_targets(df: pd.DataFrame, clip: int) -> pd.DataFrame:
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul_true"] = (last - out["cycle"]).astype(float)
    out["rul"] = out["rul_true"].clip(upper=clip).astype("int32")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE]


def build(df: pd.DataFrame, norm: ConditionNormalizer, windows: tuple[int, ...]) -> pd.DataFrame:
    return add_rolling_features(norm.transform(df), SENSORS, windows)


def calib_frame(feat: pd.DataFrame, units: np.ndarray, window: int) -> pd.DataFrame:
    pts = sample_truncations(feat, units, window, stride=10, horizon=CALIB_HORIZON)
    pts = pts.rename(columns={"rul": "y_raw"})
    merged = pts.merge(feat, on=["unit", "cycle"], how="left", validate="one_to_one")
    assert merged.notna().all().all(), "calibration merge lost rows"
    merged["y"] = np.minimum(merged["rul_true"], CAP)
    return merged


def evaluate(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    return interval_metrics(y, lo, hi, DEFAULT_BINS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W5: domain shift of conformal RUL intervals")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = Config()
    if args.data_dir:
        cfg = cfg.replace(data_dir=args.data_dir)

    rows: list[dict] = []
    t0 = time.time()

    for src_name, tgt_name in PAIRS:
        src = load_subset(src_name, cfg.data_dir)
        tgt = load_subset(tgt_name, cfg.data_dir)
        src_train = with_targets(src.train, cfg.rul_clip)
        tgt_train = with_targets(tgt.train, cfg.rul_clip)

        src_norm = ConditionNormalizer(N_CONDITIONS[src_name]).fit(src_train)
        src_feat = build(src_train, src_norm, cfg.windows)
        feats = feature_columns(src_feat)

        fit_units, calib_units = split_units(
            src_train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed
        )
        fit = src_feat[src_feat["unit"].isin(fit_units)]
        sel = select_model(fit, feats, seed=cfg.seed, max_rounds=800)
        model = train_gbdt(fit, feats, params=sel["params"], seed=cfg.seed,
                           num_boost_round=sel["rounds"])

        src_calib = calib_frame(src_feat, calib_units, WINDOW)
        p_c = np.clip(model.predict(src_calib), 0.0, CAP)
        q_src = conformal_quantile(np.abs(src_calib["y"].to_numpy() - p_c), ALPHA)

        # E1: source test, source normaliser, source calibration
        src_test_feat = add_rolling_features(src_norm.transform(src.test), SENSORS, cfg.windows)
        src_last = last_cycle_rows(src_test_feat)
        y_src = np.minimum(np.asarray(src.rul, dtype=float), CAP)
        p_src = np.clip(model.predict(src_last), 0.0, CAP)
        rows.append({"pair": f"{src_name}->{src_name}", "stage": "E1 in-distribution", "k": 0,
                     "q": q_src, "point_rmse": float(np.sqrt(np.mean((y_src - p_src) ** 2))),
                     **evaluate(y_src, p_src - q_src, p_src + q_src)})

        # Target side -------------------------------------------------------
        tgt_test = with_targets(tgt.test, cfg.rul_clip)
        y_tgt = np.minimum(np.asarray(tgt.rul, dtype=float), CAP)

        tgt_norm_adapted = ConditionNormalizer(N_CONDITIONS[tgt_name]).fit(tgt_train)
        variants = {
            "E2 naive transfer": (src_norm, tgt_test),
            "E3 adapted transfer": (tgt_norm_adapted, tgt_test),
        }
        tgt_feat_adapted = None
        for stage, (norm, raw) in variants.items():
            feat = add_rolling_features(norm.transform(raw), SENSORS, cfg.windows)
            if stage.startswith("E3"):
                tgt_feat_adapted = feat
            last = last_cycle_rows(feat)
            p = np.clip(model.predict(last), 0.0, CAP)
            rows.append({"pair": f"{src_name}->{tgt_name}", "stage": stage, "k": 0,
                         "q": q_src, "point_rmse": float(np.sqrt(np.mean((y_tgt - p) ** 2))),
                         **evaluate(y_tgt, p - q_src, p + q_src)})

        # E4: few-shot recalibration on k target units ----------------------
        tgt_train_adapted = build(tgt_train, tgt_norm_adapted, cfg.windows)
        tgt_units = np.sort(tgt_train["unit"].unique())
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(tgt_units)
        p_adapted = np.clip(model.predict(last_cycle_rows(tgt_feat_adapted)), 0.0, CAP)
        for k in K_GRID:
            if k > len(tgt_units):
                continue
            units_k = order[:k]
            cal = calib_frame(tgt_train_adapted, units_k, WINDOW)
            p_k = np.clip(model.predict(cal), 0.0, CAP)
            q_k = conformal_quantile(np.abs(cal["y"].to_numpy() - p_k), ALPHA)
            rows.append({"pair": f"{src_name}->{tgt_name}", "stage": "E4 few-shot", "k": k,
                         "q": q_k, "point_rmse": float(np.sqrt(np.mean((y_tgt - p_adapted) ** 2))),
                         **evaluate(y_tgt, p_adapted - q_k, p_adapted + q_k)})

        # E5: feature-space perturbations at the adapted operating point ----
        X = tgt_feat_adapted[feats].to_numpy()
        for sigma in NOISE_LEVELS:
            rng2 = np.random.default_rng(cfg.seed + int(sigma * 100))
            noisy = tgt_feat_adapted.copy()
            noisy[feats] = X + rng2.normal(0.0, sigma, size=X.shape).astype("float32")
            p = np.clip(model.predict(last_cycle_rows(noisy)), 0.0, CAP)
            rows.append({"pair": f"{src_name}->{tgt_name}", "stage": f"E5 noise {sigma}", "k": 0,
                         "q": q_src, "point_rmse": float(np.sqrt(np.mean((y_tgt - p) ** 2))),
                         **evaluate(y_tgt, p - q_src, p + q_src)})

        p_drop = None
        noisy = tgt_feat_adapted.copy()
        drop_cols = [c for c in feats if any(s in c for s in ("s_4", "s_11", "s_12", "s_7"))]
        noisy[drop_cols] = 0.0
        p_drop = np.clip(model.predict(last_cycle_rows(noisy)), 0.0, CAP)
        rows.append({"pair": f"{src_name}->{tgt_name}", "stage": "E5 sensor dropout", "k": 0,
                     "q": q_src, "point_rmse": float(np.sqrt(np.mean((y_tgt - p_drop) ** 2))),
                     **evaluate(y_tgt, p_drop - q_src, p_drop + q_src)})

        print(f"[{src_name}->{tgt_name}] {time.time() - t0:.1f}s  rounds={sel['rounds']} "
              f"features={len(feats)} calib_n={len(src_calib)}", flush=True)

    tab = pd.DataFrame(rows)
    tab_dir = cfg.reports_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tab_dir / "w5_domain_shift.csv", index=False)
    plot_fewshot(tab, cfg.reports_dir / "figures")
    print("\n=== W5 coverage / width @ alpha=0.10 ===")
    print(tab[["pair", "stage", "k", "coverage", "mean_width", "point_rmse"]].round(3).to_string(index=False))
    return 0


def plot_fewshot(tab: pd.DataFrame, out_dir: Path) -> Path:
    """Coverage / width as a function of the number of labelled target units."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(tab["pair"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for pair in pairs:
        sub = tab[(tab["pair"] == pair) & (tab["stage"] == "E4 few-shot")].sort_values("k")
        axes[0].plot(sub["k"], sub["coverage"], marker="o", label=f"{pair} few-shot")
        axes[1].plot(sub["k"], sub["mean_width"], marker="o", label=f"{pair} few-shot")
        base = tab[(tab["pair"] == pair) & (tab["stage"] == "E3 adapted transfer")]
        if len(base):
            axes[0].axhline(base["coverage"].iloc[0], ls=":", lw=1, color="grey")
            axes[1].axhline(base["mean_width"].iloc[0], ls=":", lw=1, color="grey")

    axes[0].axhline(0.9, color="black", lw=1, ls="--", label="nominal 90%")
    axes[0].set_ylabel("coverage")
    axes[1].set_ylabel("mean interval width")
    for ax, title in zip(axes, ("coverage vs k", "width vs k")):
        ax.set_xlabel("labelled target units k")
        ax.set_xscale("log", base=2)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("W5: few-shot recalibration on the target domain (dotted = no recalibration)")
    fig.tight_layout()
    path = out_dir / "w5_fewshot.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())