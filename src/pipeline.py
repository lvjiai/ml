"""The one inference entry point: raw sensor rows -> RUL point + interval.

Every experiment script builds features the same way, so this module owns that
recipe.  The Streamlit demo and the end-to-end test both go through
:func:`predict`, which means the demo cannot silently drift from the W2-W4
pipeline.

A *bundle* is everything needed to answer a query without touching the training
data again: the point GBDT, the condition normaliser, the feature list, and the
CQR quantile models plus their conformal offset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import Config
from src.data.schema import SENSORS
from src.features.conditions import ConditionNormalizer
from src.features.rolling import add_rolling_features
from src.models.base import last_cycle_rows


@dataclass(frozen=True)
class RULBundle:
    """Frozen model package for one C-MAPSS subset."""

    subset: str
    booster: lgb.Booster
    norm: ConditionNormalizer
    features: tuple[str, ...]
    windows: tuple[int, ...]
    rul_clip: int
    alpha: float = 0.1
    cqr_q: float | None = None
    q_lo: lgb.Booster | None = None
    q_hi: lgb.Booster | None = None

    @property
    def has_interval(self) -> bool:
        return self.cqr_q is not None and self.q_lo is not None and self.q_hi is not None


def load_bundle(
    subset: str, artifacts_dir: Path | None = None, cfg: Config | None = None
) -> RULBundle:
    """Load the saved artifacts of one subset (no training, no test data)."""
    cfg = cfg or Config()
    art = Path(artifacts_dir or cfg.artifacts_dir)
    meta = json.loads((art / f"features_{subset}.json").read_text(encoding="utf-8"))
    # LightGBM's C++ layer cannot read non-ASCII paths; go through Python.
    booster = lgb.Booster(model_str=(art / f"gbdt_{subset}.txt").read_text(encoding="utf-8"))
    norm = joblib.load(art / f"norm_{subset}.joblib")

    conf_path = art / f"conformal_{subset}.json"
    if conf_path.exists():
        conf = json.loads(conf_path.read_text(encoding="utf-8"))
        q_lo = lgb.Booster(
            model_str=(art / f"cqr_lo_{subset}.txt").read_text(encoding="utf-8")
        )
        q_hi = lgb.Booster(
            model_str=(art / f"cqr_hi_{subset}.txt").read_text(encoding="utf-8")
        )
        return RULBundle(
            subset=subset,
            booster=booster,
            norm=norm,
            features=tuple(meta["features"]),
            windows=tuple(meta["windows"]),
            rul_clip=int(meta.get("rul_clip", cfg.rul_clip)),
            alpha=float(conf["alpha"]),
            cqr_q=float(conf["cqr_q"]),
            q_lo=q_lo,
            q_hi=q_hi,
        )

    return RULBundle(
        subset=subset,
        booster=booster,
        norm=norm,
        features=tuple(meta["features"]),
        windows=tuple(meta["windows"]),
        rul_clip=int(meta.get("rul_clip", cfg.rul_clip)),
    )


def save_conformal(
    subset: str,
    q_lo: lgb.Booster,
    q_hi: lgb.Booster,
    cqr_q: float,
    split_q: float,
    artifacts_dir: Path,
    alpha: float,
) -> tuple[Path, Path, Path]:
    """Persist the CQR models and their offsets next to the point model."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    lo_path = artifacts_dir / f"cqr_lo_{subset}.txt"
    hi_path = artifacts_dir / f"cqr_hi_{subset}.txt"
    for booster, path in ((q_lo, lo_path), (q_hi, hi_path)):
        path.write_text(booster.model_to_string(), encoding="utf-8")
    meta_path = artifacts_dir / f"conformal_{subset}.json"
    meta_path.write_text(
        json.dumps(
            {"alpha": alpha, "cqr_q": float(cqr_q), "split_q": float(split_q)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return lo_path, hi_path, meta_path


def build_features(df: pd.DataFrame, bundle: RULBundle) -> pd.DataFrame:
    """Raw sensor table -> the exact feature frame the models were fitted on."""
    df = df.copy()
    if "unit" not in df.columns:
        df["unit"] = 0
    if "cycle" not in df.columns:
        df["cycle"] = np.arange(1, len(df) + 1)
    feat = add_rolling_features(bundle.norm.transform(df), SENSORS, bundle.windows)
    missing = [c for c in bundle.features if c not in feat.columns]
    if missing:
        raise KeyError(f"feature frame is missing {len(missing)} columns, e.g. {missing[:3]}")
    return feat


def predict_last(bundle: RULBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Predict at the last cycle of every unit in ``df``.

    ``df`` needs the sensor columns and the three settings; it may be a
    truncated trajectory (the demo's case) or a full one.  Returns one row per
    unit with the point estimate and, when available, the CQR interval.
    """
    feat = build_features(df, bundle)
    last = last_cycle_rows(feat)
    x = last[list(bundle.features)].to_numpy()
    point = np.clip(bundle.booster.predict(x), 0.0, float(bundle.rul_clip))

    out = pd.DataFrame(
        {
            "unit": last["unit"].to_numpy(),
            "cycle": last["cycle"].to_numpy(),
            "rul_point": point,
        }
    )
    if bundle.has_interval:
        lo = bundle.q_lo.predict(x) - bundle.cqr_q
        hi = bundle.q_hi.predict(x) + bundle.cqr_q
        out["rul_lo"] = np.clip(lo, 0.0, float(bundle.rul_clip))
        out["rul_hi"] = np.clip(hi, 0.0, float(bundle.rul_clip))
        # Quantile models can cross on noisy inputs; enforce an ordered interval.
        out["rul_lo"] = np.minimum(out["rul_lo"], out["rul_point"])
        out["rul_hi"] = np.maximum(out["rul_hi"], out["rul_point"])
    return out


def predict_one(bundle: RULBundle, df: pd.DataFrame) -> dict[str, float]:
    """Convenience wrapper for a single trajectory (what the demo calls)."""
    row = predict_last(bundle, df).iloc[0]
    result = {
        "unit": float(row["unit"]),
        "cycle": float(row["cycle"]),
        "rul_point": float(row["rul_point"]),
    }
    if "rul_lo" in row:
        result["rul_lo"] = float(row["rul_lo"])
        result["rul_hi"] = float(row["rul_hi"])
    return result

