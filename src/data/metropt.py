"""Loader for the real-world MetroPT-3 compressor dataset (W5).

MetroPT-3 is a single 213-day time series (2020-02-01 → 2020-09-01) logged at
~0.084 Hz (10 s, after de-duplicating a 1 Hz recording). It ships without any
label file: the four documented air-leak events live in the PDF that is
distributed with the zip, so they are transcribed here as constants.

The C-MAPSS pipeline works on a 3-channel "canonical" view (pressure /
temperature / load) because the two datasets share physical meanings, not
sensor names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_METROPT_CSV

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

# Physical groups used for cross-domain comparison with C-MAPSS.
CANONICAL = {
    "pressure": ["TP2", "TP3", "H1", "Reservoirs"],
    "temperature": ["Oil_temperature"],
    "load": ["Motor_current"],
}

# (id, start, end) transcribed from "Data Description_Metro.pdf".
FAILURE_WINDOWS = [
    ("#1", "2020-04-18 00:00", "2020-04-18 23:59"),
    ("#2", "2020-05-29 23:30", "2020-05-30 06:00"),
    ("#3", "2020-06-05 10:00", "2020-06-07 14:30"),
    ("#4", "2020-07-15 14:30", "2020-07-15 19:00"),
]


def load_metropt(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV, sorted by time, with compact dtypes."""
    path = Path(path) if path is not None else DEFAULT_METROPT_CSV
    dtypes = {c: "float32" for c in ANALOG}
    dtypes.update({c: "int8" for c in DIGITAL})
    df = pd.read_csv(path, dtype=dtypes, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_10s(df: pd.DataFrame) -> pd.DataFrame:
    """Put the series on a strict 10 s grid (mean within each bin).

    The logging is not exactly regular (10 s 88%, 9–12 s, rare long outages),
    so every downstream rolling window would otherwise mix different spans.
    """
    out = (
        df.set_index("timestamp")
        .resample("10s")
        .mean(numeric_only=True)
        .ffill()
        .reset_index()
    )
    return out


def add_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three cross-domain channels (mean of each physical group)."""
    out = df.copy()
    for name, cols in CANONICAL.items():
        out[f"canon_{name}"] = out[cols].mean(axis=1)
    return out


def failure_windows() -> pd.DataFrame:
    rows = [{"id": i, "start": pd.Timestamp(s), "end": pd.Timestamp(e)} for i, s, e in FAILURE_WINDOWS]
    return pd.DataFrame(rows)
