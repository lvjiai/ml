import numpy as np
import pandas as pd

from src.data.metropt import (
    ANALOG,
    CANONICAL,
    DIGITAL,
    FAILURE_WINDOWS,
    add_canonical,
    failure_windows,
    resample_10s,
)


def _toy(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2020-02-01", periods=n, freq="10s")
    df = pd.DataFrame({"timestamp": ts})
    for c in ANALOG:
        df[c] = rng.normal(5.0, 1.0, n).astype("float32")
    for c in DIGITAL:
        df[c] = 1
    return df


def test_canonical_channels_exist():
    out = add_canonical(_toy())
    for name in CANONICAL:
        assert f"canon_{name}" in out.columns
    assert np.allclose(out["canon_temperature"], out["Oil_temperature"])


def test_resample_puts_series_on_10s_grid():
    df = _toy(600)
    # knock two records out to simulate jitter
    df = df.drop(index=[10, 11]).reset_index(drop=True)
    out = resample_10s(df)
    diffs = out["timestamp"].diff().dropna().dt.total_seconds()
    assert set(np.unique(diffs)) == {10.0}
    assert len(out) == 600  # same time span, ffill keeps every 10 s bin


def test_failure_windows_match_pdf_table():
    w = failure_windows()
    assert len(w) == 4
    assert list(w["id"]) == [f[0] for f in FAILURE_WINDOWS]
    assert (w["end"] > w["start"]).all()
    # events do not overlap
    for a, b in zip(w["start"].iloc[1:], w["end"].iloc[:-1]):
        assert a > b
