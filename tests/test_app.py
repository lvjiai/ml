"""End-to-end smoke test: the Streamlit app renders and produces a prediction.

This is the only test that goes through the real artifacts, so it doubles as a
check that the shipped models still load (no retraining, no data leakage).
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app():
    at = streamlit.AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    return at


def test_app_has_no_exceptions(app) -> None:
    assert [str(e.value) for e in app.exception] == []


def test_app_shows_metrics(app) -> None:
    labels = [m.label for m in app.metric]
    assert "还能再跑多少个周期" in labels
    assert "合理范围（约 90%）" in labels


def test_app_point_prediction_is_sane(app) -> None:
    values = {m.label: m.value for m in app.metric}
    point = float(values["还能再跑多少个周期"])
    assert 0.0 <= point <= 125.0

    lo, hi = values["合理范围（约 90%）"].split(" – ")
    lo, hi = float(lo), float(hi)
    assert lo <= point <= hi, "point estimate must lie inside its own interval"
    assert 0.0 <= lo and hi <= 125.0
