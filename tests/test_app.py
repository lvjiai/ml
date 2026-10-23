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
    assert "剩余周期数" in labels
    assert "90% 范围" in labels


def test_app_point_prediction_is_sane(app) -> None:
    values = {m.label: m.value for m in app.metric}
    point = float(values["剩余周期数"])
    assert 0.0 <= point <= 125.0

    lo, hi = values["90% 范围"].split(" – ")
    lo, hi = float(lo), float(hi)
    assert lo <= point <= hi, "point estimate must lie inside its own interval"
    assert 0.0 <= lo and hi <= 125.0


def test_hidden_truth_keeps_full_life_trajectories(app) -> None:
    """Hiding the truth must not switch to the truncated test split.

    The test split of C-MAPSS stops at an arbitrary cycle (FD001 unit 1 ends at
    31), so the prediction curve used to collapse to a single point.
    """
    app.radio[0].set_value("不显示").run()
    assert [str(e.value) for e in app.exception] == []
    values = {m.label: m.value for m in app.metric}
    assert values["真实值"] == "不显示"

    sliders = {s.label: s for s in app.slider}
    slider = sliders["观测到的周期数"]
    # Full life of FD001 unit 1 is 192; the truncated test copy ends at 31.
    assert slider.max >= 150, f"slider max {slider.max} looks like a truncated test unit"

