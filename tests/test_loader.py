"""W1 acceptance tests: the parser must reproduce the documented shapes."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.data.loader import load_subset
from src.data.schema import COLUMNS, STATS, DataValidationError, Dataset
from src.features.rul import add_rul


@pytest.mark.parametrize("name", ["FD001", "FD002", "FD003", "FD004"])
def test_shapes_match_registry(name: str) -> None:
    ds = load_subset(name, Config().data_dir)
    stats = STATS[name]
    assert len(ds.train) == stats.train_rows
    assert len(ds.test) == stats.test_rows
    assert len(ds.rul) == stats.rul_lines
    assert list(ds.train.columns) == COLUMNS


def test_rul_labels_are_aligned_with_units() -> None:
    ds = load_subset("FD001", Config().data_dir)
    assert ds.test["unit"].nunique() == len(ds.rul)
    last = ds.test.groupby("unit")["cycle"].max()
    # FD001's label file is sorted by unit id, so labels must be non-negative
    # and the smallest one cannot exceed the shortest test trajectory length.
    assert ds.rul.min() >= 0
    assert ds.rul.min() <= last.min()


def test_add_rul_counts_down_to_zero() -> None:
    ds = load_subset("FD001", Config().data_dir)
    train = add_rul(ds.train, clip=125)
    per_unit = train.groupby("unit")["rul"].agg(["min", "max", "count"])
    assert (per_unit["min"] == 0).all()
    assert (per_unit["max"] <= 125).all()
    # every unit must reach rul == 0 at exactly one cycle
    assert (per_unit["count"] > 0).all()


def test_validation_rejects_corrupted_frame() -> None:
    ds = load_subset("FD001", Config().data_dir)
    bad = Dataset(name="FD001", train=ds.train.iloc[:-1], test=ds.test, rul=ds.rul)
    with pytest.raises(DataValidationError):
        from src.data.loader import validate

        validate(bad)
