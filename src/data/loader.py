"""Parse and validate the C-MAPSS text files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.schema import (
    COLUMNS,
    STATS,
    Dataset,
    DataValidationError,
)

_TRAIN = "train_{name}.txt"
_TEST = "test_{name}.txt"
_RUL = "RUL_{name}.txt"


def read_table(path: Path) -> pd.DataFrame:
    """Read one whitespace-separated C-MAPSS table into a typed frame."""
    if not path.is_file():
        raise FileNotFoundError(f"missing data file: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        dtype=str,
    )
    if df.shape[1] != len(COLUMNS):
        raise DataValidationError(
            f"{path.name}: expected {len(COLUMNS)} columns, got {df.shape[1]}"
        )
    for col in ("unit", "cycle"):
        df[col] = df[col].astype("int32")
    for col in COLUMNS[2:]:
        df[col] = df[col].astype("float32")
    return df


def read_rul(path: Path) -> np.ndarray:
    """Read an RUL label file (one integer per test unit)."""
    if not path.is_file():
        raise FileNotFoundError(f"missing RUL file: {path}")
    return np.loadtxt(path, dtype=np.int32)


def load_subset(name: str, data_dir: Path) -> Dataset:
    train = read_table(data_dir / _TRAIN.format(name=name))
    test = read_table(data_dir / _TEST.format(name=name))
    rul = read_rul(data_dir / _RUL.format(name=name))
    ds = Dataset(name=name, train=train, test=test, rul=rul)
    validate(ds)
    return ds


def load_all(config: Config) -> dict[str, Dataset]:
    return {name: load_subset(name, config.data_dir) for name in config.subsets}


def _check_cycles(df: pd.DataFrame, what: str) -> None:
    """Each unit must start at cycle 1 and increase by exactly one."""
    first = df.groupby("unit")["cycle"].min()
    if not (first == 1).all():
        raise DataValidationError(f"{what}: some units do not start at cycle 1")
    diffs = df.groupby("unit")["cycle"].diff().dropna()
    if not (diffs == 1).all():
        raise DataValidationError(f"{what}: cycle numbers are not contiguous")


def validate(ds: Dataset) -> None:
    """Assert that a subset matches the documented C-MAPSS shape."""
    stats = STATS[ds.name]

    for part, df, rows, units, what in (
        ("train", ds.train, stats.train_rows, stats.train_units, "training"),
        ("test", ds.test, stats.test_rows, stats.test_units, "test"),
    ):
        if df.shape[1] != len(COLUMNS):
            raise DataValidationError(f"{ds.name} {part}: column count mismatch")
        if len(df) != rows:
            raise DataValidationError(
                f"{ds.name} {part}: expected {rows} rows, got {len(df)}"
            )
        n_units = int(df["unit"].nunique())
        if n_units != units:
            raise DataValidationError(
                f"{ds.name} {part}: expected {units} units, got {n_units}"
            )
        if df.isna().to_numpy().any():
            raise DataValidationError(f"{ds.name} {part}: contains NaN")
        if int(df["unit"].min()) != 1:
            raise DataValidationError(f"{ds.name} {part}: unit ids must start at 1")
        _check_cycles(df, f"{ds.name} {part}")

        # Unit ids must be contiguous 1..n so that RUL labels can be aligned.
        expected_ids = np.arange(1, n_units + 1)
        if not np.array_equal(np.sort(df["unit"].unique()), expected_ids):
            raise DataValidationError(f"{ds.name} {part}: unit ids are not contiguous")

    if ds.rul.shape[0] != stats.rul_lines:
        raise DataValidationError(
            f"{ds.name}: expected {stats.rul_lines} RUL labels, got {ds.rul.shape[0]}"
        )
    if ds.rul.shape[0] != stats.test_units:
        raise DataValidationError(
            f"{ds.name}: RUL labels ({ds.rul.shape[0]}) != test units ({stats.test_units})"
        )
    if (ds.rul < 0).any():
        raise DataValidationError(f"{ds.name}: negative RUL label")
