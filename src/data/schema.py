"""Column names, dataset container and the ground-truth shapes of C-MAPSS.

The constants in ``STATS`` were measured on the files distributed with the
NASA Turbofan Engine Degradation Simulation Data Set (outer zip 12,429,152 B,
inner ``CMAPSSData.zip``).  They are used by ``validate()`` as regression
tests: a silent parser change or a corrupted download must fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS: list[str] = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)
SETTINGS: list[str] = COLUMNS[2:5]
SENSORS: list[str] = COLUMNS[5:]
# 26 columns: 2 keys + 3 operational settings + 21 sensors.
N_COLUMNS: int = len(COLUMNS)


@dataclass(frozen=True)
class SubsetStats:
    train_rows: int
    test_rows: int
    train_units: int
    test_units: int
    rul_lines: int


STATS: dict[str, SubsetStats] = {
    "FD001": SubsetStats(20631, 13096, 100, 100, 100),
    "FD002": SubsetStats(53759, 33991, 260, 259, 259),
    "FD003": SubsetStats(24720, 16596, 100, 100, 100),
    "FD004": SubsetStats(61249, 41214, 249, 248, 248),
}


@dataclass(frozen=True)
class Dataset:
    """One C-MAPSS subset: run-to-failure train, truncated test, RUL labels."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul: np.ndarray  # len == number of test units, aligned with unit order


class DataValidationError(RuntimeError):
    """Raised when a parsed subset does not match the documented shape."""
