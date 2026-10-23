"""Global configuration.

Every path and hyper-parameter used by the pipeline originates here so that a
single ``dataclasses.replace`` can reproduce an experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where the extracted C-MAPSS files live. Override with the RUL_DATA_DIR
# environment variable when moving the dataset.
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "RUL_DATA_DIR",
        r"E:\新建文件夹\ml-projects\rul-cmapss\cmapssdata",
    )
)

# MetroPT-3 real-world compressor series (W5). Override with METROPT_CSV.
DEFAULT_METROPT_CSV = Path(
    os.environ.get(
        "METROPT_CSV",
        r"E:\新建文件夹\rul-cmapss\metropt\MetroPT3.csv",
    )
)


@dataclass(frozen=True)
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    reports_dir: Path = PROJECT_ROOT / "reports"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    subsets: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
    rul_clip: int = 125
    windows: tuple[int, ...] = (5, 10, 20, 30)
    calib_frac: float = 0.2
    alpha: float = 0.1
    seed: int = 42

    def replace(self, **kwargs) -> "Config":
        return replace(self, **kwargs)
