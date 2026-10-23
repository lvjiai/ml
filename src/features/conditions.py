"""Operating-condition discovery and normalisation.

FD002/FD004 mix six operating conditions in which the same sensor reading means
different things.  Clustering the three operational settings and standardising
each sensor *within* its condition removes that nuisance variation before any
feature is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data.schema import SENSORS, SETTINGS

# How many operating conditions each subset actually contains (from the
# C-MAPSS readme: FD001/FD003 = one, FD002/FD004 = six).
N_CONDITIONS: dict[str, int] = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass
class ConditionNormalizer:
    """Fit on the training split only; reused unchanged for test data."""

    n_clusters: int = 1
    seed: int = 42
    setting_scaler: StandardScaler = field(default_factory=StandardScaler)
    kmeans: KMeans | None = None
    means: np.ndarray | None = None  # (n_clusters, n_sensors)
    stds: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "ConditionNormalizer":
        z = self.setting_scaler.fit_transform(train[SETTINGS].to_numpy())
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.seed
        ).fit(z)
        labels = self.kmeans.labels_
        self.means = np.zeros((self.n_clusters, len(SENSORS)))
        self.stds = np.ones((self.n_clusters, len(SENSORS)))
        values = train[SENSORS].to_numpy(dtype=np.float64)
        for k in range(self.n_clusters):
            mask = labels == k
            self.means[k] = values[mask].mean(axis=0)
            # A degenerate (constant) sensor would give std == 0; keep it at 1
            # so that the normalised value stays finite.
            self.stds[k] = np.where(values[mask].std(axis=0) > 1e-9,
                                    values[mask].std(axis=0), 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.kmeans is None:
            raise RuntimeError("ConditionNormalizer.transform called before fit")
        z = self.setting_scaler.transform(df[SETTINGS].to_numpy())
        labels = self.kmeans.predict(z)
        out = df.copy()
        values = df[SENSORS].to_numpy(dtype=np.float64)
        out[SENSORS] = ((values - self.means[labels]) / self.stds[labels]).astype(
            "float32"
        )
        out["condition"] = labels.astype("int16")
        return out
