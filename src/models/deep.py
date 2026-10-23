"""Strong window model: dilated conv + transformer encoder + attention pooling.

Why this shape:
- Convolutions over the window capture local degradation patterns at several
  time scales (dilation 1/2/4).
- A small transformer encoder mixes information across the whole window.
- Attention pooling lets the model weight the cycles that matter most instead
  of averaging them (mean pooling hid the last-cycle signal).
- For the multi-condition subsets (FD002/FD004) a learned condition embedding
  is added to the pooled representation, so the model can specialise without
  training a separate expert per condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13",
    "s_14", "s_15", "s_17", "s_20", "s_21",
]  # 14 informative sensors (drops the 7 near-constant ones)


def make_windows(
    df: pd.DataFrame, sensors: list[str], window: int, cap: float = 125.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All windows of length `window` per unit.

    Returns X (n, window, n_sensors), y (n,), unit (n,), cycle (n,). The target
    is the truncated RUL at the window's last cycle.
    """
    xs, ys, units, cycles, conds = [], [], [], [], []
    for unit, sub in df.groupby("unit", sort=True):
        values = sub[sensors].to_numpy(dtype=np.float32)
        n = len(values)
        max_cycle = int(sub["cycle"].iloc[-1])
        for end in range(window - 1, n):
            xs.append(values[end - window + 1 : end + 1])
            rul = max_cycle - int(sub["cycle"].iloc[end])
            ys.append(min(cap, float(rul)))
            units.append(unit)
            cycles.append(int(sub["cycle"].iloc[end]))
            conds.append(int(sub["condition"].iloc[end]))
    return (
        np.stack(xs) if xs else np.empty((0, window, len(sensors)), dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(units, dtype=np.int32),
        np.asarray(cycles, dtype=np.int32),
        np.asarray(conds, dtype=np.int32),
    )


def last_windows(
    df: pd.DataFrame, sensors: list[str], window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window per unit: the last `window` observed cycles (left-padded)."""
    xs, units, conds = [], [], []
    for unit, sub in df.groupby("unit", sort=True):
        values = sub[sensors].to_numpy(dtype=np.float32)
        if len(values) >= window:
            xs.append(values[-window:])
        else:
            pad = np.repeat(values[:1], window - len(values), axis=0)
            xs.append(np.concatenate([pad, values], axis=0))
        units.append(unit)
        conds.append(int(sub["condition"].iloc[-1]))
    return np.stack(xs), np.asarray(units, dtype=np.int32), np.asarray(conds, dtype=np.int32)


@dataclass
class TrainedDeep:
    model: nn.Module
    sensors: list[str]
    window: int
    cond_levels: list[int]
    cond_mean: np.ndarray
    cond_std: np.ndarray
    val_rmse: float
    epochs: int

    def predict(self, X: np.ndarray, cond: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.from_numpy(X), torch.from_numpy(cond))
        return pred.numpy().ravel()


class ConvTransformer(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        n_cond: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cond_emb = nn.Embedding(max(n_cond, 1) + 1, d_model)
        self.inp = nn.Conv1d(n_sensors, d_model, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 5, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=8, dilation=4),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.zeros(256, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.attn = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, D, T)
        h = self.conv(self.inp(x.transpose(1, 2))).transpose(1, 2)
        h = self.norm(h)
        h = h + self.pos[: h.size(1)].unsqueeze(0)
        h = self.encoder(h)
        w = torch.softmax(self.attn(h), dim=1)
        pooled = (h * w).sum(dim=1)
        cond_vec = self.cond_emb(cond.long())
        z = torch.cat([pooled, cond_vec], dim=-1)
        return self.head(z).squeeze(-1)
