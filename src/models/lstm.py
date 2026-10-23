"""Sequence model: LSTM regressor for RUL (W3 contrast to the GBDT model)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


class RULNet(nn.Module):
    """LSTM over a window of sensor readings -> remaining useful life."""

    def __init__(
        self,
        n_features: int = 21,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def build_sequences(
    df: pd.DataFrame,
    sensors: list[str],
    window: int,
    with_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Stack rolling windows of shape (window, n_sensors) per cycle.

    Row ``t`` uses rows ``t-window+1 .. t`` of the same unit: causal by
    construction. Units shorter than ``window`` are skipped.
    """
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta: list[tuple[int, int]] = []
    for unit, g in df.groupby("unit", sort=True):
        arr = g[sensors].to_numpy(dtype=np.float32)
        rul = g["rul"].to_numpy(dtype=np.float32) if with_target else None
        for t in range(window - 1, len(arr)):
            xs.append(arr[t - window + 1 : t + 1])
            if rul is not None:
                ys.append(float(rul[t]))
            meta.append((int(unit), int(g["cycle"].iloc[t])))
    X = np.stack(xs)
    y = np.asarray(ys, dtype=np.float32) if with_target else np.empty(0, dtype=np.float32)
    return X, y, pd.DataFrame(meta, columns=["unit", "cycle"])


def last_window(df: pd.DataFrame, sensors: list[str], window: int) -> tuple[np.ndarray, pd.DataFrame]:
    """One window per unit, ending at its last recorded cycle (test-time path).

    A few test units are recorded for fewer than ``window`` cycles; their
    earliest reading is repeated on the left so every unit still gets a row.
    """
    xs: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []
    for unit, g in df.groupby("unit", sort=True):
        arr = g[sensors].to_numpy(dtype=np.float32)[-window:]
        if len(arr) < window:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        xs.append(arr)
        meta.append((int(unit), int(g["cycle"].iloc[-1])))
    return np.stack(xs), pd.DataFrame(meta, columns=["unit", "cycle"])


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int = 64,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 6,
    seed: int = 42,
) -> tuple[RULNet, list[float], int]:
    """Train with early stopping on the validation units; returns the best weights."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = RULNet(n_features=X_train.shape[2], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=10.0)

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    x_val = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    best_rmse = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    history: list[float] = []
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_val).numpy()
        rmse = float(np.sqrt(np.mean((np.clip(pred, 0, 125) - y_val) ** 2)))
        history.append(rmse)
        if rmse < best_rmse - 1e-4:
            best_rmse = rmse
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch


@torch.no_grad()
def predict(model: RULNet, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size])
        out.append(model(xb).numpy())
    return np.concatenate(out)
