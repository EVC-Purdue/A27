"""Trains the GRU velocity correction on buffered drive data; runs in the localization node's background process.

https://github.com/EVC-Purdue/A27/issues/28
"""

import io

import numpy as np
import torch

from autonomous_kart.nodes.localization.gru import features
from kart_interfaces.msg import GpsFix


class GruModel(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gru = torch.nn.GRU(4, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)


def to_npz(model: GruModel, mu: np.ndarray, sd: np.ndarray) -> bytes:
    buf = io.BytesIO()
    w = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    np.savez(
        buf,
        w_ih=w["gru.weight_ih_l0"],
        w_hh=w["gru.weight_hh_l0"],
        b_ih=w["gru.bias_ih_l0"],
        b_hh=w["gru.bias_hh_l0"],
        w_head=w["head.weight"],
        b_head=w["head.bias"],
        mu=mu,
        sd=sd,
    )
    return buf.getvalue()


def from_npz(npz: bytes) -> tuple[GruModel, np.ndarray, np.ndarray]:
    w = np.load(io.BytesIO(npz))
    model = GruModel(w["w_hh"].shape[1]).double()
    names = {"w_ih": "gru.weight_ih_l0", "w_hh": "gru.weight_hh_l0", "b_ih": "gru.bias_ih_l0", "b_hh": "gru.bias_hh_l0"}
    names.update(w_head="head.weight", b_head="head.bias")
    model.load_state_dict({names[k]: torch.from_numpy(w[k]) for k in names})
    return model, w["mu"], w["sd"]


def build_windows(imu, telemetry, gps, gap_s: float, max_fix_gap_s: float, seq_len: int):
    """Cut telemetry into sequences of GRU features and body-frame velocity residual targets (NaN where unknown)."""
    imu = np.array(sorted(imu))
    t, wheel, steering = np.array(telemetry).T
    gyro = imu[np.clip(np.searchsorted(imu[:, 0], t, side="right") - 1, 0, None), 1]
    dt = np.diff(t, prepend=-np.inf)
    gap = dt > gap_s
    accel = np.where(gap, 0.0, np.diff(wheel, prepend=wheel[0]) / np.where(gap, 1.0, dt))
    x = features(wheel, gyro, accel, steering).T

    # Body-frame GPS velocity over each pair of consecutive RTK-fixed fixes with heading
    fixes = np.array([g[:4] for g in gps if g[4] == GpsFix.FIX_RTK_FIXED and g[5]])
    ft, fx, fy, fh = fixes.T
    fdt = np.diff(ft)
    vwx, vwy = np.diff(fx) / fdt, np.diff(fy) / fdt
    dh = np.diff(fh)
    h = fh[:-1] + np.arctan2(np.sin(dh), np.cos(dh)) / 2
    vbx = np.cos(h) * vwx + np.sin(h) * vwy
    vby = -np.sin(h) * vwx + np.cos(h) * vwy

    i = np.searchsorted(ft, t, side="right") - 1
    ok = (i >= 0) & (i < len(fdt))
    i = np.clip(i, 0, len(fdt) - 1)
    ok &= (fdt[i] > 0) & (fdt[i] <= max_fix_gap_s)
    y = np.where(ok[:, None], np.stack([vbx[i] - wheel, vby[i]], axis=1), np.nan)

    windows = [s for s in range(0, len(t) - seq_len + 1, seq_len) if not gap[s + 1 : s + seq_len].any()]
    windows = [s for s in windows if ok[s : s + seq_len].any()]
    return np.array([x[s : s + seq_len] for s in windows]), np.array([y[s : s + seq_len] for s in windows])


def masked_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(y)
    return ((pred - torch.nan_to_num(y)) ** 2)[mask].mean()


def evaluate(model, mu, sd, x, y) -> float:
    with torch.no_grad():
        return float(masked_mse(model(torch.from_numpy((x - mu) / sd)), torch.from_numpy(y)))


def fit(x, y, hidden: int, lr: float, epochs: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    mu = x.reshape(-1, 4).mean(axis=0)
    sd = x.reshape(-1, 4).std(axis=0)
    sd[sd == 0] = 1.0
    xs, ys = torch.from_numpy((x - mu) / sd), torch.from_numpy(y)
    model = GruModel(hidden).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in torch.randperm(len(xs)).split(batch_size):
            opt.zero_grad()
            loss = masked_mse(model(xs[batch]), ys[batch])
            loss.backward()
            opt.step()
    return model, mu, sd


def train(imu, telemetry, gps, current: bytes | None, p: dict) -> tuple[bytes | None, str]:
    """One training cycle: new weights if they beat the current ones on held-out data, and a log line either way."""
    torch.set_num_threads(p["train_threads"])
    try:
        x, y = build_windows(imu, telemetry, gps, p["gru_gap_s"], p["train_max_fix_gap_s"], p["train_seq_len"])
    except (ValueError, IndexError):
        return None, "GRU training skipped: not enough data"
    n_val = int(len(x) * p["train_val_fraction"])
    if n_val == 0 or n_val == len(x):
        return None, f"GRU training skipped: only {len(x)} sequences"

    model, mu, sd = fit(
        x[:-n_val],
        y[:-n_val],
        p["train_hidden"],
        p["train_lr"],
        p["train_epochs"],
        p["train_batch_size"],
        p["train_seed"],
    )
    x_val, y_val = x[-n_val:], y[-n_val:]
    new_mse = evaluate(model, mu, sd, x_val, y_val)
    old_mse = evaluate(*from_npz(current), x_val, y_val) if current else float(np.nanmean(y_val**2))
    if new_mse > old_mse * (1.0 - p["train_min_improvement"]):
        return None, f"GRU training rejected: held-out MSE {new_mse:.4f} vs current {old_mse:.4f}"
    return to_npz(model, mu, sd), f"GRU training accepted: held-out MSE {new_mse:.4f} vs {old_mse:.4f}"
