"""Checks GRU training builds windows, fits a GRU and exports weights the numpy GRU reproduces.

https://github.com/EVC-Purdue/A27/issues/28
"""

import math

import numpy as np
import pytest

pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")

from autonomous_kart.nodes.localization.gru import GruCorrection  # noqa: E402
from autonomous_kart.nodes.localization.gru_training import (  # noqa: E402
    build_windows,
    evaluate,
    fit,
    to_npz,
)
from kart_interfaces.msg import GpsFix  # noqa: E402


def drive_log(seconds=20.0, wheel=5.0, slip=0.4):
    # A kart driving east at wheel + slip m/s with a constant lateral slide
    t = np.arange(0.0, seconds, 0.02)
    imu = [(ti, 0.0) for ti in np.arange(0.0, seconds, 0.01)]
    telemetry = [(ti, wheel, 0.0) for ti in t]
    gps = [(ti, (wheel + slip) * ti, 0.2 * ti, 0.0, GpsFix.FIX_RTK_FIXED, True) for ti in np.arange(0.0, seconds, 0.05)]
    return imu, telemetry, gps


def test_targets_are_body_velocity_residual():
    x, y = build_windows(*drive_log(), gap_s=0.1, max_fix_gap_s=0.3, seq_len=50)
    assert x.shape[1:] == (50, 4) and y.shape[1:] == (50, 2)
    assert np.nanmean(y[..., 0]) == pytest.approx(0.4, abs=1e-6)
    assert np.nanmean(y[..., 1]) == pytest.approx(0.2, abs=1e-6)


def test_fit_beats_zero_and_exports_to_numpy_gru():
    x, y = build_windows(*drive_log(), gap_s=0.1, max_fix_gap_s=0.3, seq_len=50)
    model, mu, sd = fit(x, y, hidden=8, lr=1e-2, epochs=30, batch_size=4, seed=0)
    assert evaluate(model, mu, sd, x, y) < np.nanmean(y**2)

    gru = GruCorrection(gap_s=0.1, clamp_mps=math.inf, min_speed_mps=0.0, max_gyro_rad_s=math.inf)
    gru.load(to_npz(model, mu, sd))
    got = np.array([gru.step(0.02 * i, *x[0, i, [0, 1, 3]]) for i in range(50)])
    with torch.no_grad():
        expected = model(torch.from_numpy((x[:1] - mu) / sd))[0].numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
