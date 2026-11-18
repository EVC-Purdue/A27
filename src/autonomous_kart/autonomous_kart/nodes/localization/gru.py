"""Numpy GRU that predicts a body-frame velocity correction for the EKF.

https://github.com/EVC-Purdue/A27/issues/28
"""

import io

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def features(wheel, gyro_z, wheel_accel, steering):
    return np.array([wheel, gyro_z, wheel_accel, steering])


class GruCorrection:
    """One-layer GRU with the torch.nn.GRU gate order (r, z, n) and a linear head to (dvx, dvy)."""

    def __init__(self, gap_s: float, clamp_mps: float, min_speed_mps: float, max_gyro_rad_s: float):
        self.gap_s = gap_s
        self.clamp_mps = clamp_mps
        self.min_speed_mps = min_speed_mps
        self.max_gyro_rad_s = max_gyro_rad_s
        self.weights = None
        self.h = None
        self.last_t = None
        self.last_wheel = 0.0

    def load(self, npz: bytes) -> None:
        self.weights = dict(np.load(io.BytesIO(npz)))
        self.h = None

    def cell(self, x, h):
        w = self.weights
        gi = w["w_ih"] @ x + w["b_ih"]
        gh = w["w_hh"] @ h + w["b_hh"]
        i_r, i_z, i_n = np.split(gi, 3)
        h_r, h_z, h_n = np.split(gh, 3)
        r = sigmoid(i_r + h_r)
        z = sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h

    def step(self, t: float, wheel: float, gyro_z: float, steering: float) -> np.ndarray:
        """Advance the GRU one telemetry sample and return the guarded (dvx, dvy) in m/s."""
        if self.last_t is None or t - self.last_t > self.gap_s or t <= self.last_t:
            self.h = None
            wheel_accel = 0.0
        else:
            wheel_accel = (wheel - self.last_wheel) / (t - self.last_t)
        self.last_t, self.last_wheel = t, wheel
        if self.weights is None:
            return np.zeros(2)

        if self.h is None:
            self.h = np.zeros(self.weights["w_hh"].shape[1])
        x = (features(wheel, gyro_z, wheel_accel, steering) - self.weights["mu"]) / self.weights["sd"]
        self.h = self.cell(x, self.h)
        dv = self.weights["w_head"] @ self.h + self.weights["b_head"]

        if abs(wheel) < self.min_speed_mps or abs(gyro_z) > self.max_gyro_rad_s:
            return np.zeros(2)
        return np.clip(dv, -self.clamp_mps, self.clamp_mps)
