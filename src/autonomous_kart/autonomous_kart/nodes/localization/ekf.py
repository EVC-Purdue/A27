"""Extended Kalman filter over [px, py, yaw, v] for kart localization.

https://github.com/EVC-Purdue/A27/issues/28
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def wrap(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Step:
    """One stored predict and the updates applied after it, for replaying a late measurement."""

    stamp: float
    inputs: tuple
    x: np.ndarray
    P: np.ndarray
    updates: list = field(default_factory=list)


class LocalizationEKF:
    def __init__(self, pos_noise: float, yaw_drift_noise: float, speed_drift_noise: float, history_s: float):
        self.pos_noise = pos_noise  # m^2/s
        self.yaw_drift_noise = yaw_drift_noise  # rad^2/s
        self.speed_drift_noise = speed_drift_noise  # (m/s)^2/s
        self.history_s = history_s

        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e6
        self.initialized = False
        self.stamp = None
        self.history: deque[Step] = deque()

    def reset(self, px: float, py: float, yaw: float, v: float, P: np.ndarray, stamp: float) -> None:
        """Seed the filter with a known state from a confident GPS fix."""
        self.x[:] = (px, py, wrap(yaw), v)
        self.P = np.array(P, dtype=float)
        self.initialized = True
        self.stamp = stamp
        self.history.clear()

    def predict(
            self,
            dt: float,
            omega_z: float,
            accel_x: float,
            omega_var: float,
            accel_var: float,
            dv: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Roll the state forward dt seconds from IMU yaw rate and longitudinal accel.

        dv is a body-frame velocity correction (m/s) added to the position integration.
        """
        px, py, yaw, v = self.x
        sin_y, cos_y = math.sin(yaw), math.cos(yaw)
        dvx, dvy = dv

        self.x[0] = px + ((v + dvx) * cos_y - dvy * sin_y) * dt
        self.x[1] = py + ((v + dvx) * sin_y + dvy * cos_y) * dt
        self.x[2] = wrap(yaw + omega_z * dt)
        self.x[3] = v + accel_x * dt

        F = np.eye(4)
        F[0, 2] = -((v + dvx) * sin_y + dvy * cos_y) * dt
        F[0, 3] = cos_y * dt
        F[1, 2] = ((v + dvx) * cos_y - dvy * sin_y) * dt
        F[1, 3] = sin_y * dt

        Q = np.diag(
            [
                self.pos_noise * dt,
                self.pos_noise * dt,
                omega_var * dt * dt + self.yaw_drift_noise * dt,
                accel_var * dt * dt + self.speed_drift_noise * dt,
            ]
        )

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def advance(self, stamp: float, omega_z: float, accel_x: float, omega_var: float, accel_var: float, dv=(0.0, 0.0)):
        """Predict up to stamp and remember the step so a late measurement can be replayed over it."""
        inputs = (stamp - self.stamp, omega_z, accel_x, omega_var, accel_var, dv)
        self.predict(*inputs)
        self.stamp = stamp
        self.history.append(Step(stamp, inputs, self.x.copy(), self.P.copy()))
        while stamp - self.history[0].stamp > self.history_s:
            self.history.popleft()

    def update_at(self, stamp: float, update, *args) -> bool:
        """Apply update(*args) as of stamp, replaying the stored predicts after it when stamp is late."""
        later = [i for i, step in enumerate(self.history) if step.stamp > stamp]
        if not later or later[0] == 0:
            # In order, or older than the history: apply now
            if self.history:
                self.history[-1].updates.append((update, args))
            return update(*args)

        start = self.history[later[0] - 1]
        self.x, self.P = start.x.copy(), start.P.copy()
        for fn, fn_args in start.updates:
            fn(*fn_args)
        accepted = update(*args)
        start.updates.append((update, args))
        for step in list(self.history)[later[0]:]:
            self.predict(*step.inputs)
            step.x, step.P = self.x.copy(), self.P.copy()
            for fn, fn_args in step.updates:
                fn(*fn_args)
        return accepted

    def update_gps_xy(self, x: float, y: float, R_xy: np.ndarray, gate: float) -> bool:
        """Fold in a GPS position fix (m); R_xy is the 2x2 GPS covariance."""
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return self._linear_update(H, np.array([x, y]), R_xy, gate)

    def update_heading(self, yaw_meas: float, var: float, gate: float) -> bool:
        """Fold in a heading observation (rad)."""
        H = np.zeros((1, 4))
        H[0, 2] = 1.0
        innovation = np.array([wrap(yaw_meas - self.x[2])])
        return self._linear_update(H, np.array([yaw_meas]), np.array([[var]]), gate, innovation=innovation)

    def update_speed(self, v_meas: float, var: float, gate: float) -> bool:
        """Fold in a speed observation (m/s)."""
        H = np.zeros((1, 4))
        H[0, 3] = 1.0
        return self._linear_update(H, np.array([v_meas]), np.array([[var]]), gate)

    def _linear_update(
            self,
            H: np.ndarray,
            z: np.ndarray,
            R: np.ndarray,
            gate: float,
            innovation: np.ndarray | None = None,
    ) -> bool:
        """Fold a linear measurement into the state unless its Mahalanobis distance fails the chi-square gate."""
        if innovation is None:
            innovation = z - H @ self.x

        S = H @ self.P @ H.T + R
        if innovation @ np.linalg.solve(S, innovation) > gate:
            return False
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ innovation
        self.x[2] = wrap(self.x[2])

        # Joseph form stays numerically stable across many updates
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True
