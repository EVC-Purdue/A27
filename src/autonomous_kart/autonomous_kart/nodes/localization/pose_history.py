"""Our recent poses, so a measurement is placed with the pose at its capture stamp.

https://github.com/EVC-Purdue/A27/issues/29

At 6 m/s, 50 ms of perception latency is 30 cm, and a yaw lag swings far
targets sideways, so arrival time is not good enough.
"""

import bisect
import math
from collections import deque

import numpy as np

Pose = tuple[float, float, float, float]  # x_m, y_m, yaw_rad, speed_mps


class PoseHistory:
    def __init__(self, horizon_s: float):
        self.horizon_s = horizon_s
        self.times_s: deque[float] = deque()
        self.poses: deque[Pose] = deque()

    def add(self, time_s: float, pose: Pose) -> None:
        if self.times_s and time_s <= self.times_s[-1]:
            return
        self.times_s.append(time_s)
        self.poses.append(pose)
        while self.times_s[0] < time_s - self.horizon_s:
            self.times_s.popleft()
            self.poses.popleft()

    def at(self, time_s: float, max_gap_s: float) -> Pose | None:
        """Interpolated pose at time_s; the nearest one within max_gap_s outside the history."""
        if not self.times_s:
            return None
        i = bisect.bisect_left(self.times_s, time_s)
        if i == 0 or i == len(self.times_s):
            j = 0 if i == 0 else len(self.times_s) - 1
            return self.poses[j] if abs(self.times_s[j] - time_s) <= max_gap_s else None
        t0, t1 = self.times_s[i - 1], self.times_s[i]
        a = (time_s - t0) / (t1 - t0)
        p0, p1 = self.poses[i - 1], self.poses[i]
        dyaw = (p1[2] - p0[2] + math.pi) % (2 * math.pi) - math.pi
        return (
            p0[0] + a * (p1[0] - p0[0]),
            p0[1] + a * (p1[1] - p0[1]),
            p0[2] + a * dyaw,
            p0[3] + a * (p1[3] - p0[3]),
        )


def body_to_world(
    xy_m: np.ndarray, cov_m2: np.ndarray, pose: Pose, pose_xy_sigma_m: float, pose_yaw_sigma_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame points + covariances -> track frame, adding our own pose uncertainty
    (yaw error grows with range)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rot = np.array([[c, -s], [s, c]])
    drot = np.array([[-s, -c], [c, -s]])  # d rot / d yaw
    xy = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
    world = xy @ rot.T + (pose[0], pose[1])
    cov = rot @ np.asarray(cov_m2, dtype=np.float64).reshape(-1, 2, 2) @ rot.T
    j = xy @ drot.T
    cov = cov + pose_yaw_sigma_rad**2 * j[:, :, None] * j[:, None, :] + pose_xy_sigma_m**2 * np.eye(2)
    return world, cov


def in_view(pose: Pose, fov_rad: float, max_range_m: float):
    """Whether a track-frame point is where the sensors would see it (for out-of-view coasting)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])

    def check(x_m: float, y_m: float) -> bool:
        dx, dy = x_m - pose[0], y_m - pose[1]
        bx, by = c * dx + s * dy, -s * dx + c * dy
        return math.hypot(bx, by) <= max_range_m and abs(math.atan2(by, bx)) <= 0.5 * fov_rad

    return check
