"""Kart and obstacle candidates from a 2D LaserScan.

https://github.com/EVC-Purdue/A27/issues/30

The scan is split into clusters wherever consecutive returns jump apart, and
clusters whose extent is kart- or obstacle-sized become measurements. Long
clusters (barriers, walls) are dropped. A lidar blob cannot tell a kart from a
tyre stack, so measurements are CLASS_UNKNOWN; the tracker names the track
when the camera sees it. Unlike the forward camera, the scan covers the karts
beside us during a pass.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarParams:
    mount_x_m: float = 0.0  # lidar position and heading in the body frame
    mount_y_m: float = 0.0
    mount_yaw_rad: float = 0.0
    max_range_m: float = 30.0
    # Split when consecutive returns are further apart than this, or than
    # gap_beam_factor beam spacings at that range (far returns are sparser).
    cluster_gap_m: float = 0.4
    gap_beam_factor: float = 4.0
    min_points: int = 3
    min_extent_m: float = 0.3
    max_extent_m: float = 2.5
    range_sigma_m: float = 0.05  # per return
    # The scan only sees the near face, so the centre is about half a
    # footprint further along the ray than the points' mean.
    center_push_frac: float = 0.5


@dataclass
class LidarObjects:
    xy_m: np.ndarray  # (M, 2) centres, body frame
    cov_m2: np.ndarray  # (M, 2, 2)
    extent_m: np.ndarray  # (M,) largest dimension of the visible points


def _empty() -> LidarObjects:
    return LidarObjects(np.zeros((0, 2)), np.zeros((0, 2, 2)), np.zeros(0))


def scan_objects(ranges_m, angle_min_rad: float, angle_increment_rad: float, p: LidarParams) -> LidarObjects:
    r = np.asarray(ranges_m, dtype=np.float64)
    a = angle_min_rad + angle_increment_rad * np.arange(r.size)
    ok = np.isfinite(r) & (r > 0.05) & (r <= p.max_range_m)
    if ok.sum() < p.min_points:
        return _empty()
    r, a = r[ok], a[ok]
    pts = np.column_stack([r * np.cos(a), r * np.sin(a)])  # lidar frame

    step = np.hypot(*np.diff(pts, axis=0).T)
    beam_gap = np.diff(a) > 1.5 * angle_increment_rad  # dropped returns between
    limit = np.maximum(p.cluster_gap_m, p.gap_beam_factor * r[:-1] * angle_increment_rad)
    breaks = np.flatnonzero((step > limit) | beam_gap) + 1
    clusters = np.split(np.arange(len(pts)), breaks)
    # A full-circle scan wraps: join the last cluster onto the first if they touch
    if len(clusters) > 1 and np.hypot(*(pts[-1] - pts[0])) <= max(p.cluster_gap_m, 1e-9):
        clusters = [np.concatenate([clusters[-1], clusters[0]])] + clusters[1:-1]

    c, s = math.cos(p.mount_yaw_rad), math.sin(p.mount_yaw_rad)
    rot = np.array([[c, -s], [s, c]])
    xy, cov, ext = [], [], []
    for idx in clusters:
        if idx.size < p.min_points:
            continue
        q = pts[idx]
        extent = float(np.max(np.hypot(*(q[:, None, :] - q[None, :, :]).transpose(2, 0, 1))))
        if not (p.min_extent_m <= extent <= p.max_extent_m):
            continue
        centre = q.mean(axis=0)
        rng = float(np.hypot(*centre))
        ray = centre / max(rng, 1e-9)
        centre = centre + ray * (p.center_push_frac * 0.5 * extent)
        # Along the ray: return noise plus the unknown depth of the far side;
        # across it: spread of the visible face.
        var_along = p.range_sigma_m**2 + (0.25 * extent) ** 2
        var_across = p.range_sigma_m**2 + (extent / math.sqrt(12.0)) ** 2 / idx.size
        basis = np.array([[ray[0], -ray[1]], [ray[1], ray[0]]])
        cov_l = basis @ np.diag([var_along, var_across]) @ basis.T
        xy.append(rot @ centre + (p.mount_x_m, p.mount_y_m))
        cov.append(rot @ cov_l @ rot.T)
        ext.append(extent)
    if not xy:
        return _empty()
    return LidarObjects(np.array(xy), np.array(cov), np.array(ext))
