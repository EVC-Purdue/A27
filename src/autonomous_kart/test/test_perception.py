"""Camera and lidar measurements, and the measurement -> track chain, with no ROS.

https://github.com/EVC-Purdue/A27/issues/29
https://github.com/EVC-Purdue/A27/issues/30
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from autonomous_kart.nodes.localization.pose_history import PoseHistory, body_to_world, in_view
from autonomous_kart.nodes.localization.tracker import KartTracker, WorldFrame
from autonomous_kart.nodes.perception.camera import camera_objects
from autonomous_kart.nodes.perception.lidar import LidarParams, scan_objects
from autonomous_kart.nodes.perception.ranging import CameraModel, RangingParams, project_ground_point

CAM = CameraModel(
    fx=320.0, fy=320.0, cx=320.0, cy=240.0, width_px=640, height_px=480, height_m=1.2, pitch_rad=math.radians(5.0)
)
KART_HEIGHT_M = 1.0
RANGING = RangingParams(kart_height_m=KART_HEIGHT_M, center_offset_m=0.0)
FULL_FRAME = (0.0, 0.0, 1.0, 1.0)
IMG = np.zeros((480, 640, 3), np.uint8)


class StubDetector:
    """Returns the queued full-frame boxes, shifted into whatever crop it is given."""

    def __init__(self):
        self.boxes_px = np.zeros((0, 4))
        self.crop_offset = (0, 0)
        self.crop_shape = None

    def __call__(self, bgr):
        self.crop_shape = bgr.shape
        ox, oy = self.crop_offset
        return SimpleNamespace(boxes=self.boxes_px - (ox, oy, ox, oy))


def box_for(x_m, y_m):
    """Full-frame box of a kart whose ground contact is at body (x_m, y_m)."""
    u, v = project_ground_point(CAM, np.array([x_m]), np.array([y_m]))
    h = CAM.fy * KART_HEIGHT_M / math.hypot(x_m, y_m)
    return np.array([[u[0] - 0.6 * h, v[0] - h, u[0] + 0.6 * h, v[0]]])


# Camera


@pytest.mark.parametrize("x_m,y_m", [(6.0, 0.0), (12.0, 2.0), (20.0, -3.0)])
def test_camera_recovers_kart_position(x_m, y_m):
    det = StubDetector()
    det.boxes_px = box_for(x_m, y_m)
    xy, cov, _ = camera_objects(det, CAM, RANGING, FULL_FRAME, IMG, 40.0)
    assert xy[0, 0] == pytest.approx(x_m, rel=0.03)
    assert xy[0, 1] == pytest.approx(y_m, abs=0.05 + 0.01 * x_m)
    assert np.all(np.linalg.eigvalsh(cov[0]) > 0)


def test_camera_roi_crops_and_offsets_boxes():
    det = StubDetector()
    det.boxes_px = box_for(10.0, 0.0)
    det.crop_offset = (64, 0)
    roi = (0.1, 0.0, 0.9, 0.7)
    xy, _, boxes = camera_objects(det, CAM, RANGING, roi, IMG, 40.0)
    assert det.crop_shape == (336, 512, 3)
    assert boxes[0] == pytest.approx(box_for(10.0, 0.0)[0])  # back in full-frame pixels
    assert xy[0, 0] == pytest.approx(10.0, rel=0.03)


def test_camera_box_cut_by_roi_is_truncated_not_trusted():
    # A box ending on the ROI bottom hides the ground contact; the ground cue
    # would put it too far, so only the (wider) size cue may be used.
    det = StubDetector()
    det.boxes_px = np.array([[300.0, 230.0, 340.0, 288.0]])  # bottom at row 288 = 0.6 * 480
    xy_roi, cov_roi, _ = camera_objects(det, CAM, RANGING, (0.0, 0.0, 1.0, 0.6), IMG, 40.0)
    xy_full, cov_full, _ = camera_objects(det, CAM, RANGING, FULL_FRAME, IMG, 40.0)
    assert cov_roi[0, 0, 0] > cov_full[0, 0, 0]


def test_camera_drops_karts_beyond_max_range():
    det = StubDetector()
    det.boxes_px = box_for(35.0, 0.0)
    xy, _, _ = camera_objects(det, CAM, RANGING, FULL_FRAME, IMG, 30.0)
    assert xy.shape == (0, 2)


# Lidar


def scan(objects, n=720, max_range_m=30.0, barrier_y_m=None, rng=None):
    """Synthetic 360 deg scan: (x, y, half_size) squares, plus an optional straight barrier."""
    a = -math.pi + 2 * math.pi * np.arange(n) / n
    r = np.full(n, np.inf)
    d = np.column_stack([np.cos(a), np.sin(a)])
    for cx, cy, half in objects:
        for lo, hi, axis in ((cx - half, cx + half, 0), (cy - half, cy + half, 1)):
            for edge in (lo, hi):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = edge / d[:, axis]
                hit = t * d[:, 1 - axis]
                c_lo, c_hi = (cy - half, cy + half) if axis == 0 else (cx - half, cx + half)
                ok = (t > 0) & (hit >= c_lo) & (hit <= c_hi)
                r = np.where(ok & (t < r), t, r)
    if barrier_y_m is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = barrier_y_m / d[:, 1]
        r = np.where((t > 0) & (t < r), t, r)
    r[r > max_range_m] = np.inf
    if rng is not None:
        r = r + rng.normal(0.0, 0.02, n)
    return r, -math.pi, 2 * math.pi / n


def test_lidar_finds_karts_ahead_and_beside():
    ranges, a0, da = scan([(8.0, 0.0, 0.7), (0.5, 2.5, 0.7)], rng=np.random.default_rng(0))
    objs = scan_objects(ranges, a0, da, LidarParams())
    assert len(objs.xy_m) == 2
    found = sorted(objs.xy_m.tolist())
    assert found[0] == pytest.approx([0.5, 2.5], abs=0.5)  # beside us: the camera cannot see this one
    assert found[1] == pytest.approx([8.0, 0.0], abs=0.5)
    assert np.all(np.linalg.eigvalsh(objs.cov_m2) > 0)


def test_lidar_drops_long_barrier_and_keeps_kart():
    ranges, a0, da = scan([(10.0, 0.0, 0.7)], barrier_y_m=4.0)
    objs = scan_objects(ranges, a0, da, LidarParams())
    assert len(objs.xy_m) == 1 and objs.xy_m[0] == pytest.approx([10.0, 0.0], abs=0.5)


def test_lidar_mount_transform():
    ranges, a0, da = scan([(8.0, 0.0, 0.7)])
    objs = scan_objects(ranges, a0, da, LidarParams(mount_x_m=1.0, mount_yaw_rad=math.pi / 2))
    assert objs.xy_m[0] == pytest.approx([1.0, 8.0], abs=0.5)  # rotated 90 deg, then shifted


def test_lidar_empty_scan():
    objs = scan_objects(np.full(360, np.inf), -math.pi, 2 * math.pi / 360, LidarParams())
    assert objs.xy_m.shape == (0, 2)


# Pose history


def test_pose_history_interpolates_through_pi_and_bounds_gaps():
    poses = PoseHistory(horizon_s=1.0)
    assert poses.at(1.0, 0.25) is None
    poses.add(1.0, (0.0, 0.0, math.pi - 0.1, 5.0))
    poses.add(1.1, (1.0, 2.0, -math.pi + 0.1, 7.0))
    x, y, yaw, v = poses.at(1.05, 0.25)
    assert (x, y, v) == pytest.approx((0.5, 1.0, 6.0))
    assert abs(abs(yaw) - math.pi) < 1e-9
    assert poses.at(1.3, 0.25) == (1.0, 2.0, -math.pi + 0.1, 7.0)
    assert poses.at(2.0, 0.25) is None
    poses.add(0.5, (9.0, 9.0, 0.0, 0.0))  # out of order: ignored
    assert poses.at(1.0, 0.25)[0] == 0.0


def test_body_to_world_rotates_and_adds_pose_uncertainty():
    w, c = body_to_world(
        np.array([[10.0, 0.0]]), np.array([np.diag([0.1, 0.01])]), (5.0, 5.0, math.pi / 2, 0.0), 0.0, 0.0
    )
    assert w[0] == pytest.approx([5.0, 15.0])
    assert c[0, 1, 1] == pytest.approx(0.1) and c[0, 0, 0] == pytest.approx(0.01)
    _, c2 = body_to_world(
        np.array([[10.0, 0.0]]), np.array([np.diag([0.1, 0.01])]), (5.0, 5.0, math.pi / 2, 0.0), 0.0, 0.02
    )
    assert c2[0, 0, 0] == pytest.approx(0.01 + (10 * 0.02) ** 2)  # yaw error grows with range


def test_in_view():
    check = in_view((0.0, 0.0, 0.0, 0.0), math.radians(90.0), 30.0)
    assert check(10.0, 0.0) and not check(0.0, 10.0) and not check(40.0, 0.0)


# Measurement -> track chain (issue #29: position and velocity error against truth)


def test_camera_to_track_chain_recovers_opponent_position_and_velocity():
    """Ego drives +x at 6 m/s, an opponent 12 m ahead at 4 m/s; camera measurements, 30 Hz, 3 s."""
    rng = np.random.default_rng(7)
    det = StubDetector()
    tracker = KartTracker(WorldFrame())
    pos_err, vel_err = [], []
    for k in range(90):
        t = k / 30.0
        ego = (6.0 * t, 0.0, 0.0, 6.0)
        opp = (12.0 + 4.0 * t, 1.0)
        bx, by = opp[0] - ego[0], opp[1]
        det.boxes_px = box_for(bx, by) + rng.normal(0.0, 1.5, 4)  # pixel-level box jitter
        xy, cov, _ = camera_objects(det, CAM, RANGING, FULL_FRAME, IMG, 40.0)
        world_xy, world_cov = body_to_world(xy, cov, ego, 0.1, math.radians(1.0))
        out = tracker.step(t, world_xy, world_cov)
        if k >= 30 and out:
            pos_err.append(math.hypot(out[0].x - opp[0], out[0].y - opp[1]))
            vel_err.append(math.hypot(out[0].vx - 4.0, out[0].vy))
    assert len({o.id for o in out}) == 1
    assert np.sqrt(np.mean(np.square(pos_err))) < 0.6
    assert np.sqrt(np.mean(np.square(vel_err))) < 1.0
