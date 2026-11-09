"""Tracker for other karts and obstacles, with no ROS.

https://github.com/EVC-Purdue/A27/issues/29
"""

import math

import numpy as np
import pytest

from autonomous_kart.nodes.localization.tracker import (
    CLASS_KART,
    CLASS_OBSTACLE,
    CLASS_UNKNOWN,
    KartTracker,
    TrackFrame,
    WorldFrame,
)

DT_S = 1.0 / 30.0
SIGMA_M = 0.3
R = np.diag([SIGMA_M**2, SIGMA_M**2])
NONE_XY, NONE_COV = np.zeros((0, 2)), np.zeros((0, 2, 2))


def circle_line(radius_m=12.0, n=200):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    # psi column deliberately garbage: TrackFrame derives heading from xy
    return [(radius_m * t, radius_m * math.sin(t), radius_m - radius_m * math.cos(t), 0.0, 1.0 / radius_m) for t in th]


def stadium_line(straight_m=60.0, radius_m=15.0, spacing_m=1.0):
    """Closed track: two straights joined by two half circles, ~250 m."""
    pts = []
    perimeter = 2 * straight_m + 2 * math.pi * radius_m
    for s in np.arange(0.0, perimeter, spacing_m):
        if s < straight_m:
            x, y, k = s, 0.0, 0.0
        elif s < straight_m + math.pi * radius_m:
            a = (s - straight_m) / radius_m
            x, y, k = straight_m + radius_m * math.sin(a), radius_m * (1 - math.cos(a)), 1.0 / radius_m
        elif s < 2 * straight_m + math.pi * radius_m:
            x, y, k = straight_m - (s - straight_m - math.pi * radius_m), 2 * radius_m, 0.0
        else:
            a = (s - 2 * straight_m - math.pi * radius_m) / radius_m
            x, y, k = -radius_m * math.sin(a), radius_m * (1 + math.cos(a)), 1.0 / radius_m
        pts.append((s, x, y, 0.0, k))
    return pts


def follow(tracker, frame, v_mps, d_m, steps, rng, drop=lambda k: False, in_fov=None):
    """One kart at constant s_dot along the line. Returns (last output, velocity RMS error, ids)."""
    errs, ids, out = [], set(), []
    for k in range(steps):
        t = k * DT_S
        s = (v_mps * t) % frame.s_total
        x, y, vx, vy = frame.to_world(s, d_m, v_mps, 0.0)
        z = NONE_XY if drop(k) else (np.array([[x, y]]) + rng.normal(0.0, SIGMA_M, 2))
        out = tracker.step(t, z, np.repeat(R[None], len(z), axis=0), ego_s=s, ego_s_dot=v_mps, in_fov=in_fov)
        if k > 60 and out:
            errs.append(math.hypot(out[0].vx - vx, out[0].vy - vy))
            ids |= {o.id for o in out}
    return out, (np.sqrt(np.mean(np.square(errs))) if errs else math.nan), ids


# Frames


def test_track_frame_closes_and_round_trips():
    fr = TrackFrame(stadium_line())
    assert fr.s_total == pytest.approx(2 * 60 + 2 * math.pi * 15, abs=1.5)
    for s, d in [(10.0, 0.0), (80.0, 1.5), (150.0, -1.0), (fr.s_total - 1.0, 0.5)]:
        x, y, _, _ = fr.to_world(s, d)
        sd, _ = fr.to_frenet(np.array([[x, y]]), np.eye(2)[None], hint_s=s)
        assert float(fr.wrap_ds(sd[0, 0] - s)) == pytest.approx(0.0, abs=0.1)
        assert sd[0, 1] == pytest.approx(d, abs=0.1)


def test_track_frame_covariance_rotates_with_line():
    fr = TrackFrame(circle_line())
    s = 0.25 * fr.s_total  # theta = pi/2: point (12, 12), tangent +y
    x, y, _, _ = fr.to_world(s, 0.0)
    _, cf = fr.to_frenet(np.array([[x, y]]), np.diag([1.0, 0.01])[None], hint_s=s)
    assert cf[0, 1, 1] == pytest.approx(1.0, rel=0.05)  # world x uncertainty is lateral here
    assert cf[0, 0, 0] == pytest.approx(0.01, rel=0.2)


def test_world_frame_is_identity():
    fr = WorldFrame()
    xy, cov = fr.to_frenet(np.array([[3.0, -2.0]]), np.eye(2)[None])
    assert xy.tolist() == [[3.0, -2.0]] and fr.to_world(1.0, 2.0, 3.0, 4.0) == (1.0, 2.0, 3.0, 4.0)
    assert fr.wrap_s(-5.0) == -5.0


# Accuracy


def test_track_frame_corner_velocity_beats_world_frame():
    # 7 m/s round a 12 m circle: the track frame explains the turn; x/y
    # constant velocity sees it as acceleration.
    fr = TrackFrame(circle_line(12.0))
    _, rms_track, ids = follow(KartTracker(fr), fr, 7.0, 0.0, 300, np.random.default_rng(0))
    assert ids == {1} and rms_track < 0.6

    tr_world = KartTracker(WorldFrame())
    rng = np.random.default_rng(0)
    errs = []
    for k in range(300):
        t = k * DT_S
        x, y, vx, vy = fr.to_world((7.0 * t) % fr.s_total, 0.0, 7.0, 0.0)
        out = tr_world.step(t, np.array([[x, y]]) + rng.normal(0, SIGMA_M, 2), R[None])
        if k > 60:
            errs.append(math.hypot(out[0].vx - vx, out[0].vy - vy))
    rms_world = np.sqrt(np.mean(np.square(errs)))
    assert rms_track < 0.6 * rms_world


def test_world_frame_tracks_straight_line_velocity():
    tr = KartTracker(WorldFrame())
    rng = np.random.default_rng(1)
    for k in range(90):
        t = k * DT_S
        out = tr.step(t, np.array([[5.0 + 6.0 * t, 1.0 + 2.0 * t]]) + rng.normal(0, SIGMA_M, 2), R[None])
    assert out[0].vx == pytest.approx(6.0, abs=0.6) and out[0].vy == pytest.approx(2.0, abs=0.6)
    assert out[0].heading_rad == pytest.approx(math.atan2(2.0, 6.0), abs=0.1)


def test_lap_keeps_one_id_across_seam():
    fr = TrackFrame(stadium_line())
    laps_s = 1.2 * fr.s_total / 7.0
    out, rms, ids = follow(KartTracker(fr), fr, 7.0, 1.0, int(laps_s / DT_S), np.random.default_rng(2))
    assert ids == {1} and rms < 0.6
    assert out[0].d == pytest.approx(1.0, abs=0.4) and out[0].s_dot == pytest.approx(7.0, abs=0.6)


def test_world_covariance_matches_track_covariance_on_a_straight():
    fr = TrackFrame(stadium_line())
    out, _, _ = follow(KartTracker(fr), fr, 6.0, 0.0, 60, np.random.default_rng(3))
    o = out[0]
    assert o.s < 60.0  # still on the first straight, heading +x
    assert np.allclose(o.cov_world, o.cov, atol=1e-6)


# Lifecycle (issue #29 "done when")


def test_single_false_positive_never_confirmed():
    tr = KartTracker(WorldFrame())
    assert tr.step(0.0, np.array([[10.0, 0.0]]), R[None]) == []
    for k in range(1, 20):
        assert tr.step(k * DT_S, NONE_XY, NONE_COV) == []


def test_keeps_one_id_through_short_dropout():
    fr = TrackFrame(stadium_line())
    out, _, ids = follow(KartTracker(fr), fr, 6.0, 0.0, 90, np.random.default_rng(4), drop=lambda k: 40 <= k < 48)
    assert ids == {1} and [o.id for o in out] == [1]


def test_confirmed_track_dropped_after_coasting_in_view():
    tr = KartTracker(WorldFrame(), {"max_coast_s": 0.4})
    for k in range(30):
        tr.step(k * DT_S, np.array([[10.0 + 0.1 * k, 0.0]]), R[None])
    t0 = 29 * DT_S
    assert len(tr.step(t0 + 0.2, NONE_XY, NONE_COV)) == 1
    assert tr.step(t0 + 0.5, NONE_XY, NONE_COV) == []


def test_out_of_view_track_coasts_longer_and_cannot_steal_detections():
    tr = KartTracker(WorldFrame(), {"max_coast_s": 0.4, "max_coast_out_of_fov_s": 2.0})
    for k in range(30):
        tr.step(k * DT_S, np.array([[2.0, 3.0]]), R[None])  # kart beside us
    t0 = 29 * DT_S
    ahead_only = lambda x, y: x > 5.0  # noqa: E731  only the kart ahead is visible
    # A new kart appears ahead, inside the coasting track's (wide) gate
    out = tr.step(t0 + 1.0, np.array([[6.0, 3.0]]), R[None], in_fov=ahead_only)
    assert [o.id for o in out] == [1]  # the unseen kart survives, untouched
    assert out[0].x == pytest.approx(2.0, abs=0.3)
    assert len(tr._tracks) == 2  # the kart ahead got its own (tentative) track


def test_duplicate_box_does_not_spawn_second_track():
    tr = KartTracker(WorldFrame(), {"confirm_hits": 1})
    tr.step(0.0, np.array([[10.0, 0.0]]), R[None])
    out = tr.step(DT_S, np.array([[10.0, 0.0], [10.2, 0.1]]), np.repeat(R[None], 2, axis=0))
    assert len(out) == 1


# Classes and motion models


def test_static_obstacle_keeps_zero_velocity():
    tr = KartTracker(WorldFrame(), {"confirm_hits": 1})
    rng = np.random.default_rng(5)
    for k in range(60):
        out = tr.step(k * DT_S, np.array([[8.0, -1.0]]) + rng.normal(0, SIGMA_M, 2), R[None], classes=[CLASS_OBSTACLE])
    assert out[0].object_class == CLASS_OBSTACLE
    assert (out[0].vx, out[0].vy) == (0.0, 0.0)
    assert out[0].x == pytest.approx(8.0, abs=0.2)


def test_kart_does_not_update_obstacle_track():
    tr = KartTracker(WorldFrame(), {"confirm_hits": 1})
    tr.step(0.0, np.array([[8.0, 0.0]]), R[None], classes=[CLASS_OBSTACLE])
    tr.step(DT_S, np.array([[8.6, 0.0]]), R[None], classes=[CLASS_KART])
    obstacle = next(t for t in tr._tracks if t.id == 1)
    assert obstacle.object_class == CLASS_OBSTACLE
    assert obstacle.x[0] == pytest.approx(8.0)  # the kart measurement did not pull it
    assert obstacle.hits == 1


def test_unknown_track_takes_the_first_named_class_and_footprint_smooths():
    tr = KartTracker(WorldFrame(), {"confirm_hits": 1})
    tr.step(0.0, np.array([[8.0, 0.0]]), R[None], classes=[CLASS_UNKNOWN], footprints_m=[1.0])
    out = tr.step(DT_S, np.array([[8.0, 0.0]]), R[None], classes=[CLASS_KART], footprints_m=[2.0])
    assert out[0].object_class == CLASS_KART
    assert 1.0 < out[0].footprint_m < 2.0


def test_track_frame_seeds_new_karts_with_ego_speed():
    fr = TrackFrame(circle_line(30.0))
    tr = KartTracker(fr, {"confirm_hits": 1})
    out = tr.step(0.0, np.array([[0.0, 0.0]]), R[None], ego_s=0.0, ego_s_dot=6.0)
    assert out[0].s_dot == 6.0 and out[0].speed_mps == pytest.approx(6.0, rel=1e-3)
