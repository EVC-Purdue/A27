"""Drives pure pursuit on a kinematic bicycle with noisy, delayed pose, on an oval and on the bundled racing line.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math
import os
import random

import pytest

from autonomous_kart.nodes.pathfinder.controllers.base import ControllerInputs, LinePoint
from autonomous_kart.nodes.pathfinder.controllers.path import Path, is_loop
from autonomous_kart.nodes.pathfinder.controllers.pure_pursuit import PurePursuitController
from autonomous_kart.nodes.pathfinder.line import load_line

LINE_CSV = os.path.join(os.path.dirname(__file__), "..", "autonomous_kart", "lines", "line.csv")
DT_S = 0.02
WHEELBASE_M = 1.05
POSE_SIGMA_M = 0.05
YAW_SIGMA_RAD = 0.01
LATENCY_TICKS = 3
SPEED_TAU_S = 0.3
STEER_TAU_S = 0.05


@pytest.fixture
def params(pathfinder_params):
    return pathfinder_params["pure_pursuit"]


def oval(straight_m=40.0, radius_m=10.0, spacing_m=0.5):
    # Two straights joined by half circles, counter-clockwise, with a 10 m/s line speed
    points = []
    for x in [i * spacing_m for i in range(int(straight_m / spacing_m))]:
        points.append(LinePoint(x, -radius_m, 0.0, 10.0))
    arc = int(math.pi * radius_m / spacing_m)
    for k in range(arc):
        a = -math.pi / 2 + math.pi * k / arc
        points.append(LinePoint(straight_m + radius_m * math.cos(a), radius_m * math.sin(a), 1.0 / radius_m, 10.0))
    for x in [straight_m - i * spacing_m for i in range(int(straight_m / spacing_m))]:
        points.append(LinePoint(x, radius_m, 0.0, 10.0))
    for k in range(arc):
        a = math.pi / 2 + math.pi * k / arc
        points.append(LinePoint(radius_m * math.cos(a), radius_m * math.sin(a), 1.0 / radius_m, 10.0))
    return tuple(points)


def straight(length=50, speed=5.0):
    return tuple(LinePoint(float(i), 0.0, 0.0, speed) for i in range(length))


def cte(line, x, y):
    # Distance to the nearest segment of the closed line
    best = math.inf
    for a, b in zip(line, line[1:] + line[:1], strict=True):
        ux, uy = b.x_m - a.x_m, b.y_m - a.y_m
        t = min(max(((x - a.x_m) * ux + (y - a.y_m) * uy) / (ux * ux + uy * uy), 0.0), 1.0)
        best = min(best, math.hypot(a.x_m + t * ux - x, a.y_m + t * uy - y))
    return best


def inputs_at(x, y, yaw, speed, line):
    return ControllerInputs(
        time_s=0.0,
        x_m=x,
        y_m=y,
        yaw_rad=yaw,
        vx_mps=speed,
        vy_mps=0.0,
        yaw_rate_radps=0.0,
        wheel_speed_mps=speed,
        steering_angle_rad=0.0,
        steering_motor_current_a=0.0,
        line=line,
    )


def drive(line, seconds, params, start=None, seed=0):
    # Rows of (x, y, speed, cte, rejoining) at every tick
    rng = random.Random(seed)
    controller = PurePursuitController(params)
    x, y, yaw = start or (line[0].x_m, line[0].y_m, Path(line).heading(0))
    v = steer = 0.0
    poses = [(x, y, yaw, v)] * (LATENCY_TICKS + 1)
    rows = []
    for _ in range(int(seconds / DT_S)):
        px, py, pyaw, pv = poses[-1 - LATENCY_TICKS]
        noisy_x = px + rng.gauss(0.0, POSE_SIGMA_M)
        noisy_y = py + rng.gauss(0.0, POSE_SIGMA_M)
        noisy_yaw = pyaw + rng.gauss(0.0, YAW_SIGMA_RAD)
        out = controller.step(inputs_at(noisy_x, noisy_y, noisy_yaw, pv, line))
        v += (out.speed_setpoint_mps - v) * DT_S / SPEED_TAU_S
        steer += (out.steering_rad - steer) * DT_S / STEER_TAU_S
        x += v * math.cos(yaw) * DT_S
        y += v * math.sin(yaw) * DT_S
        yaw += v / WHEELBASE_M * math.tan(steer) * DT_S
        poses.append((x, y, yaw, v))
        rows.append((x, y, v, cte(line, x, y), controller.rejoin is not None))
    return rows


def percentile(values, q):
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)]


def distance_driven(rows):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rows, rows[1:], strict=False))


def test_laps_oval_within_tolerance(params):
    rows = drive(oval(), 40.0, params)
    errors = [r[3] for r in rows[int(5.0 / DT_S) :]]  # after the launch from standstill
    assert percentile(errors, 0.95) < 0.3
    assert max(errors) < 0.6
    assert distance_driven(rows) > 2 * (2 * 40.0 + 2 * math.pi * 10.0)


def test_laps_racing_line(params):
    line = load_line(LINE_CSV)
    rows = drive(line, 75.0, params)
    errors = [r[3] for r in rows[int(5.0 / DT_S) :]]
    assert percentile(errors, 0.95) < 0.5
    assert not any(r[4] for r in rows)
    assert distance_driven(rows) > 421.0


def test_slows_for_corners(params):
    rows = drive(oval(), 40.0, {**params, "speed_cap_mps": 12.0})[int(10.0 / DT_S) :]
    corner = [r[2] for r in rows if r[0] > 45.0]
    on_straight = [r[2] for r in rows if 15.0 < r[0] < 25.0]
    assert max(corner) < math.sqrt(params["lateral_accel_max_mps2"] * 10.0) + 0.3
    assert max(on_straight) > max(corner) + 0.5


def test_rejoins_from_far_off_line(params):
    line = load_line(LINE_CSV)
    start_yaw = Path(line).heading(0)
    side = start_yaw + math.pi / 2
    start = (line[0].x_m + 5.0 * math.cos(side), line[0].y_m + 5.0 * math.sin(side), start_yaw)
    rows = drive(line, 15.0, params, start=start)
    assert rows[0][4]
    assert not rows[-1][4]
    assert max(r[3] for r in rows[-int(5.0 / DT_S) :]) < 0.5


def test_steers_toward_line(params):
    controller = PurePursuitController(params)
    assert controller.step(inputs_at(0.0, -1.0, 0.0, 3.0, straight())).steering_rad > 0.0
    controller.reset()
    assert controller.step(inputs_at(0.0, 1.0, 0.0, 3.0, straight())).steering_rad < 0.0


def test_line_speed_caps_setpoint(params):
    out = PurePursuitController(params).step(inputs_at(0.0, 0.0, 0.0, 3.0, straight(speed=2.0)))
    assert out.speed_setpoint_mps == 2.0


def test_stops_at_end_of_open_line(params):
    controller = PurePursuitController(params)
    out = controller.step(inputs_at(19.0, 0.05, 0.0, 6.0, straight(length=20, speed=6.0)))
    assert out.speed_setpoint_mps == pytest.approx(0.0, abs=0.1)
    assert out.steering_rad == 0.0
    out = controller.step(inputs_at(25.0, 0.0, 0.0, 6.0, straight(length=20, speed=6.0)))
    assert out.speed_setpoint_mps == pytest.approx(0.0, abs=0.1)


def test_short_open_lines_are_not_loops(params):
    assert not is_loop(straight(length=2), params["loop_gap_m"], params["loop_min_length_m"])
    hairpin = tuple(LinePoint(float(i), 0.0, 0.0, 5.0) for i in range(10)) + tuple(
        LinePoint(float(i), 2.0, 0.0, 5.0) for i in reversed(range(10))
    )
    assert not is_loop(hairpin, params["loop_gap_m"], params["loop_min_length_m"])
    assert is_loop(oval(), params["loop_gap_m"], params["loop_min_length_m"])


def test_shorter_line_does_not_crash(params):
    controller = PurePursuitController(params)
    controller.step(inputs_at(45.0, 0.0, 0.0, 3.0, straight(length=50)))
    out = controller.step(inputs_at(5.0, 0.0, 0.0, 3.0, straight(length=10)))
    assert out.speed_setpoint_mps >= 0.0


def test_picks_the_straight_heading_the_kart_s_way(params):
    # Kart on the return straight of a narrow oval, nearer the outbound straight's points
    line = oval(radius_m=1.5)
    controller = PurePursuitController(params)
    controller.step(inputs_at(20.0, 0.5, math.pi, 3.0, line))
    assert line[controller.line_index].y_m == pytest.approx(1.5)
