"""Drives pure pursuit around an oval on a kinematic bicycle with noisy, delayed pose, with no ROS.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math
import random

from autonomous_kart.nodes.pathfinder.controllers.base import ControllerInputs, LinePoint
from autonomous_kart.nodes.pathfinder.controllers.pure_pursuit import PurePursuitController

DT_S = 0.02
WHEELBASE_M = 1.05
POSE_SIGMA_M = 0.05
YAW_SIGMA_RAD = 0.01
LATENCY_TICKS = 3
SPEED_TAU_S = 0.3
STEER_TAU_S = 0.05


def oval(straight_m=40.0, radius_m=10.0, spacing_m=0.5):
    # Two straights joined by half circles, counter-clockwise, as LinePoints with a 10 m/s line speed
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


def cte(line, x, y):
    # Distance to the nearest segment of the closed line
    best = math.inf
    for a, b in zip(line, line[1:] + line[:1], strict=True):
        ux, uy = b.x_m - a.x_m, b.y_m - a.y_m
        t = min(max(((x - a.x_m) * ux + (y - a.y_m) * uy) / (ux * ux + uy * uy), 0.0), 1.0)
        best = min(best, math.hypot(a.x_m + t * ux - x, a.y_m + t * uy - y))
    return best


def drive(line, seconds, params, seed=0):
    rng = random.Random(seed)
    controller = PurePursuitController(params)
    x, y, yaw, v, steer = line[0].x_m, line[0].y_m, 0.0, 0.0, 0.0
    poses = [(x, y, yaw, v)] * (LATENCY_TICKS + 1)
    rows = []
    for k in range(int(seconds / DT_S)):
        px, py, pyaw, pv = poses[-1 - LATENCY_TICKS]
        out = controller.step(
            ControllerInputs(
                time_s=k * DT_S,
                x_m=px + rng.gauss(0.0, POSE_SIGMA_M),
                y_m=py + rng.gauss(0.0, POSE_SIGMA_M),
                yaw_rad=pyaw + rng.gauss(0.0, YAW_SIGMA_RAD),
                vx_mps=pv,
                vy_mps=0.0,
                yaw_rate_radps=0.0,
                wheel_speed_mps=pv,
                steering_angle_rad=steer,
                steering_motor_current_a=0.0,
                line=line,
            )
        )
        v += (out.speed_setpoint_mps - v) * DT_S / SPEED_TAU_S
        steer += (out.steering_rad - steer) * DT_S / STEER_TAU_S
        x += v * math.cos(yaw) * DT_S
        y += v * math.sin(yaw) * DT_S
        yaw += v / WHEELBASE_M * math.tan(steer) * DT_S
        poses.append((x, y, yaw, v))
        rows.append((x, y, v, cte(line, x, y)))
    return rows


def percentile(values, q):
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)]


def test_laps_oval_within_tolerance(pathfinder_params):
    line = oval()
    rows = drive(line, 40.0, pathfinder_params["pure_pursuit"])
    errors = [r[3] for r in rows[int(5.0 / DT_S) :]]  # after the launch from standstill
    assert percentile(errors, 0.95) < 0.3
    assert max(errors) < 0.6
    distance_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rows, rows[1:], strict=False))
    lap_m = 2 * 40.0 + 2 * math.pi * 10.0
    assert distance_m > 2 * lap_m


def test_slows_for_corners(pathfinder_params):
    line = oval()
    rows = drive(line, 40.0, {**pathfinder_params["pure_pursuit"], "speed_cap_mps": 12.0})[int(10.0 / DT_S) :]
    corner = [v for x, _, v, _ in rows if x > 45.0]
    straight = [v for x, _, v, _ in rows if 15.0 < x < 25.0]
    assert max(corner) < math.sqrt(pathfinder_params["pure_pursuit"]["lateral_accel_max_mps2"] * 10.0) + 0.3
    assert max(straight) > max(corner) + 0.5


def test_steers_toward_line(pathfinder_params):
    line = tuple(LinePoint(float(i), 0.0, 0.0, 5.0) for i in range(50))
    controller = PurePursuitController(pathfinder_params["pure_pursuit"])
    inputs = dict(
        time_s=0.0,
        x_m=0.0,
        yaw_rad=0.0,
        vx_mps=3.0,
        vy_mps=0.0,
        yaw_rate_radps=0.0,
        wheel_speed_mps=3.0,
        steering_angle_rad=0.0,
        steering_motor_current_a=0.0,
        line=line,
    )
    assert controller.step(ControllerInputs(y_m=-1.0, **inputs)).steering_rad > 0.0
    controller.reset()
    assert controller.step(ControllerInputs(y_m=1.0, **inputs)).steering_rad < 0.0


def test_line_speed_caps_setpoint(pathfinder_params):
    line = tuple(LinePoint(float(i), 0.0, 0.0, 2.0) for i in range(50))
    inputs = ControllerInputs(0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, line=line)
    assert PurePursuitController(pathfinder_params["pure_pursuit"]).step(inputs).speed_setpoint_mps == 2.0
