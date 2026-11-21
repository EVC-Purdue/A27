"""Cubic Bezier path from the kart back onto the racing line, as A26 pure pursuit used when the kart was far off it.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math

from autonomous_kart.nodes.pathfinder.controllers.base import LinePoint
from autonomous_kart.nodes.pathfinder.controllers.path import heading_at, index_ahead


def rejoin_path(line, closest, closed, x, y, yaw, params):
    # params is the pure_pursuit block of pathfinder.yaml
    merge_lookahead_m = params["rejoin_merge_lookahead_m"]
    min_radius_m = params["rejoin_min_radius_m"]
    samples = params["rejoin_samples"]
    start_speed_fraction = params["rejoin_start_speed_fraction"]

    start = (x, y)
    # A curve tighter than the kart can turn gets one retry with the merge point farther ahead
    for lookahead_m in (merge_lookahead_m, params["rejoin_retry_lookahead_factor"] * merge_lookahead_m):
        merge = index_ahead(line, closest, lookahead_m, closed)
        if merge == closest:  # past the end of an open line, nothing is left to merge onto
            return None
        end = (line[merge].x_m, line[merge].y_m)
        gap_m = math.dist(start, end)
        if gap_m < params["rejoin_min_gap_m"]:
            return None
        handle_m = max(gap_m * params["rejoin_handle_fraction"], params["rejoin_min_handle_m"])
        controls = bezier_controls(start, yaw, end, heading_at(line, merge), handle_m)
        tightest = max(abs(bezier_curvature(controls, k / samples)) for k in range(samples + 1))
        if tightest <= 1.0 / min_radius_m:
            break

    # Starts slow and ramps up to the merge point's speed
    merge_speed = line[merge].target_speed_mps
    points = []
    for k in range(samples + 1):
        t = k / samples
        px, py = bezier_point(controls, t)
        speed = merge_speed * (start_speed_fraction + (1.0 - start_speed_fraction) * t**2)
        points.append(LinePoint(px, py, bezier_curvature(controls, t), speed))
    return tuple(points)


def bezier_controls(start, start_yaw, end, end_yaw, handle_m):
    # Leaves the kart along its heading and arrives along the line's heading
    p1 = (start[0] + handle_m * math.cos(start_yaw), start[1] + handle_m * math.sin(start_yaw))
    p2 = (end[0] - handle_m * math.cos(end_yaw), end[1] - handle_m * math.sin(end_yaw))
    return start, p1, p2, end


def bezier_point(controls, t):
    p0, p1, p2, p3 = controls
    u = 1.0 - t
    weights = (u**3, 3 * u**2 * t, 3 * u * t**2, t**3)
    return (
        sum(w * p[0] for w, p in zip(weights, controls, strict=True)),
        sum(w * p[1] for w, p in zip(weights, controls, strict=True)),
    )


def bezier_curvature(controls, t):
    p0, p1, p2, p3 = controls
    u = 1.0 - t
    dx = 3 * (u**2 * (p1[0] - p0[0]) + 2 * u * t * (p2[0] - p1[0]) + t**2 * (p3[0] - p2[0]))
    dy = 3 * (u**2 * (p1[1] - p0[1]) + 2 * u * t * (p2[1] - p1[1]) + t**2 * (p3[1] - p2[1]))
    ddx = 6 * (u * (p2[0] - 2 * p1[0] + p0[0]) + t * (p3[0] - 2 * p2[0] + p1[0]))
    ddy = 6 * (u * (p2[1] - 2 * p1[1] + p0[1]) + t * (p3[1] - 2 * p2[1] + p1[1]))
    speed_cubed = (dx * dx + dy * dy) ** 1.5
    return 0.0 if speed_cubed < 1e-9 else (dx * ddy - dy * ddx) / speed_cubed
