"""Rejoin: a Bezier curve from the kart back onto the racing line, used when the kart is far off it.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math

from autonomous_kart.nodes.pathfinder.controllers.base import LinePoint
from autonomous_kart.nodes.pathfinder.controllers.path import Path


class RejoinPlanner:
    def __init__(self, params: dict):
        self.merge_distance_m = params["rejoin_merge_distance_m"]
        self.min_radius_m = params["rejoin_min_radius_m"]
        self.retry_factor = params["rejoin_retry_factor"]
        self.handle_fraction = params["rejoin_handle_fraction"]
        self.min_handle_m = params["rejoin_min_handle_m"]
        self.min_gap_m = params["rejoin_min_gap_m"]
        self.points = params["rejoin_points"]
        self.start_speed_fraction = params["rejoin_start_speed_fraction"]

    def plan(self, line, line_index, x, y, yaw):
        # Returns the rejoin Path, or None when there is nothing ahead worth curving to
        curve, merge_index = self.curve_to_line(line, line_index, x, y, yaw, self.merge_distance_m)
        if curve is not None and self.too_tight(curve):
            farther_m = self.merge_distance_m * self.retry_factor
            curve, merge_index = self.curve_to_line(line, line_index, x, y, yaw, farther_m)
        if curve is None:
            return None
        return self.sample(curve, line.points[merge_index].target_speed_mps)

    def curve_to_line(self, line, line_index, x, y, yaw, merge_distance_m):
        merge_index = line.index_ahead(line_index, merge_distance_m)
        if merge_index == line_index:  # the kart is at the end of an open line
            return None, merge_index
        merge = line.points[merge_index]
        gap_m = math.hypot(merge.x_m - x, merge.y_m - y)
        if gap_m < self.min_gap_m:
            return None, merge_index

        # The curve leaves the kart along its heading and meets the line along the line's heading
        handle_m = max(gap_m * self.handle_fraction, self.min_handle_m)
        merge_heading = line.heading(merge_index)
        curve = Bezier(
            (x, y),
            (x + handle_m * math.cos(yaw), y + handle_m * math.sin(yaw)),
            (merge.x_m - handle_m * math.cos(merge_heading), merge.y_m - handle_m * math.sin(merge_heading)),
            (merge.x_m, merge.y_m),
        )
        return curve, merge_index

    def too_tight(self, curve):
        max_curvature = 1.0 / self.min_radius_m
        return any(abs(curve.curvature(k / self.points)) > max_curvature for k in range(self.points + 1))

    def sample(self, curve, merge_speed_mps):
        points = []
        for k in range(self.points + 1):
            t = k / self.points
            x, y = curve.point(t)
            # Slow at the start of the curve, up to the line's speed where it merges
            speed_fraction = self.start_speed_fraction + (1.0 - self.start_speed_fraction) * t**2
            points.append(LinePoint(x, y, curve.curvature(t), merge_speed_mps * speed_fraction))
        return Path(tuple(points))


class Bezier:
    # Cubic Bezier through four (x, y) control points, with t from 0 at the start to 1 at the end
    def __init__(self, p0, p1, p2, p3):
        self.x = (p0[0], p1[0], p2[0], p3[0])
        self.y = (p0[1], p1[1], p2[1], p3[1])

    def point(self, t):
        return self.position(self.x, t), self.position(self.y, t)

    def curvature(self, t):
        dx, dy = self.first_derivative(self.x, t), self.first_derivative(self.y, t)
        ddx, ddy = self.second_derivative(self.x, t), self.second_derivative(self.y, t)
        speed_cubed = (dx * dx + dy * dy) ** 1.5
        if speed_cubed < 1e-9:
            return 0.0
        return (dx * ddy - dy * ddx) / speed_cubed

    @staticmethod
    def position(c, t):
        u = 1.0 - t
        return u**3 * c[0] + 3 * u**2 * t * c[1] + 3 * u * t**2 * c[2] + t**3 * c[3]

    @staticmethod
    def first_derivative(c, t):
        u = 1.0 - t
        return 3 * (u**2 * (c[1] - c[0]) + 2 * u * t * (c[2] - c[1]) + t**2 * (c[3] - c[2]))

    @staticmethod
    def second_derivative(c, t):
        u = 1.0 - t
        return 6 * (u * (c[2] - 2 * c[1] + c[0]) + t * (c[3] - 2 * c[2] + c[1]))
