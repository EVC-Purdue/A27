"""Pure pursuit controller with curvature-aware speed.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math

from autonomous_kart.nodes.pathfinder.controllers.base import (
    Controller,
    ControllerInputs,
    ControllerOutput,
)


def distance(a, b):
    return math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)


def clamp(value, low, high):
    return max(low, min(value, high))


class PurePursuitController(Controller):
    name = "pure_pursuit"

    def __init__(self, params: dict):
        super().__init__(params)
        self.wheelbase_m = params["wheelbase_m"]
        self.max_steering_rad = params["max_steering_rad"]
        self.lookahead_time_s = params["lookahead_time_s"]
        self.min_lookahead_m = params["min_lookahead_m"]
        self.max_lookahead_m = params["max_lookahead_m"]
        self.lateral_accel_max_mps2 = params["lateral_accel_max_mps2"]
        self.brake_decel_mps2 = params["brake_decel_mps2"]
        self.speed_cap_mps = params["speed_cap_mps"]
        self.preview_m = params["preview_m"]
        self.search_window = params["search_window"]
        self.resync_m = params["resync_m"]
        self.closed_gap_m = params["closed_gap_m"]

        self.closest = None  # index of the line point nearest the kart, kept between ticks
        self.closed = False  # whether the line is a loop

    def reset(self) -> None:
        self.closest = None

    def step(self, inputs: ControllerInputs) -> ControllerOutput:
        line = inputs.line
        if len(line) < 2:
            return ControllerOutput()
        self.closed = distance(line[0], line[-1]) < self.closed_gap_m

        self.closest = self.find_closest_point(line, inputs)
        lookahead_m = clamp(inputs.vx_mps * self.lookahead_time_s, self.min_lookahead_m, self.max_lookahead_m)
        target = self.point_ahead(line, self.closest, lookahead_m)

        steering_rad = self.steer_toward(target, inputs)
        speed_mps = self.speed_setpoint(line, self.closest)
        return ControllerOutput(speed_setpoint_mps=speed_mps, steering_rad=steering_rad)

    def find_closest_point(self, line, inputs):
        def distance_to_kart(i):
            return math.hypot(line[i].x_m - inputs.x_m, line[i].y_m - inputs.y_m)

        if self.closest is None:
            return min(range(len(line)), key=distance_to_kart)

        # Search a short window ahead of last tick's point, so the kart never jumps to a nearby part of the track
        window = []
        i = self.closest
        while i is not None and len(window) < self.search_window:
            window.append(i)
            i = self.next_index(line, i)
        closest = min(window, key=distance_to_kart)

        # Far from the window means the kart lost the line, so search all of it
        if distance_to_kart(closest) > self.resync_m:
            closest = min(range(len(line)), key=distance_to_kart)
        return closest

    def point_ahead(self, line, start, distance_m):
        travelled_m = 0.0
        i = start
        while travelled_m < distance_m:
            j = self.next_index(line, i)
            if j is None or j == start:
                break
            travelled_m += distance(line[i], line[j])
            i = j
        return line[i]

    def steer_toward(self, target, inputs):
        dx = target.x_m - inputs.x_m
        dy = target.y_m - inputs.y_m
        angle_to_target = math.atan2(dy, dx) - inputs.yaw_rad
        distance_to_target = max(math.hypot(dx, dy), 1e-3)

        # The arc through the target that starts tangent to the kart's heading
        curvature = 2.0 * math.sin(angle_to_target) / distance_to_target
        steering = math.atan(self.wheelbase_m * curvature)
        return clamp(steering, -self.max_steering_rad, self.max_steering_rad)

    def speed_setpoint(self, line, start):
        # Fastest speed from which the kart can still brake down to every point's allowed speed in the preview
        setpoint = self.speed_cap_mps
        travelled_m = 0.0
        i = start
        while True:
            point = line[i]
            corner_speed = math.sqrt(self.lateral_accel_max_mps2 / max(abs(point.curvature_radpm), 1e-6))
            allowed = min(point.target_speed_mps, corner_speed, self.speed_cap_mps)
            setpoint = min(setpoint, math.sqrt(allowed**2 + 2.0 * self.brake_decel_mps2 * travelled_m))

            j = self.next_index(line, i)
            if j is None or j == start or travelled_m >= self.preview_m:
                return setpoint
            travelled_m += distance(line[i], line[j])
            i = j

    def next_index(self, line, i):
        # The point after i; a closed line wraps to the start, an open one ends with None
        if i + 1 < len(line):
            return i + 1
        return 0 if self.closed else None
