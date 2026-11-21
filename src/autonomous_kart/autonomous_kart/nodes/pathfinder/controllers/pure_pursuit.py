"""Pure pursuit controller with curvature-aware speed, and a Bezier rejoin when the kart is far off the line.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math

from autonomous_kart.nodes.pathfinder.controllers.base import (
    Controller,
    ControllerInputs,
    ControllerOutput,
)
from autonomous_kart.nodes.pathfinder.controllers.path import (
    clamp,
    distance,
    distance_to_segment,
    heading_at,
    index_ahead,
    is_closed,
    next_index,
    previous_index,
    wrap_angle,
)
from autonomous_kart.nodes.pathfinder.controllers.rejoin import rejoin_path


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
        self.closed_min_length_m = params["closed_min_length_m"]
        self.search_max_heading_error_rad = params["search_max_heading_error_rad"]
        self.min_target_m = params["min_target_m"]
        self.rejoin_cte_activate_m = params["rejoin_cte_activate_m"]
        self.rejoin_cte_deactivate_m = params["rejoin_cte_deactivate_m"]
        self.rejoin_end_points = params["rejoin_end_points"]
        self.reset()

    def reset(self) -> None:
        self.line_length = 0  # points in the line the state below refers to
        self.closed = False  # whether the line is a loop
        self.closest = None  # index of the racing line point nearest the kart
        self.rejoin = None  # Bezier path back to the racing line, while the kart is far off it
        self.rejoin_closest = 0

    def step(self, inputs: ControllerInputs) -> ControllerOutput:
        line = inputs.line
        if len(line) < 2:
            return ControllerOutput()
        if len(line) != self.line_length:
            self.reset()
            self.line_length = len(line)
            self.closed = is_closed(line, self.closed_gap_m, self.closed_min_length_m)

        self.closest = self.find_closest_point(line, self.closest, self.closed, inputs)
        self.update_rejoin(line, inputs)

        # Follow the rejoin path while there is one, else the racing line
        if self.rejoin is None:
            path, closest, closed = line, self.closest, self.closed
        else:
            path, closest, closed = self.rejoin, self.rejoin_closest, False

        lookahead_m = clamp(inputs.vx_mps * self.lookahead_time_s, self.min_lookahead_m, self.max_lookahead_m)
        target = path[index_ahead(path, closest, lookahead_m, closed)]
        steering_rad = self.steer_toward(target, inputs)
        # The rejoin ends on the racing line, so only the end of an open racing line is a place to stop
        speed_mps = self.speed_setpoint(path, closest, closed, stop_at_end=self.rejoin is None)
        return ControllerOutput(speed_setpoint_mps=speed_mps, steering_rad=steering_rad)

    def find_closest_point(self, path, previous, closed, inputs):
        def distance_to_kart(i):
            return math.hypot(path[i].x_m - inputs.x_m, path[i].y_m - inputs.y_m)

        if previous is None:
            return self.search_whole_path(path, inputs)

        # Search a short window ahead of last tick's point, so the kart never jumps to a nearby part of the track
        window = []
        i = previous
        while i is not None and len(window) < self.search_window:
            window.append(i)
            i = next_index(i, len(path), closed)
        closest = min(window, key=distance_to_kart)

        # Far from the window means the kart lost the line, so search all of it
        if distance_to_kart(closest) > self.resync_m:
            closest = self.search_whole_path(path, inputs)
        return closest

    def search_whole_path(self, path, inputs):
        # Only points heading the way the kart faces, so a nearby straight run the other way is never picked
        facing = [
            i
            for i in range(len(path))
            if abs(wrap_angle(heading_at(path, i) - inputs.yaw_rad)) < self.search_max_heading_error_rad
        ]
        return min(
            facing or range(len(path)),
            key=lambda i: math.hypot(path[i].x_m - inputs.x_m, path[i].y_m - inputs.y_m),
        )

    def cross_track_error(self, line, inputs):
        n = len(line)
        neighbours = [previous_index(self.closest, n, self.closed), next_index(self.closest, n, self.closed)]
        return min(
            distance_to_segment(inputs.x_m, inputs.y_m, line[self.closest], line[j])
            for j in neighbours
            if j is not None
        )

    def update_rejoin(self, line, inputs):
        cte_m = self.cross_track_error(line, inputs)
        if self.rejoin is None:
            if cte_m > self.rejoin_cte_activate_m:
                self.rejoin = rejoin_path(
                    line, self.closest, self.closed, inputs.x_m, inputs.y_m, inputs.yaw_rad, self.params
                )
                self.rejoin_closest = 0
            return

        self.rejoin_closest = self.find_closest_point(self.rejoin, self.rejoin_closest, False, inputs)
        near_end = self.rejoin_closest >= len(self.rejoin) - self.rejoin_end_points
        if cte_m < self.rejoin_cte_deactivate_m or near_end:
            self.rejoin = None

    def steer_toward(self, target, inputs):
        dx = target.x_m - inputs.x_m
        dy = target.y_m - inputs.y_m
        distance_to_target = math.hypot(dx, dy)
        if distance_to_target < self.min_target_m:
            return 0.0
        angle_to_target = math.atan2(dy, dx) - inputs.yaw_rad

        # The arc through the target that starts tangent to the kart's heading
        curvature = 2.0 * math.sin(angle_to_target) / distance_to_target
        steering = math.atan(self.wheelbase_m * curvature)
        return clamp(steering, -self.max_steering_rad, self.max_steering_rad)

    def speed_setpoint(self, path, start, closed, stop_at_end):
        # Fastest speed from which the kart can still brake down to every point's allowed speed in the preview
        setpoint = self.speed_cap_mps
        travelled_m = 0.0
        i = start
        while True:
            setpoint = min(setpoint, self.braking_speed(self.allowed_speed(path[i]), travelled_m))
            j = next_index(i, len(path), closed)
            if j is None:
                if stop_at_end:
                    setpoint = min(setpoint, self.braking_speed(0.0, travelled_m))
                return setpoint
            if j == start or travelled_m >= self.preview_m:
                return setpoint
            travelled_m += distance(path[i], path[j])
            i = j

    def allowed_speed(self, point):
        corner_speed = math.sqrt(self.lateral_accel_max_mps2 / max(abs(point.curvature_radpm), 1e-6))
        return min(point.target_speed_mps, corner_speed, self.speed_cap_mps)

    def braking_speed(self, speed_mps, distance_m):
        # Speed from which the kart slows to speed_mps within distance_m at the brake deceleration
        return math.sqrt(speed_mps**2 + 2.0 * self.brake_decel_mps2 * distance_m)
