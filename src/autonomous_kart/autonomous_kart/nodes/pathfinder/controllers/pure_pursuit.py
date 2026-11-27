"""Pure pursuit: steers at a point ahead on the racing line and slows for the corners coming up.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math
from dataclasses import replace

from autonomous_kart.nodes.pathfinder.controllers.base import (
    Controller,
    ControllerInputs,
    ControllerOutput,
    Plan,
)
from autonomous_kart.nodes.pathfinder.controllers.path import (
    Path,
    angle_between,
    clamp,
    distance,
    is_loop,
)
from autonomous_kart.nodes.pathfinder.controllers.rejoin import RejoinPlanner


class PurePursuitController(Controller):
    name = "pure_pursuit"

    def __init__(self, params: dict):
        super().__init__(params)
        # Steering
        self.wheelbase_m = params["wheelbase_m"]
        self.max_steering_rad = params["max_steering_rad"]
        self.lookahead_time_s = params["lookahead_time_s"]
        self.min_lookahead_m = params["min_lookahead_m"]
        self.max_lookahead_m = params["max_lookahead_m"]
        self.min_target_m = params["min_target_m"]
        self.plan_distance_m = params["plan_distance_m"]
        # Speed
        self.speed_cap_mps = params["speed_cap_mps"]
        self.lateral_accel_max_mps2 = params["lateral_accel_max_mps2"]
        self.brake_decel_mps2 = params["brake_decel_mps2"]
        # Finding the kart on the line
        self.search_window = params["search_window"]
        self.resync_m = params["resync_m"]
        self.search_max_heading_error_rad = params["search_max_heading_error_rad"]
        self.loop_gap_m = params["loop_gap_m"]
        self.loop_min_length_m = params["loop_min_length_m"]
        # Rejoin
        self.rejoin_start_cte_m = params["rejoin_start_cte_m"]
        self.rejoin_stop_cte_m = params["rejoin_stop_cte_m"]
        self.rejoin_end_points = params["rejoin_end_points"]
        self.rejoin_planner = RejoinPlanner(params)
        self.reset()

    def reset(self) -> None:
        self.line = None  # the racing line
        self.line_index = None  # index of the line point nearest the kart
        self.rejoin = None  # curve back to the line, set only while the kart is far off it
        self.rejoin_index = 0  # index of the rejoin point nearest the kart

    def step(self, inputs: ControllerInputs) -> ControllerOutput:
        if len(inputs.line) < 2:
            return ControllerOutput()
        if self.line is None or inputs.line != self.line.points:
            self.reset()
            self.line = Path(inputs.line, is_loop(inputs.line, self.loop_gap_m, self.loop_min_length_m))
            self.set_speeds(self.line, stop_at_end=True)

        self.line_index = self.nearest_index(self.line, self.line_index, inputs)
        self.update_rejoin(inputs)

        # Follow the rejoin curve while there is one, otherwise the racing line
        if self.rejoin is not None:
            path, index = self.rejoin, self.rejoin_index
        else:
            path, index = self.line, self.line_index
        return ControllerOutput(
            speed_setpoint_mps=path.speeds_mps[index],
            steering_rad=self.steering(path, index, inputs),
            plan=self.plan(path, index),
        )

    def plan(self, path, index):
        # The path ahead of the kart, with the speed the kart is driven at each point
        indices = path.indices_ahead(index, self.plan_distance_m)
        return Plan(points=tuple(replace(path.points[i], target_speed_mps=path.speeds_mps[i]) for i in indices))

    def steering(self, path, index, inputs):
        # Aim at a point ahead on the path, farther ahead at higher speed
        lookahead_m = clamp(inputs.vx_mps * self.lookahead_time_s, self.min_lookahead_m, self.max_lookahead_m)
        target = path.points[path.index_ahead(index, lookahead_m)]
        dx = target.x_m - inputs.x_m
        dy = target.y_m - inputs.y_m
        distance_to_target = math.hypot(dx, dy)
        if distance_to_target < self.min_target_m:
            return 0.0

        # The kart reaches the target by driving a circular arc of this curvature
        angle_to_target = math.atan2(dy, dx) - inputs.yaw_rad
        curvature = 2.0 * math.sin(angle_to_target) / distance_to_target
        # Bicycle model: the steering angle that makes the kart turn at that curvature
        steering_rad = math.atan(self.wheelbase_m * curvature)
        return clamp(steering_rad, -self.max_steering_rad, self.max_steering_rad)

    def set_speeds(self, path, stop_at_end):
        # Start from the fastest each point can be taken on its own
        speeds = [self.corner_speed(point) for point in path.points]
        if stop_at_end and not path.is_loop:
            speeds[-1] = 0.0

        # Walk backwards so each point is slow enough to brake down to the point after it
        laps = 2 if path.is_loop else 1  # the second lap carries braking across the start of a loop
        for _ in range(laps):
            for i in reversed(range(len(speeds))):
                j = path.next(i)
                if j is not None:
                    gap_m = distance(path.points[i], path.points[j])
                    speeds[i] = min(speeds[i], math.sqrt(speeds[j] ** 2 + 2.0 * self.brake_decel_mps2 * gap_m))
        path.speeds_mps = speeds

    def corner_speed(self, point):
        speed_mps = min(point.target_speed_mps, self.speed_cap_mps)
        curvature = abs(point.curvature_radpm)
        if curvature > 0.0:
            # Lateral acceleration is speed^2 * curvature
            speed_mps = min(speed_mps, math.sqrt(self.lateral_accel_max_mps2 / curvature))
        return speed_mps

    def nearest_index(self, path, last_index, inputs):
        x, y = inputs.x_m, inputs.y_m
        # Normally the kart is a few points past where it was last tick
        if last_index is not None:
            index = path.nearest(path.indices_from(last_index, self.search_window), x, y)
            if path.distance_to(index, x, y) < self.resync_m:
                return index

        # First tick, or the kart is nowhere near there: search the whole path
        # Points heading the other way are skipped, so the opposite straight is never picked
        facing_kart = [
            i
            for i in range(len(path.points))
            if angle_between(path.heading(i), inputs.yaw_rad) < self.search_max_heading_error_rad
        ]
        return path.nearest(facing_kart or range(len(path.points)), x, y)

    def update_rejoin(self, inputs):
        off_line_m = self.line.cross_track_error(self.line_index, inputs.x_m, inputs.y_m)

        if self.rejoin is None:
            if off_line_m > self.rejoin_start_cte_m:
                self.start_rejoin(inputs)
            return

        # Drop the curve once the kart is back near the line or has driven to the end of the curve
        self.rejoin_index = self.nearest_index(self.rejoin, self.rejoin_index, inputs)
        at_end_of_curve = self.rejoin_index >= len(self.rejoin.points) - self.rejoin_end_points
        if off_line_m < self.rejoin_stop_cte_m or at_end_of_curve:
            self.rejoin = None

    def start_rejoin(self, inputs):
        self.rejoin = self.rejoin_planner.plan(self.line, self.line_index, inputs.x_m, inputs.y_m, inputs.yaw_rad)
        self.rejoin_index = 0
        if self.rejoin is not None:
            # The curve ends on the racing line, so the kart keeps moving at its end
            self.set_speeds(self.rejoin, stop_at_end=False)
