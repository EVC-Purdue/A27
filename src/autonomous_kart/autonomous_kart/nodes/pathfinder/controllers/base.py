"""Interface shared by every Pathfinder controller, free of ROS so tests build inputs directly.

https://github.com/EVC-Purdue/A27/issues/13
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LinePoint:
    x_m: float
    y_m: float
    curvature_radpm: float
    target_speed_mps: float


@dataclass(frozen=True)
class Track:
    id: int
    object_class: int
    footprint_size_m: float
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float


@dataclass(frozen=True)
class ControllerInputs:
    time_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    vx_mps: float  # body frame
    vy_mps: float  # body frame
    yaw_rate_radps: float
    wheel_speed_mps: float
    steering_angle_rad: float
    steering_motor_current_a: float
    line: tuple[LinePoint, ...] = ()
    tracks: tuple[Track, ...] = ()


@dataclass(frozen=True)
class Plan:
    points: tuple[LinePoint, ...] = ()  # path the controller is following ahead of the kart
    cost_terms: tuple[str, ...] = ()  # names of the cost terms, empty for a controller without costs
    step_costs: tuple[tuple[float, ...], ...] = ()  # one row per step between points, one value per cost term
    best_cost: float = 0.0  # total cost of the plan
    cost_field_start_m: float = 0.0  # offset to the side of the line of the first cost_field value
    cost_field_step_m: float = 0.0  # offset between cost_field values
    cost_field: tuple[float, ...] = ()  # position cost at each offset


@dataclass(frozen=True)
class ControllerOutput:
    speed_setpoint_mps: float = 0.0
    steering_rad: float = 0.0  # positive left
    electrical_brake: float = 0.0  # 0 to 1
    mechanical_brake: bool = False
    plan: Plan = Plan()


class Controller(ABC):
    name: ClassVar[str]

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def step(self, inputs: ControllerInputs) -> ControllerOutput: ...

    def reset(self) -> None:  # noqa: B027
        pass
