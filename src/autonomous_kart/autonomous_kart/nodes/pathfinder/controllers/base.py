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
class ControllerOutput:
    speed_setpoint_mps: float = 0.0
    steering_rad: float = 0.0  # positive left
    electrical_brake: float = 0.0  # 0 to 1
    mechanical_brake: bool = False


class Controller(ABC):
    name: ClassVar[str]

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def step(self, inputs: ControllerInputs) -> ControllerOutput: ...

    def reset(self) -> None:  # noqa: B027
        pass
