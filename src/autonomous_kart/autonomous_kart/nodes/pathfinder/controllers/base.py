"""Interface shared by every Pathfinder controller, free of ROS so tests build inputs directly.

https://github.com/EVC-Purdue/A27/issues/13
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LinePoint:
    x: float  # m
    y: float  # m
    curvature: float  # 1/m
    target_speed: float  # m/s


@dataclass(frozen=True)
class Track:
    id: int
    object_class: int
    footprint_size: float  # m
    x: float  # m
    y: float  # m
    vx: float  # m/s
    vy: float  # m/s


@dataclass(frozen=True)
class ControllerInputs:
    t: float  # s
    x: float  # m
    y: float  # m
    yaw: float  # rad
    vx: float  # m/s, body frame
    vy: float  # m/s, body frame
    yaw_rate: float  # rad/s
    wheel_speed: float  # m/s
    steering_angle: float  # rad
    steering_motor_current: float  # A
    line: tuple[LinePoint, ...] = ()
    tracks: tuple[Track, ...] = ()


@dataclass(frozen=True)
class ControllerOutput:
    speed_setpoint: float = 0.0  # m/s
    steering: float = 0.0  # rad, positive left
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
