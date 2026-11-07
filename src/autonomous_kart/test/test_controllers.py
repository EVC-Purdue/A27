"""Checks every Pathfinder controller honours the shared interface, with no ROS.

https://github.com/EVC-Purdue/A27/issues/13
https://github.com/EVC-Purdue/A27/issues/36
https://github.com/EVC-Purdue/A27/issues/37
"""

import math

import pytest

from autonomous_kart.nodes.pathfinder.controllers import CONTROLLERS, make_controller
from autonomous_kart.nodes.pathfinder.controllers.base import (
    Controller,
    ControllerInputs,
    ControllerOutput,
    LinePoint,
    Track,
)


def straight_line_inputs(**overrides):
    fields = dict(
        t=0.0,
        x=0.0,
        y=0.0,
        yaw=0.0,
        vx=5.0,
        vy=0.0,
        yaw_rate=0.0,
        wheel_speed=5.0,
        steering_angle=0.0,
        steering_motor_current=0.0,
        line=tuple(LinePoint(float(i), 0.0, 0.0, 8.0) for i in range(50)),
        tracks=(),
    )
    fields.update(overrides)
    return ControllerInputs(**fields)


def test_registry_has_all_controllers():
    assert set(CONTROLLERS) == {"pure_pursuit", "mpcc", "rl"}


def test_unknown_controller_raises():
    with pytest.raises(ValueError):
        make_controller("mpc", {})


@pytest.mark.parametrize("name", sorted(CONTROLLERS))
@pytest.mark.parametrize(
    "inputs",
    [
        straight_line_inputs(),
        straight_line_inputs(vx=0.0, wheel_speed=0.0),
        straight_line_inputs(tracks=(Track(1, 1, 1.5, 10.0, 0.0, 3.0, 0.0),)),
    ],
    ids=["straight", "standstill", "kart_ahead"],
)
def test_step_returns_valid_output(name, inputs):
    controller = make_controller(name, {})
    assert isinstance(controller, Controller)
    out = controller.step(inputs)
    assert isinstance(out, ControllerOutput)
    assert math.isfinite(out.speed_setpoint) and math.isfinite(out.steering)
    assert 0.0 <= out.electrical_brake <= 1.0
    assert isinstance(out.mechanical_brake, bool)


@pytest.mark.parametrize("name", sorted(CONTROLLERS))
def test_reset_then_step(name):
    controller = make_controller(name, {})
    controller.step(straight_line_inputs())
    controller.reset()
    assert isinstance(controller.step(straight_line_inputs()), ControllerOutput)
