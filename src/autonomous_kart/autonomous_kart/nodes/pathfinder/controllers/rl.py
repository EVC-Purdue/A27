"""RL policy blended with pure pursuit.

https://github.com/EVC-Purdue/A27/issues/36
"""
from autonomous_kart.nodes.pathfinder.controllers.base import (
    Controller,
    ControllerInputs,
    ControllerOutput,
)


class RlController(Controller):
    name = "rl"

    def step(self, inputs: ControllerInputs) -> ControllerOutput:
        return ControllerOutput()
