"""MPCC controller.

https://github.com/EVC-Purdue/A27/issues/37
"""

from autonomous_kart.nodes.pathfinder.controllers.base import (
    Controller,
    ControllerInputs,
    ControllerOutput,
)


class MpccController(Controller):
    name = "mpcc"

    def step(self, inputs: ControllerInputs) -> ControllerOutput:
        return ControllerOutput()
