"""Pure pursuit controller with curvature-aware speed.

https://github.com/EVC-Purdue/A27/issues/13
"""

from autonomous_kart.nodes.pathfinder.controllers.base import (
    Controller,
    ControllerInputs,
    ControllerOutput,
)


class PurePursuitController(Controller):
    name = "pure_pursuit"

    def step(self, inputs: ControllerInputs) -> ControllerOutput:
        return ControllerOutput()
