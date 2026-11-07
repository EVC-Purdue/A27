"""Controller registry; the pathfinder `controller` param picks one by name.

https://github.com/EVC-Purdue/A27/issues/13
"""
from autonomous_kart.nodes.pathfinder.controllers.base import Controller
from autonomous_kart.nodes.pathfinder.controllers.mpcc import MpccController
from autonomous_kart.nodes.pathfinder.controllers.pure_pursuit import PurePursuitController
from autonomous_kart.nodes.pathfinder.controllers.rl import RlController

CONTROLLERS = {c.name: c for c in (PurePursuitController, MpccController, RlController)}


def make_controller(name: str, params: dict) -> Controller:
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller '{name}', available: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](params)
