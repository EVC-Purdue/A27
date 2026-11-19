"""Checks pathfinder forwards fresh MANUAL commands and stops the kart when they go stale.

https://github.com/EVC-Purdue/A27/issues/19
"""

import os

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.duration import Duration  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from autonomous_kart.nodes.pathfinder.pathfinder_node import PathfinderNode  # noqa: E402
from kart_interfaces.msg import DriveCommand  # noqa: E402

PARAMS = os.path.join(os.path.dirname(__file__), "..", "autonomous_kart", "params")


class Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture
def node():
    files = [os.path.join(PARAMS, f) for f in ("system.yaml", "pathfinder.yaml")]
    rclpy.init(args=["--ros-args", "--params-file", files[0], "--params-file", files[1]])
    node = PathfinderNode()
    node.cmd_pub = Recorder()
    node.on_state(String(data="MANUAL"))
    yield node
    node.destroy_node()
    rclpy.shutdown()


def speed_after_tick(node):
    node.tick()
    return node.cmd_pub.msgs[-1].speed_setpoint


def test_fresh_command_forwarded(node):
    node.on_manual(DriveCommand(speed_setpoint=2.0, steering=0.1))
    node.tick()
    cmd = node.cmd_pub.msgs[-1]
    assert (cmd.speed_setpoint, cmd.steering) == (2.0, 0.1)


def test_stale_command_stops(node):
    node.on_manual(DriveCommand(speed_setpoint=2.0))
    node.manual_time = node.get_clock().now() - Duration(seconds=node.manual_timeout_s + 0.1)
    assert speed_after_tick(node) == 0.0


def test_state_change_drops_command(node):
    node.on_manual(DriveCommand(speed_setpoint=2.0))
    node.on_state(String(data="IDLE"))
    assert speed_after_tick(node) == 0.0
    node.on_state(String(data="MANUAL"))
    assert speed_after_tick(node) == 0.0
