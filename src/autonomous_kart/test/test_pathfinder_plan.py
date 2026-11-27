"""Checks pathfinder publishes the controller's plan on every AUTONOMOUS tick.

https://github.com/EVC-Purdue/A27/issues/14
"""

import os

import pytest

rclpy = pytest.importorskip("rclpy")

from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from autonomous_kart.nodes.pathfinder.controllers import make_controller  # noqa: E402
from autonomous_kart.nodes.pathfinder.controllers.base import LinePoint, Plan  # noqa: E402
from autonomous_kart.nodes.pathfinder.pathfinder_node import PathfinderNode, to_plan_msg  # noqa: E402

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
    node.plan_pub = Recorder()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_plan_published_in_autonomous(node):
    start = node.line_points[0]
    odom = Odometry()
    odom.pose.pose.position.x = start.x
    odom.pose.pose.position.y = start.y
    node.on_odom(odom)
    node.tick()
    assert node.plan_pub.msgs == []
    node.on_state(String(data="AUTONOMOUS"))
    node.tick()
    plan = node.plan_pub.msgs[-1]
    assert (plan.points[0].x, plan.points[0].y) == (start.x, start.y)
    assert len(plan.points) > 1
    assert plan.cost_terms == []


def test_plan_msg_carries_costs():
    plan = Plan(
        points=(LinePoint(0.0, 0.0, 0.0, 1.0), LinePoint(1.0, 0.0, 0.0, 1.0), LinePoint(2.0, 0.0, 0.0, 1.0)),
        cost_terms=("d", "speed"),
        step_costs=((1.0, 2.0), (3.0, 1.5)),
        best_cost=7.5,
        cost_field_start_m=-0.1,
        cost_field_step_m=0.1,
        cost_field=(4.0, 0.0, 4.0),
    )
    msg = to_plan_msg(Odometry().header, plan)
    assert [p.x for p in msg.points] == [0.0, 1.0, 2.0]
    assert msg.cost_terms == ["d", "speed"]
    assert list(msg.step_costs) == [1.0, 2.0, 3.0, 1.5]
    assert msg.best_cost == 7.5
    assert (msg.cost_field_start, msg.cost_field_step, list(msg.cost_field)) == (-0.1, 0.1, [4.0, 0.0, 4.0])


def test_empty_plan_not_published(node):
    node.controller = make_controller("mpcc", {})
    node.on_odom(Odometry())
    node.on_state(String(data="AUTONOMOUS"))
    node.tick()
    assert node.plan_pub.msgs == []
