"""Checks pathfinder loads a racing line CSV and publishes the sim lockstep heartbeat.

https://github.com/EVC-Purdue/A27/issues/13
https://github.com/EVC-Purdue/A27/issues/14
"""

import os

import pytest

from autonomous_kart.nodes.pathfinder.line import load_line

PARAMS = os.path.join(os.path.dirname(__file__), "..", "autonomous_kart", "params")
CSV = """s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2
0.0,1.0,2.0,0.0,0.01,14.0,0.0
1.0,2.0,2.5,0.1,0.02,13.5,0.0
"""


@pytest.fixture
def line_csv(tmp_path):
    path = tmp_path / "line.csv"
    path.write_text(CSV)
    return str(path)


def test_load_line(line_csv):
    points = load_line(line_csv)
    assert [(p.x_m, p.y_m, p.curvature_radpm, p.target_speed_mps) for p in points] == [
        (1.0, 2.0, 0.01, 14.0),
        (2.0, 2.5, 0.02, 13.5),
    ]


class Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


def test_node_publishes_line_and_heartbeat(line_csv, tmp_path):
    rclpy = pytest.importorskip("rclpy")
    from autonomous_kart.nodes.pathfinder.pathfinder_node import PathfinderNode

    episode = tmp_path / "episode.yaml"
    episode.write_text(f"pathfinder_node:\n  ros__parameters:\n    line_path: {line_csv}\n")
    files = [os.path.join(PARAMS, f) for f in ("system.yaml", "pathfinder.yaml")] + [str(episode)]
    rclpy.init(args=["--ros-args"] + [a for f in files for a in ("--params-file", f)])
    node = PathfinderNode()
    try:
        node.line_pub, node.tick_pub = Recorder(), Recorder()
        node.tick()
        node.tick()
        assert [(p.x, p.target_speed) for p in node.line_pub.msgs[-1].points] == [(1.0, 14.0), (2.0, 13.5)]
        assert [m.frame_id for m in node.tick_pub.msgs] == ["1", "2"]
    finally:
        node.destroy_node()
        rclpy.shutdown()
