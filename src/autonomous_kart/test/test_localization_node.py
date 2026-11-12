"""Checks the localization node: init from a heading fix, re-init after repeated RTK rejects, background GRU training.

https://github.com/EVC-Purdue/A27/issues/28
"""

import math
import os

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from sensor_msgs.msg import Imu  # noqa: E402

from autonomous_kart.nodes.localization.localization_node import LocalizationNode  # noqa: E402
from kart_interfaces.msg import GpsFix  # noqa: E402

PARAMS = os.path.join(os.path.dirname(__file__), "..", "autonomous_kart", "params", "localization.yaml")


@pytest.fixture
def node():
    rclpy.init(args=["--ros-args", "--params-file", PARAMS])
    node = LocalizationNode()
    node.on_imu("imu/front", Imu())
    yield node
    node.destroy_node()
    rclpy.shutdown()


def fix(t, x, y, heading_valid=True, heading=0.3):
    msg = GpsFix()
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9))
    msg.fix_quality = GpsFix.FIX_RTK_FIXED
    msg.x, msg.y = x, y
    msg.heading, msg.heading_valid = heading, heading_valid
    msg.position_covariance = [1e-4, 0.0, 0.0, 1e-4]
    msg.heading_variance = 1e-4
    return msg


def test_waits_for_heading_before_init(node):
    node.on_gps(fix(1.0, 2.0, 3.0, heading_valid=False))
    assert not node.ekf.initialized
    node.on_gps(fix(1.1, 2.0, 3.0))
    assert node.ekf.initialized
    assert node.ekf.x[:3] == pytest.approx([2.0, 3.0, 0.3])


def test_drops_no_fix_and_non_finite(node):
    msg = fix(1.0, 2.0, 3.0)
    msg.fix_quality = GpsFix.FIX_NONE
    node.on_gps(msg)
    node.on_gps(fix(1.0, math.nan, 3.0))
    assert not node.ekf.initialized


def test_reinit_after_repeated_rtk_rejects(node):
    node.on_gps(fix(1.0, 0.0, 0.0))
    t = 1.0
    for _ in range(node.param("reinit_rejects")):
        t += 0.1
        node.on_gps(fix(t, 50.0, 0.0))
    # Enough rejects, but not yet spanning reinit_s
    assert node.ekf.x[0] == pytest.approx(0.0)
    while t < 1.0 + node.param("reinit_s") + 0.15:
        t += 0.1
        node.on_gps(fix(t, 50.0, 0.0))
    assert node.ekf.x[0] == pytest.approx(50.0)


def test_gru_mode_trains_in_background_and_saves_accepted_weights(tmp_path):
    pytest.importorskip("torch")
    from concurrent.futures import Future

    from autonomous_kart.nodes.localization.gru_training import GruModel, to_npz

    overrides = ["-p", "mode:=gru", "-p", f"train_weights_dir:={tmp_path}"]
    rclpy.init(args=["--ros-args", "--params-file", PARAMS, *overrides])
    node = LocalizationNode()
    try:
        node.train_cycle()
        assert node.training is None  # no system_state yet, so training must not start

        done = Future()
        npz = to_npz(GruModel(4).double(), np.zeros(4), np.ones(4))
        done.set_result((npz, "GRU training accepted"))
        node.training = done
        node.collect_training()
        assert node.training is None and node.weights == npz
        saved = list(tmp_path.glob("gru_*.npz"))
        assert len(saved) == 1 and saved[0].read_bytes() == npz
    finally:
        node.trainer.shutdown()
        node.destroy_node()
        rclpy.shutdown()
