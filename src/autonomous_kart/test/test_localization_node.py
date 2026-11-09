"""Checks the localization node: init from a heading fix, re-init after repeated RTK rejects, background GRU training,
and tracks from perception measurements.

https://github.com/EVC-Purdue/A27/issues/28
https://github.com/EVC-Purdue/A27/issues/29
"""

import math
import os

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from sensor_msgs.msg import Imu  # noqa: E402

from autonomous_kart.nodes.localization.localization_node import LocalizationNode  # noqa: E402
from kart_interfaces.msg import GpsFix, ObjectMeasurement, ObjectMeasurementArray  # noqa: E402

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


def measurements(t, x, y):
    """One kart at body (x, y) m, seen at time t."""
    msg = ObjectMeasurementArray()
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9))
    obj = ObjectMeasurement(object_class=ObjectMeasurement.CLASS_KART, footprint_size=1.5)
    obj.position.x, obj.position.y = x, y
    obj.position_covariance = [0.04, 0.0, 0.0, 0.04]
    msg.objects = [obj]
    return msg


def test_measurements_become_a_track_in_odom(node):
    node.on_gps(fix(1.0, 2.0, 3.0))
    node.tick()  # stores our pose at the EKF stamp
    for k in range(node.param("tracker.confirm_hits") - 1):
        node.on_measurements(measurements(1.0 + 0.05 * k, 10.0, 0.0))
        assert node.tracker.predicted(1.0 + 0.05 * k) == []  # not confirmed yet
    node.on_measurements(measurements(1.1, 10.0, 0.0))
    (track,) = node.tracker.predicted(1.1)
    # The kart is 10 m ahead of a pose at (2, 3) heading 0.3 rad
    assert (track.x, track.y) == pytest.approx((2.0 + 10.0 * math.cos(0.3), 3.0 + 10.0 * math.sin(0.3)), abs=0.05)
    assert track.object_class == ObjectMeasurement.CLASS_KART


def test_measurement_without_a_pose_is_dropped(node):
    node.on_measurements(measurements(1.0, 10.0, 0.0))  # EKF not initialized, so no pose
    node.on_gps(fix(1.0, 2.0, 3.0))
    node.tick()
    node.on_measurements(measurements(5.0, 10.0, 0.0))  # further than pose_max_gap_s from any pose
    assert node.tracker._tracks == []


def test_gru_mode_trains_in_background_and_saves_accepted_weights(tmp_path):
    pytest.importorskip("torch")
    from concurrent.futures import Future

    from autonomous_kart.nodes.localization.gru_training import GruModel, to_npz

    overrides = ["-p", "mode:=gru", "-p", f"train_weights_dir:={tmp_path}"]
    rclpy.init(args=["--ros-args", "--params-file", PARAMS, *overrides])
    node = LocalizationNode()
    try:
        node.train_cycle()
        assert node.trainer.running is None  # no system_state yet, so training must not start

        done = Future()
        npz = to_npz(GruModel(4).double(), np.zeros(4), np.ones(4))
        done.set_result((npz, "GRU training accepted"))
        node.trainer.running = done
        node.collect_training()
        assert node.trainer.running is None and node.trainer.weights == npz
        saved = list(tmp_path.glob("gru_*.npz"))
        assert len(saved) == 1 and saved[0].read_bytes() == npz
    finally:
        node.trainer.shutdown()
        node.destroy_node()
        rclpy.shutdown()
