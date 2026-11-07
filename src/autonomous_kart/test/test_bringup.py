"""Launches bringup_sim and checks every topic flows and cmd_drive reaches e_comms.

https://github.com/EVC-Purdue/A27/issues/11
"""

import os
import shutil
import signal
import subprocess
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import PoseWithCovarianceStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import Image, Imu, LaserScan  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from autonomous_kart.common import QOS_HIGH, QOS_IMAGE, QOS_LOW, QOS_STATE  # noqa: E402
from kart_interfaces.msg import (  # noqa: E402
    DriveCommand,
    GpsFix,
    KartTelemetry,
    Line,
    ObjectMeasurementArray,
    TrackedObjectArray,
)

TOPICS = {
    "system_state": (String, QOS_STATE),
    "camera/image_raw": (Image, QOS_IMAGE),
    "gps": (GpsFix, QOS_LOW),
    "imu/front": (Imu, QOS_HIGH),
    "imu/back": (Imu, QOS_HIGH),
    "scan": (LaserScan, QOS_LOW),
    "e_comms/telemetry": (KartTelemetry, QOS_HIGH),
    "mechanical_brake/command": (Bool, QOS_HIGH),
    "mechanical_brake/state": (Bool, QOS_LOW),
    "odom": (Odometry, QOS_HIGH),
    "tf": (TFMessage, QOS_HIGH),
    "localization/tracks": (TrackedObjectArray, QOS_HIGH),
    "perception/measurements": (ObjectMeasurementArray, QOS_LOW),
    "slam/pose": (PoseWithCovarianceStamped, QOS_LOW),
    "gs3d/pose": (PoseWithCovarianceStamped, QOS_LOW),
    "pathfinder/line": (Line, QOS_HIGH),
    "cmd_drive": (DriveCommand, QOS_HIGH),
}


@pytest.fixture(scope="module")
def stack():
    if shutil.which("ros2") is None:
        pytest.skip("ros2 not on PATH")
    proc = subprocess.Popen(
        ["ros2", "launch", "autonomous_kart", "bringup_sim.launch.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    rclpy.init()
    node = rclpy.create_node("bringup_listener")
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(proc.pid, signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
        print(out)


def test_every_topic_flows(stack):
    seen = set()
    for topic, (msg_type, qos) in TOPICS.items():
        stack.create_subscription(msg_type, topic, lambda _msg, t=topic: seen.add(t), qos)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and seen != set(TOPICS):
        rclpy.spin_once(stack, timeout_sec=0.1)
    assert set(TOPICS) - seen == set()


def test_cmd_drive_reaches_e_comms(stack):
    deadline = time.monotonic() + 10.0
    subscribers = []
    while time.monotonic() < deadline:
        subscribers = [i.node_name for i in stack.get_subscriptions_info_by_topic("cmd_drive")]
        if "e_comms_node" in subscribers:
            break
        time.sleep(0.2)
    assert "e_comms_node" in subscribers
