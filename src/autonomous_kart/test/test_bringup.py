"""Launches bringup_sim: topics flow, cmd_drive reaches e_comms, the API and bridge answer, and drives are recorded.

https://github.com/EVC-Purdue/A27/issues/11
https://github.com/EVC-Purdue/A27/issues/20
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

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
    "host/info": (String, QOS_STATE),
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


def wait_for(check, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return check()
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def test_master_api_answers(stack):
    with wait_for(lambda: urllib.request.urlopen("http://localhost:8000/state", timeout=2)) as response:
        assert json.load(response)["state"] == "IDLE"


def test_foxglove_bridge_listens(stack):
    wait_for(lambda: socket.create_connection(("localhost", 8765), timeout=2)).close()


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f"http://localhost:8000{path}", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def test_drive_is_recorded(stack):
    since = wait_for(lambda: api("/runs"))["now"]
    api("/state", {"state": "MANUAL"})
    time.sleep(12.0)
    api("/state", {"state": "IDLE"})

    deadline = time.monotonic() + 20.0
    finished = []
    while time.monotonic() < deadline and not finished:
        finished = [r for r in api(f"/runs?since={since}")["runs"] if not r["recording"]]
        time.sleep(0.5)
    assert len(finished) == 1
    run = finished[0]
    paths = [f["path"] for f in run["files"]]
    assert len([p for p in paths if p.endswith(".mcap")]) >= 2
    assert {"bag/metadata.yaml", "manifest.json", "params_start.json"} <= set(paths)
    manifest = api(f"/runs/{run['name']}/files/manifest.json")
    assert manifest["trigger_state"] == "MANUAL" and manifest["hosts"]
