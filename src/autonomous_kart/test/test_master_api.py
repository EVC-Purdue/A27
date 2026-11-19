"""Checks the master HTTP API: state changes, MANUAL drive commands, run downloads and the log stream.

https://github.com/EVC-Purdue/A27/issues/19
https://github.com/EVC-Purdue/A27/issues/20
"""

import os

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("flask")

from rcl_interfaces.msg import Log  # noqa: E402

from autonomous_kart import runs  # noqa: E402
from autonomous_kart.nodes.master.master_node import MasterNode, make_app  # noqa: E402


class Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture
def node(tmp_path):
    rclpy.init(args=["--ros-args", "-p", "system_state:=IDLE", "-p", "api_port:=0", "-p", f"bag_dir:={tmp_path}"])
    node = MasterNode()
    node.state_pub = Recorder()
    node.manual_pub = Recorder()
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def client(node):
    return make_app(node).test_client()


def test_state_change_publishes(node, client):
    assert client.get("/state").json == {"state": "IDLE"}
    assert client.post("/state", json={"state": "MANUAL"}).status_code == 200
    assert node.state == "MANUAL"
    assert node.state_pub.msgs[-1].data == "MANUAL"


def test_bad_state_rejected(node, client):
    assert client.post("/state", json={"state": "FAST"}).status_code == 400
    assert client.post("/state").status_code == 400
    assert node.state == "IDLE"
    assert node.state_pub.msgs == []


def test_manual_only_in_manual(node, client):
    assert client.post("/manual", json={"speed": 1.0, "steering": 0.0}).status_code == 409
    assert node.manual_pub.msgs == []


def test_manual_publishes_command(node, client):
    client.post("/state", json={"state": "MANUAL"})
    assert client.post("/manual", json={"speed": 2.0, "steering": -0.1}).status_code == 200
    cmd = node.manual_pub.msgs[-1]
    assert (cmd.speed_setpoint, cmd.steering) == (2.0, -0.1)


@pytest.mark.parametrize(
    "body", [{}, {"speed": 1.0}, {"speed": "fast", "steering": 0.0}, {"speed": float("nan"), "steering": 0.0}]
)
def test_bad_manual_rejected(node, client, body):
    client.post("/state", json={"state": "MANUAL"})
    assert client.post("/manual", json=body).status_code == 400
    assert node.manual_pub.msgs == []


def make_run(bag_dir, name, recording):
    os.makedirs(os.path.join(bag_dir, name, "bag"))
    for i in range(2):
        with open(os.path.join(bag_dir, name, "bag", f"bag_{i}.mcap"), "wb") as f:
            f.write(bytes(range(100)))
    if recording:
        open(os.path.join(bag_dir, name, runs.RECORDING), "w").close()


def test_runs_listed_and_download_resumes(node, client):
    make_run(node.bag_dir, "run_a", recording=True)
    reply = client.get("/runs").json
    assert reply["runs"][0]["files"] == [{"path": "bag/bag_0.mcap", "size": 100}]
    assert client.get("/runs/run_a/files/bag/bag_1.mcap").status_code == 404
    response = client.get("/runs/run_a/files/bag/bag_0.mcap", headers={"Range": "bytes=60-"})
    assert response.status_code == 206
    assert response.data == bytes(range(60, 100))
    assert client.get(f"/runs?since={reply['now'] + 1}").json["runs"] == []


def test_confirm_waits_for_recording_to_stop(node, client):
    make_run(node.bag_dir, "run_a", recording=True)
    assert client.post("/runs/run_a/confirm").status_code == 409
    os.remove(os.path.join(node.bag_dir, "run_a", runs.RECORDING))
    assert client.post("/runs/run_a/confirm").status_code == 200
    assert client.get("/runs").json["runs"][0]["confirmed"]
    assert client.post("/runs/run_b/confirm").status_code == 404


def test_files_outside_runs_refused(node, client):
    make_run(node.bag_dir, "run_a", recording=False)
    assert client.get("/runs/run_a/files/../run_a/bag/bag_0.mcap").status_code == 404
    assert client.get("/runs/..%2F/files/etc/passwd", follow_redirects=True).status_code == 404


def read_events(client, count, after=0):
    response = client.get(f"/logs?after={after}", buffered=False)
    events = []
    for chunk in response.response:
        events.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        if len(events) == count:
            break
    response.close()
    return events


def test_two_log_readers_see_every_entry(node, client):
    for i in range(3):
        node.on_log(Log(name="test", msg=f"entry {i}"))
    first = read_events(client, 3)
    second = read_events(client, 3)
    assert first == second
    assert [e.split("\n")[0] for e in first] == ["id: 1", "id: 2", "id: 3"]
    assert read_events(client, 1, after=2) == first[2:]
