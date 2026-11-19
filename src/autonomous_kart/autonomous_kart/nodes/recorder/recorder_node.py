"""Recorder node: records MCAP bags in MANUAL and AUTONOMOUS and writes each run's manifest.

https://github.com/EVC-Purdue/A27/issues/20
"""

import json
import os
import signal
import subprocess
import threading
import time
from datetime import UTC, datetime

from rcl_interfaces.msg import ParameterEvent
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.qos import QoSProfile, qos_profile_parameter_events
from std_msgs.msg import String

from autonomous_kart import runs
from autonomous_kart.common import QOS_STATE, run_node

DRIVING = ("MANUAL", "AUTONOMOUS")
IMAGE_TYPES = ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"]
STOP_TIMEOUT_S = 10.0
SERVICE_TIMEOUT_S = 2.0
CLEANUP_PERIOD_S = 30.0

# One latched host/info per host, so keep more than one
QOS_HOSTS = QoSProfile(depth=16, reliability=QOS_STATE.reliability, durability=QOS_STATE.durability)


def wait(future, timeout_s):
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    return future.result() if future.done() else None


class RecorderNode(Node):
    def __init__(self):
        super().__init__("recorder_node")
        self.declare_parameter("bag_dir", Parameter.Type.STRING)
        self.declare_parameter("split_s", Parameter.Type.INTEGER)
        self.declare_parameter("record_images", Parameter.Type.BOOL)
        self.declare_parameter("min_free_gb", Parameter.Type.DOUBLE)
        self.bag_dir = os.path.expanduser(self.get_parameter("bag_dir").value)
        os.makedirs(self.bag_dir, exist_ok=True)

        self.proc = None
        self.run_dir = None
        self.manifest = None
        self.system_state = None
        self.hosts = {}

        self.create_subscription(String, "system_state", self.on_state, QOS_STATE)
        self.create_subscription(String, "host/info", self.on_host, QOS_HOSTS)
        self.create_subscription(ParameterEvent, "/parameter_events", self.on_params, qos_profile_parameter_events)
        self.create_timer(1.0, self.check)
        self.create_timer(CLEANUP_PERIOD_S, self.cleanup)
        self.cleanup()

    def on_state(self, msg):
        self.system_state = msg.data
        if msg.data in DRIVING and self.proc is None:
            self.start()
        elif msg.data not in DRIVING and self.proc is not None:
            self.stop()

    def on_host(self, msg):
        info = json.loads(msg.data)
        self.hosts[info["host"]] = info
        if self.run_dir is not None:
            self.write_manifest()

    def on_params(self, msg):
        if self.run_dir is None or msg.node == f"/{self.get_name()}":
            return
        changes = [(p.name, parameter_value_to_python(p.value)) for p in msg.new_parameters + msg.changed_parameters]
        changes += [(p.name, None) for p in msg.deleted_parameters]
        with open(os.path.join(self.run_dir, "param_changes.jsonl"), "a") as f:
            for name, value in changes:
                line = {"time": time.time(), "node": msg.node, "name": name, "value": value}
                f.write(json.dumps(line, default=str) + "\n")

    def check(self):
        # A recorder that died mid-drive closes its run and a new one starts
        if self.proc is not None and self.proc.poll() is not None:
            self.get_logger().error(f"ros2 bag record exited with {self.proc.returncode}, restarting")
            self.stop()
            if self.system_state in DRIVING:
                self.start()

    def cleanup(self):
        min_free_bytes = self.get_parameter("min_free_gb").value * 1e9
        for name in runs.cleanup(self.bag_dir, min_free_bytes):
            self.get_logger().info(f"deleted confirmed run {name}")

    def start(self):
        tag = "auto" if self.system_state == "AUTONOMOUS" else "manual"
        self.run_dir = os.path.join(self.bag_dir, f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}_{tag}")
        os.makedirs(self.run_dir)
        open(os.path.join(self.run_dir, runs.RECORDING), "w").close()

        cmd = ["ros2", "bag", "record", "--all-topics", "--storage", "mcap"]
        cmd += ["--storage-preset-profile", "zstd_fast"]
        cmd += ["--max-bag-duration", str(self.get_parameter("split_s").value)]
        cmd += ["--output", os.path.join(self.run_dir, "bag")]
        if not self.get_parameter("record_images").value:
            cmd += ["--exclude-topic-types", *IMAGE_TYPES]
        self.log = open(os.path.join(self.run_dir, "recorder.log"), "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)

        self.manifest = {"started_at": datetime.now(UTC).isoformat(), "trigger_state": self.system_state}
        self.write_manifest()
        threading.Thread(target=self.save_params, args=(self.run_dir,), daemon=True).start()
        self.get_logger().info(f"recording {self.run_dir}")

    def stop(self):
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        # Marker goes first so the manifest rewrite after it marks the run as updated for clients
        os.remove(os.path.join(self.run_dir, runs.RECORDING))
        self.manifest["stopped_at"] = datetime.now(UTC).isoformat()
        self.write_manifest()
        self.get_logger().info(f"stopped {self.run_dir}")
        self.proc = None
        self.run_dir = None

    def write_manifest(self):
        hosts = {}
        os.makedirs(os.path.join(self.run_dir, "hosts"), exist_ok=True)
        for host, info in self.hosts.items():
            hosts[host] = {k: info[k] for k in ("branch", "commit", "dirty")}
            if info["patch"]:
                with open(os.path.join(self.run_dir, "hosts", f"{host}.patch"), "w") as f:
                    f.write(info["patch"])
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump({**self.manifest, "hosts": hosts}, f, indent=2)

    def save_params(self, run_dir):
        # Runs on its own thread so the service calls can wait while the executor spins
        params = {}
        for name, namespace in self.get_node_names_and_namespaces():
            node = f"{namespace.rstrip('/')}/{name}"
            if name.startswith("_") or name in (self.get_name(), "rosbag2_recorder"):
                continue
            list_client = self.create_client(ListParameters, f"{node}/list_parameters")
            get_client = self.create_client(GetParameters, f"{node}/get_parameters")
            if list_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
                names = wait(list_client.call_async(ListParameters.Request()), SERVICE_TIMEOUT_S)
                if names is not None:
                    request = GetParameters.Request(names=names.result.names)
                    values = wait(get_client.call_async(request), SERVICE_TIMEOUT_S)
                    if values is not None:
                        params[node] = {
                            n: parameter_value_to_python(v) for n, v in zip(request.names, values.values, strict=True)
                        }
            self.destroy_client(list_client)
            self.destroy_client(get_client)
        with open(os.path.join(run_dir, "params_start.json"), "w") as f:
            json.dump(params, f, indent=2, default=str)

    def destroy_node(self):
        if self.proc is not None:
            self.stop()
        super().destroy_node()


def main(args=None):
    run_node(RecorderNode, args)


if __name__ == "__main__":
    main()
