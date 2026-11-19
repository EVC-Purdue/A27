"""Master node: publishes the latched system_state and serves the HTTP API for commands, runs and logs.

https://github.com/EVC-Purdue/A27/issues/19
"""

import itertools
import json
import logging
import math
import os
import threading
import time
from collections import deque

from flask import Flask, Response, jsonify, request, send_from_directory
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from autonomous_kart import runs
from autonomous_kart.common import QOS_LOW, QOS_STATE, SYSTEM_STATES, run_node
from kart_interfaces.msg import DriveCommand

logging.getLogger("werkzeug").setLevel(logging.ERROR)

LOG_BUFFER = 10000
HEARTBEAT_S = 15.0


class MasterNode(Node):
    def __init__(self):
        super().__init__("master_node")
        self.declare_parameter("system_state", Parameter.Type.STRING)
        self.declare_parameter("api_port", Parameter.Type.INTEGER)
        self.declare_parameter("bag_dir", Parameter.Type.STRING)
        self.bag_dir = os.path.expanduser(self.get_parameter("bag_dir").value)
        state = self.get_parameter("system_state").value
        if state not in SYSTEM_STATES:
            raise ValueError(f"system_state must be one of {SYSTEM_STATES}, got {state}")

        self.state_pub = self.create_publisher(String, "system_state", QOS_STATE)
        self.manual_pub = self.create_publisher(DriveCommand, "manual/command", QOS_LOW)
        self.set_state(state)

        # ROS and journald logs from every host, numbered so each reader keeps its own cursor
        self.logs = deque(maxlen=LOG_BUFFER)
        self.log_seq = 0
        self.log_cond = threading.Condition()
        self.create_subscription(Log, "/rosout", self.on_log, QOS_LOW)
        self.create_subscription(Log, "journal", self.on_log, QOS_LOW)

        port = self.get_parameter("api_port").value
        app = make_app(self)
        threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": port}, daemon=True).start()

    def set_state(self, state):
        self.state = state
        self.state_pub.publish(String(data=state))
        self.get_logger().info(f"system_state {state}")

    def manual(self, speed, steering):
        cmd = DriveCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.speed_setpoint = speed
        cmd.steering = steering
        self.manual_pub.publish(cmd)

    def on_log(self, msg):
        entry = {"time": msg.stamp.sec + msg.stamp.nanosec * 1e-9, "level": msg.level, "name": msg.name, "msg": msg.msg}
        with self.log_cond:
            self.log_seq += 1
            self.logs.append((self.log_seq, entry))
            self.log_cond.notify_all()

    def logs_after(self, seq, timeout_s):
        with self.log_cond:
            if seq > self.log_seq:
                seq = 0  # cursor from before a master restart
            if seq == self.log_seq:
                self.log_cond.wait(timeout_s)
            if not self.logs:
                return []
            start = max(0, seq - self.logs[0][0] + 1)
            return list(itertools.islice(self.logs, start, None))


def make_app(node):
    app = Flask(__name__)

    @app.get("/state")
    def get_state():
        return jsonify(state=node.state)

    @app.post("/state")
    def post_state():
        state = (request.get_json(silent=True) or {}).get("state")
        if state not in SYSTEM_STATES:
            return jsonify(error=f"state must be one of {SYSTEM_STATES}"), 400
        node.set_state(state)
        return jsonify(state=state)

    # Laptop sends this at ~5 Hz; pathfinder stops the kart when it goes stale
    @app.post("/manual")
    def post_manual():
        if node.state != "MANUAL":
            return jsonify(error=f"state is {node.state}, not MANUAL"), 409
        body = request.get_json(silent=True) or {}
        try:
            speed, steering = float(body["speed"]), float(body["steering"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="speed (m/s) and steering (rad) must be numbers"), 400
        if not (math.isfinite(speed) and math.isfinite(steering)):
            return jsonify(error="speed and steering must be finite"), 400
        node.manual(speed, steering)
        return jsonify(ok=True)

    # Clients pass the previous reply's "now" as since, so they only see runs that changed
    @app.get("/runs")
    def get_runs():
        try:
            since = float(request.args.get("since", 0.0))
        except ValueError:
            return jsonify(error="since must be a unix time in seconds"), 400
        now = time.time()
        return jsonify(now=now, runs=runs.list_runs(node.bag_dir, since))

    # Range requests resume a dropped download
    @app.get("/runs/<name>/files/<path:file>")
    def get_run_file(name, file):
        run_dir = os.path.join(node.bag_dir, name)
        if not name.startswith("run_") or "/" in name or file not in runs.closed_files(run_dir):
            return jsonify(error=f"{name}/{file} is not a closed file"), 404
        return send_from_directory(run_dir, file)

    @app.post("/runs/<name>/confirm")
    def confirm_run(name):
        run_dir = os.path.join(node.bag_dir, name)
        if not name.startswith("run_") or "/" in name or not os.path.isdir(run_dir):
            return jsonify(error=f"no run {name}"), 404
        if runs.is_recording(run_dir):
            return jsonify(error=f"{name} is still recording"), 409
        open(os.path.join(run_dir, runs.CONFIRMED), "w").close()
        return jsonify(name=name, confirmed=True)

    # Server-sent events; a reconnecting client resumes from Last-Event-ID
    @app.get("/logs")
    def get_logs():
        try:
            after = int(request.headers.get("Last-Event-ID", request.args.get("after", 0)))
        except ValueError:
            return jsonify(error="after must be an integer"), 400

        def stream(seq):
            while True:
                entries = node.logs_after(seq, HEARTBEAT_S)
                if not entries:
                    yield ": keepalive\n\n"
                for entry_seq, entry in entries:
                    yield f"id: {entry_seq}\ndata: {json.dumps(entry)}\n\n"
                    seq = entry_seq

        return Response(stream(after), mimetype="text/event-stream")

    return app


def main(args=None):
    run_node(MasterNode, args)


if __name__ == "__main__":
    main()
