"""Host node: one per host, publishes the host's commit and forwards its journald onto the journal topic.

https://github.com/EVC-Purdue/A27/issues/20
"""

import json
import os
import re
import shutil
import socket
import subprocess
import threading

from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import Log
from rclpy.node import Node
from std_msgs.msg import String

from autonomous_kart.common import QOS_LOW, QOS_STATE, run_node

# journald priority to rcl_interfaces/Log level
LEVELS = {0: Log.FATAL, 1: Log.FATAL, 2: Log.FATAL, 3: Log.ERROR, 4: Log.WARN, 5: Log.INFO, 6: Log.INFO, 7: Log.DEBUG}


def git(workspace, *args):
    cmd = ["git", "-c", f"safe.directory={workspace}", "-C", workspace, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def host_info(host, workspace):
    patch = git(workspace, "diff", "HEAD")
    return {
        "host": host,
        "branch": git(workspace, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "commit": git(workspace, "rev-parse", "HEAD").strip(),
        "dirty": bool(patch),
        "patch": patch,
    }


def to_log(host, entry):
    log = Log()
    usec = int(entry.get("__REALTIME_TIMESTAMP", 0))
    log.stamp.sec, log.stamp.nanosec = usec // 1_000_000, (usec % 1_000_000) * 1000
    log.level = LEVELS.get(int(entry.get("PRIORITY", 6)), Log.INFO)
    log.name = f"{host}/{entry.get('SYSLOG_IDENTIFIER', 'journal')}"
    message = entry.get("MESSAGE", "")
    log.msg = message if isinstance(message, str) else str(message)
    return log


class HostNode(Node):
    def __init__(self):
        self.host = socket.gethostname()
        super().__init__("host_" + re.sub(r"\W", "_", self.host))
        # install/autonomous_kart/share/autonomous_kart sits four levels below the workspace
        workspace = os.path.abspath(os.path.join(get_package_share_directory("autonomous_kart"), *[".."] * 4))

        self.info_pub = self.create_publisher(String, "host/info", QOS_STATE)
        self.info_pub.publish(String(data=json.dumps(host_info(self.host, workspace))))
        self.journal_pub = self.create_publisher(Log, "journal", QOS_LOW)

        self.journal = None
        if shutil.which("journalctl") is None:
            self.get_logger().warn("journalctl not found, journald is not forwarded")
        else:
            cmd = ["journalctl", "--follow", "--lines=0", "--output=json"]
            self.journal = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self.forward_journal, daemon=True).start()

    def forward_journal(self):
        for line in self.journal.stdout:
            try:
                self.journal_pub.publish(to_log(self.host, json.loads(line)))
            except (ValueError, TypeError):
                continue

    def destroy_node(self):
        if self.journal is not None:
            self.journal.terminate()
        super().destroy_node()


def main(args=None):
    run_node(HostNode, args)


if __name__ == "__main__":
    main()
