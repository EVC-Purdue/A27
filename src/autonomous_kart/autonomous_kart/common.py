"""QoS profiles and the entry-point helper shared by every node.

https://github.com/EVC-Purdue/A27/issues/11
"""

import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

SYSTEM_STATES = ("IDLE", "MANUAL", "AUTONOMOUS", "STOPPED")

# Default for slow topics (20 Hz and below): every message should arrive
QOS_LOW = QoSProfile(
    depth=10,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# Default for fast topics (50 Hz and up): dropped messages are replaced
QOS_HIGH = QoSProfile(
    depth=5,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

# Latched so a node started late still gets the current state
QOS_STATE = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Slow image consumers drop stale frames
QOS_IMAGE = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def with_default_params(name, args):
    # Loads system.yaml and <name>.yaml first, so command-line and launch params override them
    args = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory("autonomous_kart"), "params")
    defaults = ["--ros-args"]
    for file in ("system.yaml", f"{name}.yaml"):
        defaults += ["--params-file", os.path.join(params, file)]
    return args[:1] + defaults + ["--"] + args[1:]


def run_node(node_cls, args=None):
    name = node_cls.__module__.rsplit(".", 1)[-1].removesuffix("_node")
    rclpy.init(args=with_default_params(name, args))
    node = node_cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
