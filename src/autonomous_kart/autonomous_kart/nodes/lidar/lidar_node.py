"""Lidar node: publishes scans with no returns on scan.

https://github.com/EVC-Purdue/A27/issues/25
"""

import math

from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from autonomous_kart.common import QOS_LOW, run_node


class LidarNode(Node):
    def __init__(self):
        super().__init__("lidar_node")
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("num_beams", 3200)
        self.num_beams = self.get_parameter("num_beams").value

        self.scan_pub = self.create_publisher(LaserScan, "scan", QOS_LOW)
        self.create_timer(1.0 / self.get_parameter("publish_hz").value, self.tick)

    def tick(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "laser"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = 2.0 * math.pi / self.num_beams
        msg.range_min = 0.05
        msg.range_max = 40.0
        msg.ranges = [math.inf] * self.num_beams
        self.scan_pub.publish(msg)


def main(args=None):
    run_node(LidarNode, args)


if __name__ == "__main__":
    main()
