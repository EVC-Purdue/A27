"""SLAM node: publishes a still pose on slam/pose.

https://github.com/EVC-Purdue/A27/issues/31
"""
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from autonomous_kart.common import run_node


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.declare_parameter("publish_hz", 10.0)

        self.create_subscription(LaserScan, "scan", self.on_msg, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "slam/pose", 10)
        self.create_timer(1.0 / self.get_parameter("publish_hz").value, self.tick)

    def on_msg(self, msg):
        pass

    def tick(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)


def main(args=None):
    run_node(SlamNode, args)


if __name__ == "__main__":
    main()
