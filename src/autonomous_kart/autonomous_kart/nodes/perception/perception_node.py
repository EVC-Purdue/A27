"""Perception node: publishes empty measurement arrays.

https://github.com/EVC-Purdue/A27/issues/30
"""

from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

from autonomous_kart.common import QOS_IMAGE, QOS_LOW, run_node
from kart_interfaces.msg import ObjectMeasurementArray


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("publish_hz", 10.0)

        self.create_subscription(Image, "camera/image_raw", self.on_msg, QOS_IMAGE)
        self.create_subscription(LaserScan, "scan", self.on_msg, QOS_LOW)
        self.measurements_pub = self.create_publisher(ObjectMeasurementArray, "perception/measurements", QOS_LOW)
        self.create_timer(1.0 / self.get_parameter("publish_hz").value, self.tick)

    def on_msg(self, msg):
        pass

    def tick(self):
        msg = ObjectMeasurementArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.measurements_pub.publish(msg)


def main(args=None):
    run_node(PerceptionNode, args)


if __name__ == "__main__":
    main()
