"""GPS node: publishes no-fix GpsFix messages on gps.

https://github.com/EVC-Purdue/A27/issues/23
"""

from rclpy.node import Node

from autonomous_kart.common import run_node
from kart_interfaces.msg import GpsFix


class GpsNode(Node):
    def __init__(self):
        super().__init__("gps_node")
        self.declare_parameter("publish_hz", 20.0)

        self.gps_pub = self.create_publisher(GpsFix, "gps", 10)
        self.create_timer(1.0 / self.get_parameter("publish_hz").value, self.tick)

    def tick(self):
        msg = GpsFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.fix_quality = GpsFix.FIX_NONE
        msg.heading_valid = False
        self.gps_pub.publish(msg)


def main(args=None):
    run_node(GpsNode, args)


if __name__ == "__main__":
    main()
