"""3DGS node.

https://github.com/EVC-Purdue/A27/issues/32
"""
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from autonomous_kart.common import IMAGE_QOS, run_node


class Gs3dNode(Node):
    def __init__(self):
        super().__init__("gs3d_node")
        self.declare_parameter("publish_hz", 10.0)

        self.create_subscription(Image, "camera/image_raw", self.on_msg, IMAGE_QOS)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "gs3d/pose", 10)
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
    run_node(Gs3dNode, args)


if __name__ == "__main__":
    main()
