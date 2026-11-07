"""Camera node: publishes blank frames on camera/image_raw.

https://github.com/EVC-Purdue/A27/issues/22
"""

from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from autonomous_kart.common import QOS_IMAGE, run_node


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("frame_rate_hz", Parameter.Type.DOUBLE)
        self.declare_parameter("width", Parameter.Type.INTEGER)
        self.declare_parameter("height", Parameter.Type.INTEGER)
        self.width = self.get_parameter("width").value
        self.height = self.get_parameter("height").value

        self.image_pub = self.create_publisher(Image, "camera/image_raw", QOS_IMAGE)
        self.create_timer(1.0 / self.get_parameter("frame_rate_hz").value, self.tick)

    def tick(self):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.width = self.width
        msg.height = self.height
        msg.encoding = "rgb8"
        msg.step = 3 * self.width
        msg.data = bytes(msg.step * self.height)
        self.image_pub.publish(msg)


def main(args=None):
    run_node(CameraNode, args)


if __name__ == "__main__":
    main()
