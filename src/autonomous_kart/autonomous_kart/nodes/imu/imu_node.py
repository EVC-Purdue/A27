"""Front IMU node: publishes a level, still IMU on imu/front.

https://github.com/EVC-Purdue/A27/issues/24
"""

from rclpy.node import Node
from sensor_msgs.msg import Imu

from autonomous_kart.common import run_node


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu_node")
        self.declare_parameter("publish_hz", 100.0)

        self.imu_pub = self.create_publisher(Imu, "imu/front", 10)
        self.create_timer(1.0 / self.get_parameter("publish_hz").value, self.tick)

    def tick(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_front"
        msg.orientation_covariance[0] = -1.0
        msg.linear_acceleration.z = 9.81
        msg.angular_velocity_covariance = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]
        msg.linear_acceleration_covariance = [1e-2, 0.0, 0.0, 0.0, 1e-2, 0.0, 0.0, 0.0, 1e-2]
        self.imu_pub.publish(msg)


def main(args=None):
    run_node(ImuNode, args)


if __name__ == "__main__":
    main()
