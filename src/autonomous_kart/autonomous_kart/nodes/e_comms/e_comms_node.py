"""e_comms node: echoes cmd_drive into telemetry and publishes imu/back.

https://github.com/EVC-Purdue/A27/issues/26
https://github.com/EVC-Purdue/A27/issues/27
"""
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from autonomous_kart.common import STATE_QOS, run_node
from kart_interfaces.msg import DriveCommand, KartTelemetry


class ECommsNode(Node):
    def __init__(self):
        super().__init__("e_comms_node")
        self.declare_parameter("publish_hz", 50.0)
        self.last_cmd = DriveCommand()
        self.system_state = None

        self.create_subscription(DriveCommand, "cmd_drive", self.on_cmd, 10)
        self.create_subscription(String, "system_state", self.on_state, STATE_QOS)
        self.telemetry_pub = self.create_publisher(KartTelemetry, "e_comms/telemetry", 10)
        self.imu_pub = self.create_publisher(Imu, "imu/back", 10)
        self.create_timer(1.0 / self.get_parameter("publish_hz").value, self.tick)

    def on_cmd(self, msg):
        self.last_cmd = msg

    def on_state(self, msg):
        self.system_state = msg.data

    def tick(self):
        now = self.get_clock().now().to_msg()
        telemetry = KartTelemetry()
        telemetry.header.stamp = now
        telemetry.header.frame_id = "base_link"
        telemetry.wheel_speed = self.last_cmd.speed_setpoint
        telemetry.steering_angle = self.last_cmd.steering
        self.telemetry_pub.publish(telemetry)

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "imu_back"
        imu.orientation_covariance[0] = -1.0
        imu.linear_acceleration.z = 9.81
        self.imu_pub.publish(imu)


def main(args=None):
    run_node(ECommsNode, args)


if __name__ == "__main__":
    main()
