"""Localization node: publishes a still odom, TF and no tracks.

https://github.com/EVC-Purdue/A27/issues/28
https://github.com/EVC-Purdue/A27/issues/29
"""

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from autonomous_kart.common import QOS_HIGH, QOS_LOW, run_node
from kart_interfaces.msg import (
    GpsFix,
    KartTelemetry,
    ObjectMeasurementArray,
    TrackedObjectArray,
)


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")
        self.declare_parameter("publish_rate_hz", Parameter.Type.DOUBLE)

        self.create_subscription(GpsFix, "gps", self.on_msg, QOS_LOW)
        self.create_subscription(Imu, "imu/front", self.on_msg, QOS_HIGH)
        self.create_subscription(Imu, "imu/back", self.on_msg, QOS_HIGH)
        self.create_subscription(KartTelemetry, "e_comms/telemetry", self.on_msg, QOS_HIGH)
        self.create_subscription(PoseWithCovarianceStamped, "slam/pose", self.on_msg, QOS_LOW)
        self.create_subscription(ObjectMeasurementArray, "perception/measurements", self.on_msg, QOS_LOW)
        self.odom_pub = self.create_publisher(Odometry, "odom", QOS_HIGH)
        self.tracks_pub = self.create_publisher(TrackedObjectArray, "localization/tracks", QOS_HIGH)
        self.transform_broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / self.get_parameter("publish_rate_hz").value, self.tick)

    def on_msg(self, msg):
        pass

    def tick(self):
        now = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id
        transform.transform.rotation.w = 1.0
        self.transform_broadcaster.sendTransform(transform)

        tracks = TrackedObjectArray()
        tracks.header = odom.header
        self.tracks_pub.publish(tracks)


def main(args=None):
    run_node(LocalizationNode, args)


if __name__ == "__main__":
    main()
