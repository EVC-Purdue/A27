"""Pathfinder node: publishes a straight line and runs the selected controller in AUTONOMOUS.

https://github.com/EVC-Purdue/A27/issues/10
https://github.com/EVC-Purdue/A27/issues/13
https://github.com/EVC-Purdue/A27/issues/33
"""

import math

from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String

from autonomous_kart.common import QOS_HIGH, QOS_STATE, run_node
from autonomous_kart.nodes.pathfinder.controllers import base, make_controller
from kart_interfaces.msg import (
    DriveCommand,
    KartTelemetry,
    Line,
    LinePoint,
    TrackedObjectArray,
)

NO_ODOM_WARN_PERIOD_S = 1.0


def to_inputs(time_s, odom, telemetry, tracks, line):
    q = odom.pose.pose.orientation
    yaw_rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return base.ControllerInputs(
        time_s=time_s,
        x_m=odom.pose.pose.position.x,
        y_m=odom.pose.pose.position.y,
        yaw_rad=yaw_rad,
        vx_mps=odom.twist.twist.linear.x,
        vy_mps=odom.twist.twist.linear.y,
        yaw_rate_radps=odom.twist.twist.angular.z,
        wheel_speed_mps=telemetry.wheel_speed,
        steering_angle_rad=telemetry.steering_angle,
        steering_motor_current_a=telemetry.steering_motor_current,
        line=tuple(base.LinePoint(p.x, p.y, p.curvature, p.target_speed) for p in line.points),
        tracks=tuple(
            base.Track(
                o.id,
                o.object_class,
                o.footprint_size,
                o.position.x,
                o.position.y,
                o.velocity.x,
                o.velocity.y,
            )
            for o in tracks.objects
        ),
    )


class PathfinderNode(Node):
    def __init__(self):
        super().__init__("pathfinder_node", automatically_declare_parameters_from_overrides=True)
        system_frequency_hz = self.get_parameter("system_frequency_hz").value
        name = self.get_parameter("controller").value
        params = {k: p.value for k, p in self.get_parameters_by_prefix(name).items()}
        self.controller = make_controller(name, params)
        self.get_logger().info(f"controller {name}")

        self.system_state = None
        self.odom = None
        self.telemetry = KartTelemetry()
        self.tracks = TrackedObjectArray()

        self.create_subscription(Odometry, "odom", self.on_odom, QOS_HIGH)
        self.create_subscription(TrackedObjectArray, "localization/tracks", self.on_tracks, QOS_HIGH)
        self.create_subscription(KartTelemetry, "e_comms/telemetry", self.on_telemetry, QOS_HIGH)
        self.create_subscription(String, "system_state", self.on_state, QOS_STATE)
        self.line_pub = self.create_publisher(Line, "pathfinder/line", QOS_HIGH)
        self.cmd_pub = self.create_publisher(DriveCommand, "cmd_drive", QOS_HIGH)
        self.brake_pub = self.create_publisher(Bool, "mechanical_brake/command", QOS_HIGH)
        self.create_timer(1.0 / system_frequency_hz, self.tick)

    def on_odom(self, msg):
        self.odom = msg

    def on_tracks(self, msg):
        self.tracks = msg

    def on_telemetry(self, msg):
        self.telemetry = msg

    def on_state(self, msg):
        if msg.data != self.system_state:
            self.controller.reset()
        self.system_state = msg.data

    def tick(self):
        now = self.get_clock().now()
        line = Line()
        line.header.stamp = now.to_msg()
        line.header.frame_id = "odom"
        line.points = [LinePoint(x=float(i), y=0.0, curvature=0.0, target_speed=0.0) for i in range(10)]
        self.line_pub.publish(line)

        # Only AUTONOMOUS runs the controller; every other state commands no motion
        out = base.ControllerOutput()
        if self.system_state == "AUTONOMOUS":
            if self.odom is None:
                self.get_logger().warn(
                    "AUTONOMOUS but no odom received",
                    throttle_duration_sec=NO_ODOM_WARN_PERIOD_S,
                )
            else:
                inputs = to_inputs(now.nanoseconds * 1e-9, self.odom, self.telemetry, self.tracks, line)
                out = self.controller.step(inputs)

        cmd = DriveCommand()
        cmd.header.stamp = line.header.stamp
        cmd.speed_setpoint = out.speed_setpoint_mps
        cmd.steering = out.steering_rad
        cmd.electrical_brake = out.electrical_brake
        self.cmd_pub.publish(cmd)
        self.brake_pub.publish(Bool(data=out.mechanical_brake))


def main(args=None):
    run_node(PathfinderNode, args)


if __name__ == "__main__":
    main()
