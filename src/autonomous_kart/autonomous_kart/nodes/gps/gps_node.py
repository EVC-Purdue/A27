"""GPS node: publishes no-fix GpsFix messages on gps.

https://github.com/EVC-Purdue/A27/issues/23
"""

from rclpy.node import Node
from rclpy.parameter import Parameter

from autonomous_kart.common import QOS_LOW, run_node
from kart_interfaces.msg import GpsFix


class GpsNode(Node):
    def __init__(self):
        super().__init__("gps_node")
        self.declare_parameter("publish_rate_hz", Parameter.Type.DOUBLE)
        self.declare_parameter("simulation_mode", Parameter.Type.BOOL)
        self.declare_parameter("sim_position_sigma_m", Parameter.Type.DOUBLE)
        self.declare_parameter("sim_heading_sigma_rad", Parameter.Type.DOUBLE)
        self.simulation_mode = self.get_parameter("simulation_mode").value
        self.sim_position_var = self.get_parameter("sim_position_sigma_m").value ** 2
        self.sim_heading_var = self.get_parameter("sim_heading_sigma_rad").value ** 2

        self.gps_pub = self.create_publisher(GpsFix, "gps", QOS_LOW)
        self.create_timer(1.0 / self.get_parameter("publish_rate_hz").value, self.tick)

    def tick(self):
        msg = GpsFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.fix_quality = GpsFix.FIX_NONE
        msg.heading_valid = False
        if self.simulation_mode:
            msg.fix_quality = GpsFix.FIX_RTK_FIXED
            msg.heading_valid = True
            msg.position_covariance = [self.sim_position_var, 0.0, 0.0, self.sim_position_var]
            msg.heading_variance = self.sim_heading_var
        self.gps_pub.publish(msg)


def main(args=None):
    run_node(GpsNode, args)


if __name__ == "__main__":
    main()
