"""Mechanical Brake node: reports the last command as its state.

https://github.com/EVC-Purdue/A27/issues/38
"""

from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String

from autonomous_kart.common import QOS_HIGH, QOS_LOW, QOS_STATE, run_node


class MechanicalBrakeNode(Node):
    def __init__(self):
        super().__init__("mechanical_brake_node")
        self.declare_parameter("publish_rate_hz", Parameter.Type.DOUBLE)
        self.engaged = False
        self.system_state = None

        self.create_subscription(Bool, "mechanical_brake/command", self.on_cmd, QOS_HIGH)
        self.create_subscription(String, "system_state", self.on_state, QOS_STATE)
        self.state_pub = self.create_publisher(Bool, "mechanical_brake/state", QOS_LOW)
        self.create_timer(1.0 / self.get_parameter("publish_rate_hz").value, self.tick)

    def on_cmd(self, msg):
        self.engaged = msg.data

    def on_state(self, msg):
        self.system_state = msg.data

    def tick(self):
        self.state_pub.publish(Bool(data=self.engaged))


def main(args=None):
    run_node(MechanicalBrakeNode, args)


if __name__ == "__main__":
    main()
