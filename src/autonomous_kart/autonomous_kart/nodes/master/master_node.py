"""Master node: publishes the latched system_state.

https://github.com/EVC-Purdue/A27/issues/19
"""

from rclpy.node import Node
from std_msgs.msg import String

from autonomous_kart.common import QOS_STATE, SYSTEM_STATES, run_node


class MasterNode(Node):
    def __init__(self):
        super().__init__("master_node")
        self.declare_parameter("system_state", "IDLE")
        state = self.get_parameter("system_state").value
        if state not in SYSTEM_STATES:
            raise ValueError(f"system_state must be one of {SYSTEM_STATES}, got {state}")

        self.state_pub = self.create_publisher(String, "system_state", QOS_STATE)
        self.state_pub.publish(String(data=state))
        self.get_logger().info(f"system_state {state}")


def main(args=None):
    run_node(MasterNode, args)


if __name__ == "__main__":
    main()
