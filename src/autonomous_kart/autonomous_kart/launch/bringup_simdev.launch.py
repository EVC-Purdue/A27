# https://github.com/EVC-Purdue/A27/issues/14
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter, SetParametersFromFile

# The sim publishes gps, imu/front, imu/back and e_comms/telemetry
NODES = ["master", "localization", "pathfinder"]


def generate_launch_description():
    params = os.path.join(get_package_share_directory("autonomous_kart"), "params")
    episode = LaunchConfiguration("episode_params")

    return LaunchDescription(
        [
            DeclareLaunchArgument("episode_params"),  # per-episode parameter YAML written by the sim
            GroupAction(
                [
                    SetParametersFromFile(os.path.join(params, "system.yaml")),
                    SetParameter(name="simulation_mode", value=True),
                    SetParameter(name="use_sim_time", value=True),
                ]
                + [
                    Node(
                        package="autonomous_kart",
                        executable=f"{n}_node",
                        name=f"{n}_node",
                        # The sim clocks each node only when it has work due
                        remappings=[("/clock", f"/sim/clock/{n}_node")],
                        parameters=[os.path.join(params, f"{n}.yaml"), episode],
                    )
                    for n in NODES
                ]
            ),
        ]
    )
