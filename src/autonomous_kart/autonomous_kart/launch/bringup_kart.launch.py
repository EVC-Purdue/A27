# https://github.com/EVC-Purdue/A27/issues/11
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter, SetParametersFromFile

NODES = [
    "master",
    "camera",
    "gps",
    "imu",
    "lidar",
    "e_comms",
    "mechanical_brake",
    "localization",
    "perception",
    "slam",
    "gs3d",
    "pathfinder",
]


def generate_launch_description():
    pkg_share = get_package_share_directory("autonomous_kart")
    params = os.path.join(pkg_share, "params")

    return LaunchDescription(
        [
            DeclareLaunchArgument("simulation_mode", default_value="false"),
            GroupAction(
                [
                    SetParametersFromFile(os.path.join(params, "system.yaml")),
                    SetParameter(name="simulation_mode", value=LaunchConfiguration("simulation_mode")),
                ]
                + [
                    Node(
                        package="autonomous_kart",
                        executable=f"{n}_node",
                        name=f"{n}_node",
                        parameters=[os.path.join(params, f"{n}.yaml")],
                    )
                    for n in NODES
                ]
            ),
        ]
    )
