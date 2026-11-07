# https://github.com/EVC-Purdue/A27/issues/11
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    kart = os.path.join(get_package_share_directory("autonomous_kart"), "launch", "bringup_kart.launch.py")
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(kart),
                launch_arguments={"simulation_mode": "true"}.items(),
            ),
        ]
    )
