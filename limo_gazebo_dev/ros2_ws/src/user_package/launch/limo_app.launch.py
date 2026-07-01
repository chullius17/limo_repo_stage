from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    trajectory_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('traj_package'),
                'launch',
                'trajectory.launch.py'
            )
        )
    )

    control_node = Node(
        package='limo_controller', 
        node_executable='controller',
        name='controller',
        output='screen'
    )

    user_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('user_package'),
                'launch',
                'user.launch.py'
            )
        )
    )

    return LaunchDescription([
        trajectory_launch,
        control_node,
        user_launch
    ])