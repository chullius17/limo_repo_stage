from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    bridge = Node(
        package='ros1_bridge',
        node_executable='parameter_bridge',
        name='camera_static_bridge',
        arguments=[
            '/camera/color/image_raw/compressed@sensor_msgs/CompressedImage'
        ],
        output='screen'
    )

    return LaunchDescription([
        bridge
    ])