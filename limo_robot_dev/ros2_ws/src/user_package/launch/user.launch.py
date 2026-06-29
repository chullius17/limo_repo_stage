from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    visualizer = Node(
        package='user_package',
        node_executable='visualizer',
        name='visualizer',
        output='screen'
    )

    ctrl_viz = Node(
        package='user_package',
        node_executable='ctrl_viz',
        name='ctrl_viz',
        output='screen'
    )

    user_srv = Node(
        package='user_package',
        node_executable='user',
        name='user_server',
        output='screen'
    )

    gui = Node(
        package='user_package',
        node_executable='gui',
        name='gui',
        output='screen'
    )

    return LaunchDescription([
        visualizer,
        ctrl_viz,
        user_srv,
        gui
    ])