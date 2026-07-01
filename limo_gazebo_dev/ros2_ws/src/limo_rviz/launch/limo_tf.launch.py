import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    use_sim_time_param = {'use_sim_time': True}

    static_tf_map_to_odom = Node(
        package='tf2_ros',
        node_executable='static_transform_publisher',
        node_name='static_tf_map_to_odom',
        arguments=['0', '0', '0', 
                   '0', '0', '0', 
                   'map', 
                   'odom'],
        output='screen',
        parameters=[use_sim_time_param]
    )

    static_tf_bs_ftprint_to_bs_link = Node(
        package='tf2_ros',
        node_executable='static_transform_publisher',
        node_name='static_tf_base_footprint_to_base_link',
        arguments=['0', '0', '0.15', 
                   '0', '0', '0',
                    'base_footprint',
                    'base_link'],
        output='screen',
        parameters=[use_sim_time_param]
    )

    map_to_odom_delayed = TimerAction(
        period=1.0,
        actions=[static_tf_map_to_odom]
    )

    base_tf_delayed = TimerAction(
        period=1.5,
        actions=[static_tf_bs_ftprint_to_bs_link]
    )

    return LaunchDescription([
        map_to_odom_delayed,
        base_tf_delayed
    ])

