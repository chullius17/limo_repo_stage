from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    center_lanes_network = Node(
        package='traj_package',
        node_executable='routes',
        name='center_lanes_network',
        output='screen',
        parameters=[{
            'flag': 'CENTER_ROAD'
        }]
    )

    open_spaces_network = Node(
        package='traj_package',
        node_executable='routes',
        name='open_spaces_network',
        output='screen',
        parameters=[{
            'flag': 'OPEN'
        }]
    )

    network_combination = Node(
        package='traj_package',
        node_executable='route_combinator',
        name='network_combination',
        output='screen',
    )

    astar = Node(
        package='traj_package',
        node_executable='astar',
        name='astar_server',
        output='screen',
    )

    coordinator = Node(
        package='traj_package',
        node_executable='coordinator',
        name='mission_coordinator',
        output='screen',
    )

    return LaunchDescription([
        center_lanes_network,
        open_spaces_network,
        network_combination,
        astar,
        #coordinator
    ])