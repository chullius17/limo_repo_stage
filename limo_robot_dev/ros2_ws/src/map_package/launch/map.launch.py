from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    costmap_mgn = Node(
        package='map_package',
        node_executable='costmap',
        name='costmap_boardwalk',
        output='screen',
        parameters=[{
            'color': 'MAGENTA'
        }]
    )

    costmap_red = Node(
        package='map_package',
        node_executable='costmap',
        name='costmap_solid',
        output='screen',
        parameters=[{
            'color': 'RED'
        }]
    )

    costmap_grn = Node(
        package='map_package',
        node_executable='costmap',
        name='costmap_dashed',
        output='screen',
        parameters=[{
            'color': 'GREEN'
        }]
    )

    mapper_mgn = Node(
        package='map_package',
        node_executable='mapper',
        name='mapper_boardwalk',
        output='screen',
        parameters=[{
            'color': 'MAGENTA'
        }]
    )

    mapper_red = Node(
        package='map_package',
        node_executable='mapper',
        name='mapper_solid',
        output='screen',
        parameters=[{
            'color': 'RED'
        }]
    )

    mapper_grn = Node(
        package='map_package',
        node_executable='mapper',
        name='mapper_dashed',
        output='screen',
        parameters=[{
            'color': 'GREEN'
        }]
    )

    display_node = Node(
        package='map_package',
        node_executable='map_display',
        name='map_displayer',
        output='screen',
    )

    saver_node = Node(
        package='map_package',
        node_executable='map_save',
        name='map_saver',
        output='screen',
    )
    
    return LaunchDescription([
        costmap_mgn,
        costmap_red,
        costmap_grn,
        mapper_mgn,
        mapper_red,
        mapper_grn,
        display_node,
        saver_node
    ])
