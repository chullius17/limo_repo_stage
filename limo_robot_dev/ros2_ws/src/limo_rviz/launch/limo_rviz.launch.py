from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def find_ros2_ws(start: Path):
    for parent in [start] + list(start.parents):
        if parent.name == "ros2_ws":
            return parent
    return None


def generate_launch_description():

    ws = find_ros2_ws(Path(__file__).resolve())
    if ws is None:
        raise RuntimeError("Workspace ros2_ws non trovato")

    default_map_path = str(ws / "src" / "ros2_maps" / "limo_map.yaml")

    map_yaml_file = LaunchConfiguration('map_yaml_file')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map_yaml_file',
        default_value=default_map_path,
        description='Path to map yaml'
    )

    use_sim_time = {'use_sim_time': True}

    # =========================
    # TF STATICI
    # =========================

    tf_map_to_odom = Node(
        package='tf2_ros',
        node_executable='static_transform_publisher',
        node_name='static_tf_map_to_odom',
        arguments=['0', '0', '0', 
                   '0', '0', '0', 
                   'map', 
                   'odom'],
        output='screen',
        parameters=[use_sim_time]
    )

    tf_base = Node(
        package='tf2_ros',
        node_executable='static_transform_publisher',
        node_name='static_tf_base_footprint_to_base_link',
        arguments=['0', '0', '0.15', 
                   '0', '0', '0',
                    'base_footprint',
                    'base_link'],
        output='screen',
        parameters=[use_sim_time]
    )

    tf_launch = TimerAction(
        period=0.5,
        actions=[tf_map_to_odom, tf_base]
    )

    # =========================
    # MAP SERVER (Nav2)
    # =========================

    map_server = Node(
        package='nav2_map_server',
        node_executable='map_server',
        node_name='map_server',
        output='screen',
        parameters=[
            {'yaml_filename': map_yaml_file},
            {'frame_id': 'map'},
            use_sim_time
        ]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        node_executable='lifecycle_manager',
        node_name='lifecycle_manager_map_server',
        output='screen',
        parameters=[
            use_sim_time,
            {'autostart': True},
            {'node_names': ['map_server']}
        ]
    )

    map_launch = TimerAction(
        period=2.0,
        actions=[map_server]
    )

    lifecycle_launch = TimerAction(
        period=3.5,
        actions=[lifecycle_manager]
    )

    # =========================
    # RVIZ
    # =========================

    rviz = Node(
        package='rviz2',
        node_executable='rviz2',
        node_name='rviz2',
        output='screen',
        parameters=[use_sim_time]
    )

    rviz_launch = TimerAction(
        period=5.0,
        actions=[rviz]
    )

    # =========================
    # LAUNCH FINAL
    # =========================

    return LaunchDescription([
        declare_map_yaml_cmd,

        tf_launch,
        map_launch,
        lifecycle_launch,
        rviz_launch
    ])