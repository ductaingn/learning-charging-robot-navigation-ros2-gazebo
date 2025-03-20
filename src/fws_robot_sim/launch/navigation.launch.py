import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml

def generate_launch_description():
    pkg_fws_robot_sim = get_package_share_directory('fws_robot_sim')
    fws_robot_sim_path = os.path.join(pkg_fws_robot_sim)
    velocity_pub_path = os.path.join(
        get_package_share_directory('velocity_pub')
    )

    current_file_path = os.path.abspath(__file__)
    src_path = os.path.dirname(current_file_path)
    for _ in range(5):
        src_path = os.path.dirname(src_path)
    src_path = os.path.join(src_path, 'src')
    
    with open(os.path.join(velocity_pub_path, 'scripts', 'map_info.yaml')) as file:
        map_info = yaml.safe_load(file)
    map_name = str(map_info['using_map_name'])
    start_x, start_y, start_z, start_roll, start_pitch, start_yaw = list(map_info[map_name]['start'])

    gazebo_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            os.environ['GZ_SIM_RESOURCE_PATH'], ':' +
            os.path.join(fws_robot_sim_path, 'worlds'), ':' + 
            src_path
            ]
        )
    ros_resource_path = SetEnvironmentVariable(
        name='ROS_PACKAGE_PATH',
        value=[
            src_path
        ]
    )

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='navigation.rviz',
        description='RViz config file'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time'
    )

    # Generate path to config file
    interactive_marker_config_file_path = os.path.join(
        get_package_share_directory('interactive_marker_twist_server'),
        'config',
        'linear.yaml'
    )

    # Path to the Slam Toolbox launch file
    nav2_localization_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'localization_launch.py'
    )

    nav2_navigation_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'navigation_launch.py'
    )

    localization_params_path = os.path.join(
        fws_robot_sim_path,
        'config',
        'amcl_localization.yaml'
    )

    navigation_params_path = os.path.join(
        fws_robot_sim_path,
        'config',
        'navigation.yaml'
    )

    map_file_path = os.path.join(
        fws_robot_sim_path,
        'maps',
        f'{map_name}.yaml'
    )

    # Launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_fws_robot_sim, 'config', LaunchConfiguration('rviz_config')])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    interactive_marker_twist_server_node = Node(
        package='interactive_marker_twist_server',
        executable='marker_server',
        name='twist_server_node',
        parameters=[interactive_marker_config_file_path],
        output='screen',
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_localization_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': localization_params_path,
                'map': map_file_path,
        }.items()
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_navigation_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': navigation_params_path,
        }.items()
    )

    publish_initialpose = Node(
        package='velocity_pub',
        executable='publish_initialpose.py',
        arguments=[
            'x', str(start_x),
            'y', str(start_y),
            'z', str(start_z),
            'roll', str(start_roll),
            'pitch', str(start_pitch),
            'yaw', str(start_yaw),
        ],
        output='screen'
    )

    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(gazebo_resource_path)
    launchDescriptionObject.add_action(ros_resource_path)
    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(rviz_config_arg)
    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(rviz_node)
    #launchDescriptionObject.add_action(interactive_marker_twist_server_node)
    launchDescriptionObject.add_action(localization_launch)
    launchDescriptionObject.add_action(navigation_launch)
    launchDescriptionObject.add_action(publish_initialpose)

    return launchDescriptionObject
