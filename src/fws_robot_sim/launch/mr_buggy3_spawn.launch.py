import os
import xacro
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.actions import RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml

def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    fws_robot_description_path = os.path.join(
        get_package_share_directory('fws_robot_description'))
    
    fws_robot_sim_path = os.path.join(
        get_package_share_directory('fws_robot_sim'))
    
    velocity_pub_path = os.path.join(
        get_package_share_directory('velocity_pub')
    )
    
    with open(os.path.join(velocity_pub_path,'scripts','map_info.yaml')) as file:
        map_info = yaml.safe_load(file)
    map_name = str(map_info['using_map_name'])
    start_x, start_y, start_z, start_roll, start_pitch, start_yaw = list(map_info[map_name]['start'])

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

    arguments = LaunchDescription([
                DeclareLaunchArgument('world', default_value=map_info['using_map_name'],
                          description='Gz sim World'),
           ]
    )

    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('ros_gz_sim'), 'launch'), '/gz_sim.launch.py']),
                launch_arguments=[
                    ('gz_args', [LaunchConfiguration('world'),
                                 '.sdf',
                                 ' -v 4',
                                 ' -r']
                    )
                ]
             )

    mr_buggy3_description_path = os.path.join(
        get_package_share_directory('mr_buggy3_description'),'MR-Buggy3','model.sdf')
    
    with open(mr_buggy3_description_path, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )
    
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-file', mr_buggy3_description_path ,
                   '-x', str(start_x),
                   '-y', str(start_y),
                   '-z', str(start_z),
                   '-R', str(start_roll),
                   '-P', str(start_pitch),
                   '-Y', str(start_yaw),
                   '-name', 'saye',
                   '-allow_renaming', 'false'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Bridge
    bridge = Node(
    # Bridge ROS topics and Gazebo messages for establishing communication
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(fws_robot_sim_path, 'config', 'ros_gz_bridge.yaml'),
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    world_control_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/world/empty/control@ros_gz_interfaces/srv/ControlWorld'],
        parameters=[{'use_sim_time': use_sim_time}]
    )


    rviz = Node(
       package='rviz2',
       executable='rviz2',
       arguments=['-d', os.path.join(fws_robot_sim_path, 'config', 'rviz.rviz')]
    )

    return LaunchDescription([
        gazebo_resource_path,
        ros_resource_path,
        arguments,
        gazebo,
        robot_state_publisher,
        gz_spawn_entity,
        bridge,
        world_control_bridge,
        rviz
    ])