import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Get package directories
    kuka_description_share = get_package_share_directory('kuka_description')
    kuka_kr16_support_share = get_package_share_directory('kuka_kr16_support')
    kuka_gazebo_share = get_package_share_directory('kuka_gazebo')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    # Set Gazebo environment variables dynamically
    paths_to_add = [
        os.path.dirname(kuka_description_share),
        os.path.dirname(kuka_kr16_support_share)
    ]
    existing_paths = os.environ.get('GAZEBO_MODEL_PATH', '')
    all_paths = [p for p in existing_paths.split(':') if p]
    for p in paths_to_add:
        if p not in all_paths:
            all_paths.append(p)
    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(all_paths)

    # Bypass online model database lookup for instant startup
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

    # Define file paths
    model_path = os.path.join(kuka_description_share, 'urdf', 'kr16_2_gazebo.urdf.xacro')
    world_path = os.path.join(kuka_gazebo_share, 'worlds', 'empty.world')

    # Launch Configurations
    gui = LaunchConfiguration('gui')

    # Declare Launch Arguments
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Start Gazebo client GUI (gzclient)'
    )

    # Robot State Publisher Node (processes xacro with sim time)
    robot_description_content = ParameterValue(Command(['xacro ', model_path]), value_type=str)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True
        }]
    )

    # Launch Gazebo Server and Client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'gui': gui
        }.items()
    )

    # Spawn Robot Entity in Gazebo (at z=0.0 since kr16_2 rests exactly on z=0)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'kr16_2_gazebo'
        ],
        output='screen'
    )

    # Controller Spawners
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output='screen'
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output='screen'
    )

    # Sequential execution: Spawn joint broadcaster after robot is spawned in Gazebo
    delay_broadcaster_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    # Spawn arm controller after joint state broadcaster is loaded
    delay_arm_controller_after_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    return LaunchDescription([
        gui_arg,
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        delay_broadcaster_after_spawn,
        delay_arm_controller_after_broadcaster
    ])
