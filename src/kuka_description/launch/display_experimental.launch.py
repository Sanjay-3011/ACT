import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Get package directories
    kuka_description_dir = get_package_share_directory('kuka_description')

    # Define default file paths for the experimental model
    default_model_path = os.path.join(kuka_description_dir, 'urdf', 'kr16_2_gazebo.urdf.xacro')
    default_rviz_config_path = os.path.join(kuka_description_dir, 'config', 'urdf.rviz')

    # Launch Configurations
    model = LaunchConfiguration('model')
    rviz_config = LaunchConfiguration('rvizconfig')

    # Robot State Publisher Node (dynamically processes xacro)
    robot_description_content = ParameterValue(Command(['xacro ', model]), value_type=str)

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # Joint State Publisher Node
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )

    # RViz 2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            name='model',
            default_value=default_model_path,
            description='Absolute path to robot xacro/urdf file'
        ),
        DeclareLaunchArgument(
            name='rvizconfig',
            default_value=default_rviz_config_path,
            description='Absolute path to rviz config file'
        ),
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node
    ])
