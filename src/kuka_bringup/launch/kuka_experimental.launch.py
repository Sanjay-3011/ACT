import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Package share directories
    kuka_description_share = get_package_share_directory('kuka_description')
    kuka_gazebo_share = get_package_share_directory('kuka_gazebo')

    # Include Gazebo simulation launch (experimental version)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kuka_gazebo_share, 'launch', 'gazebo_experimental.launch.py')
        )
    )

    # RViz 2 Node (configured to use simulation time)
    rviz_config_path = os.path.join(kuka_description_share, 'config', 'urdf.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}]
    )

    nodes_to_launch = [gazebo_launch, rviz_node]

    return LaunchDescription(nodes_to_launch)

