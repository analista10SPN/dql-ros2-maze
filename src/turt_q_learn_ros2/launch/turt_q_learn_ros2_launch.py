from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to your package directory
    package_dir = get_package_share_directory('turt_q_learn_ros2')

    # Define the path to the .sdf model file
    sdf_model_file = os.path.join(package_dir, 'models', 'brick_world', 'model.sdf')

    # Launch Gazebo with an empty world
    world_file = os.path.join(package_dir, 'worlds', 'brick.world')

    return LaunchDescription([
        # Launch Gazebo with an empty world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file],
            output='screen'
        ),
        # Use spawn_entity node to spawn the SDF model into Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-file', sdf_model_file, '-sdf', '-entity', 'brick_model'],
            output='screen'
        ),
        # Launch your custom pathfinder node
        Node(
            package='turt_q_learn_ros2',
            executable='turt_q_learn_hypers.py',
            name='turt_q_learn_hypers',
            output='screen',
        ),
    ])
