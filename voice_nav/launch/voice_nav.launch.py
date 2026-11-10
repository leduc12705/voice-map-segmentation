from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('zones_file', default_value='zones.yaml'),
        DeclareLaunchArgument('mode', default_value='stdin',
                              description="'vosk' (microphone) or 'stdin' (type text)"),
        DeclareLaunchArgument('model_path', default_value=''),
        Node(package='voice_nav', executable='voice_nav_node', output='screen',
             parameters=[{'zones_file': LaunchConfiguration('zones_file')}]),
        Node(package='voice_nav', executable='speech_node', output='screen',
             emulate_tty=True,
             parameters=[{'mode': LaunchConfiguration('mode'),
                          'model_path': LaunchConfiguration('model_path')}]),
    ])
