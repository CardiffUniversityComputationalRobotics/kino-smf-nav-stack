from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ! PACKAGES DIR
    kino_smf_move_base_planning_dir = FindPackageShare(
        package="kino_smf_move_base_planning"
    ).find("kino_smf_move_base_planning")

    # ! PARAMETERS
    planning_config_file_param = DeclareLaunchArgument(
        "planning_config_file",
        default_value=[
            kino_smf_move_base_planning_dir,
            "/config/kino_smf_move_base_planning.yaml",
        ],
        description="Configuration file for planning.",
    )

    # ! NODES
    planning_node = Node(
        package="kino_smf_move_base_planning",
        executable="kino_smf_move_base_planner",
        name="kino_smf_move_base_planner",
        output="screen",
        parameters=[LaunchConfiguration("planning_config_file")],
        on_exit=Shutdown(),
        emulate_tty=True,
    )

    # ! LAUNCH DESCRIPTION DECLARATION
    ld = LaunchDescription()

    # ! PARAMETERS
    ld.add_action(planning_config_file_param)

    # ! NODES
    ld.add_action(planning_node)

    return ld
