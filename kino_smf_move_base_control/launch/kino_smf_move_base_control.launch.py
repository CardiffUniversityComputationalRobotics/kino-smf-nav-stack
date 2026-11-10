from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ! PACKAGES DIR
    kino_smf_move_base_control_dir = FindPackageShare(
        package="kino_smf_move_base_control"
    ).find("kino_smf_move_base_control")

    # ! PARAMETERS
    control_config_file_param = DeclareLaunchArgument(
        "control_config_file",
        default_value=[
            kino_smf_move_base_control_dir,
            "/config/kino_smf_move_base_control.yaml",
        ],
        description="Configuration file for control.",
    )

    # ! NODES
    control_node = Node(
        package="kino_smf_move_base_control",
        executable="base_controller",
        name="kino_smf_move_base_controller",
        output="screen",
        parameters=[LaunchConfiguration("control_config_file")],
        on_exit=Shutdown(),
        emulate_tty=True,
    )

    # ! LAUNCH DESCRIPTION DECLARATION
    ld = LaunchDescription()

    # ! PARAMETERS
    ld.add_action(control_config_file_param)

    # ! NODES
    ld.add_action(control_node)

    return ld
