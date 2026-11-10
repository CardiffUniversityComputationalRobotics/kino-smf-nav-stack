from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ! PACKAGES DIR
    kino_smf_move_base_mapping_dir = FindPackageShare(
        package="kino_smf_move_base_mapping"
    ).find("kino_smf_move_base_mapping")

    # ! PARAMETERS
    mapping_config_file_param = DeclareLaunchArgument(
        "mapping_config_file",
        default_value=[
            kino_smf_move_base_mapping_dir,
            "/config/kino_smf_move_base_mapping.yaml",
        ],
        description="Configuration file for mapping.",
    )

    # ! NODES
    mapping_node = Node(
        package="kino_smf_move_base_mapping",
        executable="kino_smf_move_base_mapper",
        name="kino_smf_move_base_mapper",
        output="screen",
        parameters=[LaunchConfiguration("mapping_config_file")],
        on_exit=Shutdown(),
    )

    # ! LAUNCH DESCRIPTION DECLARATION
    ld = LaunchDescription()

    # ! PARAMETERS
    ld.add_action(mapping_config_file_param)

    # ! NODES
    ld.add_action(mapping_node)

    return ld
