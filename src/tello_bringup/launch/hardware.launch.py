"""Bring the whole stack up against a real DJI Tello.

The same graph as ``sim.launch.py`` with one parameter changed, which is the point
of the ``TelloLink`` interface.

READ ``docs/HARDWARE.md`` FIRST.

The UDP driver in this repository has never exchanged a packet with a Tello. It is
written from DJI's published SDK 2.0 document and exercised against a loopback
socket and against the emulator. The driver node logs a warning saying so
every time it starts. Several constants it depends on, the camera intrinsics, the
mount offset, the rc channel scaling and the units of the reported acceleration,
are unverified and ``docs/HARDWARE.md`` lists them with the ten-minute checks
that would settle each.

The video is decoded with PyAV (``pip install av``).

    ros2 launch tello_bringup hardware.launch.py
    ros2 service call /tello_mission/arm std_srvs/srv/Trigger

``emulator.launch.py`` includes this file unchanged and overrides only
``tello_address`` and ``local_command_port``, which is how the launch tests fly
exactly this graph against the emulator.

Fly tethered, with the geofence tightened and the hold timeout shortened and keep
a hand on ``/tello_mission/abort``.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _repo_file(*parts: str) -> str:
    """Resolve a path in the source tree rather than the install space.

    The marker map, the missions and the aircraft configuration are inputs to the
    system rather than parts of it: they are edited between runs and they live at
    the repository root, not inside a package. Copying them into every package's
    share directory would mean editing a map and launching the stale one.
    """
    share = Path(get_package_share_directory("tello_bringup"))
    # share/tello_bringup -> ... -> repository root, when built with colcon from
    # the repository root. Falls back to the source tree beside this file.
    for candidate in (share.parents[3] if len(share.parents) > 3 else share,
                      Path(__file__).resolve().parents[4]):
        path = candidate.joinpath(*parts)
        if path.exists():
            return str(path)
    return str(Path(*parts))


def generate_launch_description() -> LaunchDescription:
    parameters = PathJoinSubstitution(
        [FindPackageShare("tello_bringup"), "config", "nodes.yaml"]
    )

    marker_map = DeclareLaunchArgument(
        "marker_map",
        default_value=_repo_file("maps", "marker_map.yaml"),
        description="surveyed marker layout",
    )
    mission = DeclareLaunchArgument(
        "mission",
        default_value=_repo_file("missions", "perimeter.yaml"),
        description="waypoint mission to fly",
    )
    aircraft = DeclareLaunchArgument(
        "aircraft_config",
        default_value=_repo_file("src", "tello_bringup", "config", "aircraft.yaml"),
        description="airframe, camera mount, rc scaling, geofence",
    )
    auto_arm = DeclareLaunchArgument(
        "auto_arm",
        default_value="false",
        description="must stay false with an aircraft on the bench. Arm by hand.",
    )
    address = DeclareLaunchArgument(
        "tello_address",
        default_value="192.168.10.1",
        description="the aircraft's address on its own access point",
    )
    local_port = DeclareLaunchArgument(
        "local_command_port",
        default_value="8889",
        description="local port for commands and their replies; 0 lets the system choose",
    )
    encoding = DeclareLaunchArgument(
        "image_encoding",
        default_value="bgr8",
        description="bgr8, as decoded, or mono8 for a third of the bytes between nodes",
    )

    common = {
        "marker_map": LaunchConfiguration("marker_map"),
        "aircraft_config": LaunchConfiguration("aircraft_config"),
    }

    return LaunchDescription(
        [
            marker_map,
            mission,
            aircraft,
            auto_arm,
            address,
            local_port,
            encoding,
            Node(
                package="tello_driver",
                executable="driver_node",
                name="tello_driver",
                output="screen",
                parameters=[
                    parameters,
                    {
                        **common,
                        "link_type": "udp",
                        "tello_address": LaunchConfiguration("tello_address"),
                        "local_command_port": LaunchConfiguration("local_command_port"),
                        "image_encoding": LaunchConfiguration("image_encoding"),
                    },
                ],
            ),
            Node(
                package="tello_localisation",
                executable="marker_node",
                name="tello_marker_localisation",
                output="screen",
                parameters=[parameters, common],
            ),
            Node(
                package="tello_localisation",
                executable="fusion_node",
                name="tello_fusion",
                output="screen",
                parameters=[parameters],
            ),
            Node(
                package="tello_control",
                executable="mission_node",
                name="tello_mission",
                output="screen",
                parameters=[
                    parameters,
                    {
                        **common,
                        "mission": LaunchConfiguration("mission"),
                        "auto_arm": LaunchConfiguration("auto_arm"),
                        "control_config": _repo_file(
                            "src", "tello_bringup", "config", "control.yaml"
                        ),
                    },
                ],
            ),
        ]
    )
