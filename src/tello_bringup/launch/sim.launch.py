"""Bring the whole stack up against the simulator.

Four nodes, one parameter file, no hardware. The only difference between this and
``hardware.launch.py`` is the driver's ``link_type``, which is the point of the
``TelloLink`` interface: everything above the driver cannot tell which it has.

    ros2 launch tello_bringup sim.launch.py
    ros2 service call /tello_mission/arm std_srvs/srv/Trigger

Note that a simulated flight inside a ROS graph runs in real time and is not
bit-for-bit reproducible, because the simulator is stepped by however much wall
clock has passed between polls. The reproducible numbers come from the offline
runner, ``scripts/run_mission.py``. See ``docs/ROS2.md``.
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
        description="take off as soon as the first estimate arrives. Fine in "
                    "simulation, a bad idea with an aircraft on the bench.",
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
            Node(
                package="tello_driver",
                executable="driver_node",
                name="tello_driver",
                output="screen",
                parameters=[parameters, {**common, "link_type": "sim"}],
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
                    },
                ],
            ),
        ]
    )
