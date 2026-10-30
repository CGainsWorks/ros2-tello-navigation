"""Bring the hardware graph up against the Tello emulator.

Two things, nothing else: the emulator (a simulated aircraft speaking Tello SDK
2.0 on 127.0.0.1:8889, sending state to 8890 and H.264 to 11111) and
``hardware.launch.py``, included unchanged with ``tello_address`` pointed at the
loopback interface. The driver node runs ``UdpLink`` with the PyAV decoder, so
every packet between the graph and the aircraft is a real UDP datagram in the
Tello's format.

    ros2 launch tello_bringup emulator.launch.py auto_arm:=true

The one difference from a real aircraft is ``local_command_port:=0``. A Tello
answers to whichever port a command came from and the driver binds 8889 for that
by default; the emulator already holds 8889 on this machine, so the driver lets
the system choose. The state and video ports are the Tello's own.

Everything the emulator reports is simulated. The fault arguments
(``outage_start``, ``outage_duration``, ``battery``, ``battery_drain``) are how
the launch tests in ``tello_bringup/test`` reach the failsafes; outage times are
seconds after the first take-off.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _repo_file(*parts: str) -> str:
    """Resolve a path in the source tree, as ``hardware.launch.py`` does."""
    share = Path(get_package_share_directory("tello_bringup"))
    for candidate in (share.parents[3] if len(share.parents) > 3 else share,
                      Path(__file__).resolve().parents[4]):
        path = candidate.joinpath(*parts)
        if path.exists():
            return str(path)
    return str(Path(*parts))


def _emulator(context):
    def get(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    arguments = [
        "--map", get("marker_map"),
        "--config", get("aircraft_config"),
        "--mission", get("mission"),
        "--seed", get("seed"),
        "--battery", get("battery"),
        "--battery-drain", get("battery_drain"),
        "--video-rate", get("video_rate"),
        "--render-workers", get("render_workers"),
    ]
    if float(get("outage_duration")) > 0.0:
        arguments += ["--outage", get("outage_start"), get("outage_duration")]
    if get("truth_log"):
        arguments += ["--truth", get("truth_log")]
    if get("event_log"):
        arguments += ["--events", get("event_log")]
    return [
        Node(
            package="tello_core",
            executable="tello_emulator",
            name="tello_emulator",
            output="screen",
            arguments=arguments,
        )
    ]


def generate_launch_description() -> LaunchDescription:
    hardware = str(Path(get_package_share_directory("tello_bringup")) / "launch"
                   / "hardware.launch.py")
    declared = [
        ("marker_map", _repo_file("maps", "marker_map.yaml"), "surveyed marker layout"),
        ("mission", _repo_file("missions", "hop.yaml"), "waypoint mission to fly"),
        ("aircraft_config",
         _repo_file("src", "tello_bringup", "config", "aircraft.yaml"),
         "airframe, camera mount, rc scaling, geofence"),
        ("auto_arm", "false", "take off once localised, without the arm service"),
        ("image_encoding", "mono8", "the emulator's camera is monochrome"),
        ("seed", "1", "emulator random seed"),
        ("battery", "1.0", "starting charge, 0 to 1 (simulated)"),
        ("battery_drain", "1.0", "battery drain multiplier (simulated)"),
        ("video_rate", "30.0", "target frame rate of the emulated camera"),
        ("render_workers", "0", "processes rendering the camera in parallel; 0 renders in-thread"),
        ("outage_start", "0.0", "seconds after take-off at which the link drops"),
        ("outage_duration", "0.0", "seconds the link stays down; 0 for none"),
        ("truth_log", "", "CSV of simulated ground truth; empty for none"),
        ("event_log", "", "JSON lines of every datagram and reply; empty for none"),
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument(name, default_value=default, description=description)
            for name, default, description in declared
        ]
        + [
            OpaqueFunction(function=_emulator),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(hardware),
                launch_arguments={
                    "tello_address": "127.0.0.1",
                    "local_command_port": "0",
                    "marker_map": LaunchConfiguration("marker_map"),
                    "mission": LaunchConfiguration("mission"),
                    "aircraft_config": LaunchConfiguration("aircraft_config"),
                    "auto_arm": LaunchConfiguration("auto_arm"),
                    "image_encoding": LaunchConfiguration("image_encoding"),
                }.items(),
            ),
        ]
    )
