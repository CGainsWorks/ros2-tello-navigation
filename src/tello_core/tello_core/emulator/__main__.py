"""Run the Tello emulator until interrupted.

    python -m tello_core.emulator --map maps/marker_map.yaml --mission missions/hop.yaml
    ros2 run tello_core tello_emulator --map ... --mission ...

Then point a driver at 127.0.0.1:8889. Everything it reports is simulated.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

import numpy as np

from ..camera import CameraModel
from ..config import AircraftConfig
from ..localisation.markers import MarkerMap
from ..sim.world import Room
from .server import EmulatorFaults, EmulatorSettings, TelloEmulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tello_emulator",
        description="A simulated DJI Tello speaking SDK 2.0 over UDP. Every output is simulated.",
    )
    parser.add_argument("--map", dest="marker_map", required=True, help="marker map YAML")
    parser.add_argument("--config", default=None, help="aircraft configuration YAML")
    parser.add_argument(
        "--mission",
        default=None,
        help="mission YAML. Used only to choose the launch pose when --start is not given: "
        "the first waypoint's x and y, facing the markers that can be seen from there.",
    )
    parser.add_argument(
        "--start", nargs=3, type=float, default=None, metavar=("X", "Y", "YAW_DEG")
    )
    parser.add_argument("--bind", default="127.0.0.1", help="address the command port binds")
    parser.add_argument("--command-port", type=int, default=8889)
    parser.add_argument("--state-port", type=int, default=8890)
    parser.add_argument("--video-port", type=int, default=11111)
    parser.add_argument("--video-rate", type=float, default=30.0)
    parser.add_argument(
        "--render-workers",
        type=int,
        default=0,
        help="processes rendering frames in parallel; 0 renders in the video thread",
    )
    parser.add_argument("--bitrate", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--battery", type=float, default=1.0, help="starting charge, 0 to 1")
    parser.add_argument("--battery-drain", type=float, default=1.0, help="drain multiplier")
    parser.add_argument(
        "--outage",
        nargs=2,
        type=float,
        action="append",
        default=[],
        metavar=("START", "DURATION"),
        help="total link outage, seconds after the first take-off. Repeatable.",
    )
    parser.add_argument(
        "--video-outage",
        nargs=2,
        type=float,
        action="append",
        default=[],
        metavar=("START", "DURATION"),
        help="video-only outage, seconds after the first take-off. Repeatable.",
    )
    parser.add_argument("--state-loss", type=float, default=0.0)
    parser.add_argument("--frame-loss", type=float, default=0.0)
    parser.add_argument("--truth", default=None, help="write simulated ground truth CSV here")
    parser.add_argument("--events", default=None, help="write the event log (JSON lines) here")
    return parser


def launch_pose(marker_map: MarkerMap, config: AircraftConfig, mission_path: str):
    """The same launch pose ``scripts/run_mission.py`` uses for a mission."""
    from ..control.mission import Mission
    from ..control.waypoint import heading_for_markers

    mission = Mission.from_yaml(mission_path)
    first = mission.waypoints[0].position
    position = np.array([float(first[0]), float(first[1]), 0.03])
    yaw = heading_for_markers(position, marker_map, config, CameraModel.tello_nominal())
    return float(first[0]), float(first[1]), float(yaw)


def main(argv=None) -> int:
    # ROS 2 launch appends --ros-args to every executable it starts. The
    # emulator is not a ROS node and has no use for them.
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    args = build_parser().parse_args(argv)

    marker_map = MarkerMap.from_yaml(args.marker_map)
    config = AircraftConfig.from_yaml(args.config) if args.config else AircraftConfig()
    if args.start is not None:
        x, y, yaw = args.start[0], args.start[1], float(np.deg2rad(args.start[2]))
    elif args.mission:
        x, y, yaw = launch_pose(marker_map, config, args.mission)
    else:
        x, y, yaw = 2.0, 1.5, 0.0

    for path in (args.truth, args.events):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    settings = EmulatorSettings(
        bind_address=args.bind,
        command_port=args.command_port,
        state_port=args.state_port,
        video_port=args.video_port,
        video_rate=args.video_rate,
        render_workers=args.render_workers,
        bitrate=args.bitrate,
        seed=args.seed,
        battery=args.battery,
        battery_drain=args.battery_drain,
        faults=EmulatorFaults(
            outages=tuple((float(s), float(d)) for s, d in args.outage),
            video_outages=tuple((float(s), float(d)) for s, d in args.video_outage),
            state_loss=args.state_loss,
            frame_loss=args.frame_loss,
        ),
        truth_path=args.truth,
        events_path=args.events,
    )
    emulator = TelloEmulator(
        Room.from_marker_map(marker_map),
        config,
        CameraModel.tello_nominal(),
        settings,
        start_position=np.array([x, y, 0.03]),
        start_yaw=yaw,
    )

    stop = threading.Event()

    def _stop(signum, frame) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    emulator.start()
    host, port = emulator.command_address
    print(
        f"tello emulator (simulated) on {host}:{port}, start ({x:.2f}, {y:.2f}) "
        f"yaw {np.rad2deg(yaw):.0f} deg, battery {args.battery:.2f}",
        flush=True,
    )
    try:
        while not stop.wait(0.5):
            if not emulator.running:
                print("emulator thread failed, stopping", file=sys.stderr, flush=True)
                emulator.close()
                return 1
    finally:
        emulator.close()
    print("tello emulator stopped: " + json.dumps(emulator.stats()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
