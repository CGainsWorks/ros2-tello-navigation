#!/usr/bin/env python3
"""Fly one mission in the simulator and write the flight log.

The entry point everything else goes through. ``make demo`` calls it, CI calls
it and ``scripts/run_failsafe_cases.sh`` calls it once per failure case, so there
is one code path from a mission file to a results directory and no chance of the
documented command doing something different from the one that produced the
numbers.

Examples
--------
::

    python scripts/run_mission.py --mission missions/perimeter.yaml
    python scripts/run_mission.py --mission missions/perimeter.yaml \\
        --fault link-loss --out results/link_loss
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.control.mission import Mission, MissionRunner, MissionSettings
from tello_core.control.pid import ControlGains
from tello_core.eval.metrics import evaluate
from tello_core.localisation.markers import MarkerMap
from tello_core.sim.dynamics import DynamicsSettings
from tello_core.sim.link import LinkFaults, SimLink, SimSettings
from tello_core.sim.world import Room

# The named fault schedules. Named rather than free-form because a failsafe
# result has to be reproducible and quotable: "the aircraft landed 1.4 s after
# the link dropped" means nothing unless the reader can run the same drop.
FAULTS = {
    "none": LinkFaults(state_loss=0.0, frame_loss=0.0, malformed_state=0.0),
    "nominal": LinkFaults(),
    "link-loss": LinkFaults(outages=((18.0, 25.0),)),
    # The same failure, timed for a short mission. The eighteen second drop
    # above lands in the middle of the perimeter flight; on the two waypoint
    # hop the aircraft is already down by then, so CI would be asserting that
    # a link loss lands the aircraft while flying a mission that never loses
    # the link. Four seconds is after take-off and well before the hop is
    # over, leaving room for the one second link timeout to expire.
    #
    # It comes back at eleven seconds for the same reason the schedule above
    # does. With the link still down the aircraft is landing itself on its own
    # dead-man timer and the controller, being silent and deaf, cannot observe
    # the touchdown; the log then ends in LAND rather than LANDED and says
    # nothing about whether the aircraft reached the ground. Restoring the link
    # after the response has committed is what makes the landing observable.
    "link-loss-early": LinkFaults(outages=((4.0, 11.0),)),
    # Video dies twelve seconds in and never returns, with the telemetry still
    # arriving. Scheduled rather than "all frames lost from the start", because
    # from the start the aircraft simply refuses to take off, which is correct and
    # tests nothing: the interesting case is losing sight of the markers while
    # airborne.
    "video-loss": LinkFaults(video_outages=((12.0, 300.0),)),
    "lossy": LinkFaults(state_loss=0.20, frame_loss=0.25, malformed_state=0.05),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mission", default="missions/perimeter.yaml")
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument(
        "--config", default="src/tello_bringup/config/aircraft.yaml", help="aircraft configuration"
    )
    parser.add_argument("--out", default="results/latest", help="directory for the flight log")
    parser.add_argument("--stem", default="flight", help="filename stem inside --out")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--fault", choices=sorted(FAULTS), default="nominal", help="link fault schedule"
    )
    parser.add_argument("--control-rate", type=float, default=20.0)
    parser.add_argument(
        "--gains",
        default="src/tello_bringup/config/control.yaml",
        help="controller gains. The same file the ROS 2 mission node reads.",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="run detection on one frame in N. Raising it trades localisation rate for speed.",
    )
    parser.add_argument("--max-flight-time", type=float, default=180.0)
    parser.add_argument(
        "--battery",
        type=float,
        default=1.0,
        help="starting battery fraction. Lower it to reach the low-battery failsafe quickly.",
    )
    parser.add_argument(
        "--battery-drain",
        type=float,
        default=1.0,
        help="multiply the battery drain rate. The aircraft's real endurance is "
             "about eight minutes and the reference mission is thirty seconds, so "
             "reaching the in-flight battery failsafes needs either a very long "
             "flight or a faster drain. The multiplier is recorded in the results "
             "metadata so a battery figure can never be read as if it came from a "
             "real discharge curve.",
    )
    parser.add_argument(
        "--no-vo", action="store_true", help="disable the visual-odometry front end"
    )
    parser.add_argument(
        "--start",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "YAW_DEG"),
        help="launch pose. Defaults to the first waypoint's x and y, facing the nearest markers.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
    config = AircraftConfig.from_yaml(REPO_ROOT / args.config)
    mission = Mission.from_yaml(REPO_ROOT / args.mission)
    camera = CameraModel.tello_nominal()

    problems = mission.validate(config, marker_map)
    if problems:
        print("mission is not flyable:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    warnings = marker_map.check_geometry()
    if warnings and not args.quiet:
        for warning in warnings:
            print(f"marker map: {warning}")

    room = Room.from_marker_map(marker_map)

    if args.start is not None:
        start_x, start_y, start_yaw_deg = args.start
    else:
        first = mission.waypoints[0].position
        start_x, start_y = float(first[0]), float(first[1])
        start_yaw_deg = _launch_heading(
            np.array([start_x, start_y, 0.03]), marker_map, config, camera
        )

    nominal = DynamicsSettings()
    dynamics = replace(
        nominal, hover_current=nominal.hover_current * max(args.battery_drain, 0.0)
    )
    link = SimLink(
        room,
        config,
        camera,
        SimSettings(seed=args.seed, faults=FAULTS[args.fault]),
        dynamics_settings=dynamics,
        start_position=np.array([start_x, start_y, 0.03]),
        start_yaw=float(np.deg2rad(start_yaw_deg)),
    )
    link.dynamics.state.battery = float(np.clip(args.battery, 0.0, 1.0))

    gains_path = REPO_ROOT / args.gains
    settings = MissionSettings(
        control_rate=args.control_rate,
        detect_every=max(1, args.detect_every),
        vo_enabled=not args.no_vo,
        max_flight_time=args.max_flight_time,
        gains=ControlGains.from_yaml(gains_path) if gains_path.is_file() else ControlGains(),
    )
    runner = MissionRunner(link, mission, marker_map, config, camera, settings)

    started = time.monotonic()
    log = runner.run()
    wall = time.monotonic() - started

    log.metadata["gains_file"] = str(args.gains)
    log.metadata["fault_schedule"] = args.fault
    log.metadata["seed"] = args.seed
    log.metadata["start_pose"] = [start_x, start_y, float(start_yaw_deg)]
    log.metadata["start_battery"] = float(args.battery)
    log.metadata["battery_drain_multiplier"] = float(args.battery_drain)
    log.metadata["wall_seconds"] = round(wall, 1)
    log.metadata["frames_rendered"] = int(getattr(link, "frames_rendered", 0))

    out = REPO_ROOT / args.out
    written = log.write(out, stem=args.stem)
    metrics = evaluate(log)
    with open(out / f"{args.stem}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    if not args.quiet:
        _report(log, metrics, written, wall)
    return 0


def _launch_heading(position, marker_map, config, camera) -> float:
    """Point the aircraft at whatever it can see from the launch point.

    Not a convenience: the aircraft will not take off until it has a fix, so a
    launch heading that faces a blank wall means a flight that never happens.
    On real hardware this is the step where somebody turns the Tello round on
    the floor before pressing go.
    """
    from tello_core.control.waypoint import heading_for_markers

    return float(np.rad2deg(heading_for_markers(position, marker_map, config, camera)))


def _report(log, metrics, written, wall: float) -> None:
    metadata = log.metadata
    duration = log.rows[-1].time_s if log.rows else 0.0
    print()
    print(f"mission        {metadata.get('mission')} ({metadata.get('fault_schedule')} link)")
    print(f"flight         {duration:.1f} s simulated in {wall:.1f} s of wall clock")
    print(f"final state    {metadata.get('final_state')} ({metadata.get('final_cause')})")
    if metadata.get("aborted"):
        print(f"aborted        {metadata['aborted']}")
        return
    print(
        f"waypoints      {metadata.get('waypoints_reached')} of {metadata.get('waypoints')} reached"
    )
    localisation = metrics.get("localisation", {})
    print(
        f"localised      {localisation.get('availability', float('nan')) * 100:.1f} per cent of "
        f"the flight, longest gap {localisation.get('longest_gap_s', float('nan')):.1f} s"
    )
    fixes = metrics.get("fixes", {})
    rate = fixes.get("fix_rate_hz", float("nan"))
    print(
        f"marker fixes   {fixes.get('fixes', 0)} at {rate:.1f} Hz, "
        f"{fixes.get('acceptance', float('nan')) * 100:.0f} per cent accepted, "
        f"median residual {fixes.get('reprojection_rmse_px_median', float('nan')):.2f} px"
    )
    if "trajectory_error" in metrics:
        error = metrics["trajectory_error"]
        print(
            f"position error {error['position_rmse_m'] * 100:.1f} cm RMSE, "
            f"{error['position_p95_m'] * 100:.1f} cm at the 95th percentile, "
            f"{error['position_max_m'] * 100:.1f} cm worst"
        )
        print(f"heading error  {error['yaw_rmse_deg']:.2f} deg RMSE")
        print(f"consistency    {error['consistency_ratio']:.2f} (error over reported sigma)")
    for event in log.failsafe_events:
        print(
            f"  failsafe {event['time_s']:>6}s  {event['from_state']} -> {event['to_state']}"
            f"  {event['cause']}  {event['detail']}"
        )
    print()
    for name, path in written.items():
        print(f"{name:<12} {path}")


if __name__ == "__main__":
    raise SystemExit(main())
