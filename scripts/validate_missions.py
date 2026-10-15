#!/usr/bin/env python3
"""Check every shipped mission against the map and the geofence.

Two failures are worth catching on the ground and both are certain to happen in
the air otherwise:

* a waypoint outside the geofence, which trips the failsafe the moment it is
  reached and turns a mission into a landing;
* a waypoint from which no marker can be seen at any heading, which means the
  aircraft flies that leg blind and the localisation failsafe notices within two
  seconds.

The second is a warning rather than an error by default, because a short blind
leg between two well-covered ones is a legitimate thing to fly. ``--strict``
makes it fatal.

Run by CI over ``missions/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.control.mission import Mission, MissionError, mission_visibility
from tello_core.localisation.markers import MarkerMap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("missions", nargs="*", default=None, help="defaults to missions/*.yaml")
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument("--config", default="src/tello_bringup/config/aircraft.yaml")
    parser.add_argument(
        "--strict", action="store_true", help="treat a blind waypoint as an error"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
    config = AircraftConfig.from_yaml(REPO_ROOT / args.config)
    camera = CameraModel.tello_nominal()

    paths = (
        [Path(p) for p in args.missions]
        if args.missions
        else sorted((REPO_ROOT / "missions").glob("*.yaml"))
    )
    if not paths:
        print("no mission files found", file=sys.stderr)
        return 2

    failures = 0
    for path in paths:
        failures += _check(path, marker_map, config, camera, args.strict)
    print()
    if failures:
        print(f"{failures} problem(s) found", file=sys.stderr)
        return 1
    print(f"all {len(paths)} mission(s) validate")
    return 0


def _check(path: Path, marker_map, config, camera, strict: bool) -> int:
    print(f"\n{path.relative_to(REPO_ROOT) if path.is_absolute() else path}")
    try:
        mission = Mission.from_yaml(path)
    except MissionError as error:
        print(f"  error: {error}")
        return 1

    problems = mission.validate(config, marker_map)
    for problem in problems:
        print(f"  error: {problem}")

    rows = mission_visibility(mission, marker_map, config, camera)
    blind = 0
    total_length = 0.0
    previous = None
    for row, waypoint in zip(rows, mission.waypoints):
        if previous is not None:
            total_length += float(np.linalg.norm(waypoint.position - previous))
        previous = waypoint.position
        marker_text = ", ".join(row["visible_markers"]) if row["visible_markers"] else "none"
        flag = " <-- sees nothing" if not row["visible_markers"] else ""
        print(
            f"  {row['index']:>2} {row['name']:<12} "
            f"({row['position'][0]:.2f}, {row['position'][1]:.2f}, {row['position'][2]:.2f})  "
            f"yaw {row['yaw_deg']:>7.1f}{' auto' if row['auto_yaw'] else '     '}  "
            f"sees {marker_text}{flag}"
        )
        if not row["visible_markers"]:
            blind += 1

    print(
        f"  {len(mission.waypoints)} waypoints, {total_length:.1f} m of path, "
        f"take-off height {mission.takeoff_height:.2f} m, {mission.loops} loop(s)"
    )
    if blind:
        level = "error" if strict else "warning"
        print(f"  {level}: {blind} waypoint(s) can see no marker at any heading")
        if strict:
            problems = problems + ["blind waypoints"]
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
