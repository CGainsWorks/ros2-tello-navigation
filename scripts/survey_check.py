#!/usr/bin/env python3
"""Position error against surveyed points: the measurement the brief asks for.

This is the one accuracy figure in the repository that would mean the same thing
on a real Tello as it does here. It needs no motion capture and no ground-truth
trajectory. Put the aircraft on a point whose position you measured with a tape,
hold it still, read what the estimator says, subtract.

What this script does in simulation is exactly that, minus the hovering: it places
the camera at each surveyed check point, renders a frame, runs the real detector
and the real pose solver and reports the difference. The result is a handful of
samples rather than a trajectory, which is the honest shape of this measurement.

On hardware
-----------
1. Mark the check points on the floor with a tape measure and record them in the
   same frame as the marker map.
2. For each point, hold the aircraft over it at the stated height, facing the
   stated heading and record ten seconds of ``nav_state``.
3. Take the mean of each hold and subtract the surveyed point.

The comparison is only as good as the survey. ``maps/marker_map.yaml`` currently
says ``survey: designed``, meaning nothing has been measured at all, so what this
script reports today is the estimator's consistency with a layout that exists only
in a file. That is stated in the output rather than left to be inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.control.waypoint import heading_for_markers
from tello_core.eval.metrics import static_survey_error
from tello_core.geometry import euler_to_matrix, make_transform
from tello_core.localisation.markers import MarkerMap
from tello_core.localisation.multi import estimate_pose
from tello_core.localisation.qr import QrDetector
from tello_core.sim.render import Renderer, RenderSettings
from tello_core.sim.world import Room
from tello_core.types import Frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument("--config", default="src/tello_bringup/config/aircraft.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--stem", default="survey_check")
    parser.add_argument("--samples", type=int, default=25, help="frames averaged per point")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--height", type=float, default=1.20)
    parser.add_argument("--json", action="store_true")
    return parser


def check_points(config: AircraftConfig, height: float):
    """The points held over, in the map frame.

    A five-point pattern: the four corners of the flyable area, inset far enough
    to be reachable and the centre. They are chosen to span the geometry rather
    than to be easy: the corners are where the aircraft is furthest from half the
    markers and closest to the fence.
    """
    fence = config.geofence
    inset = 0.65
    low = fence.min_corner + inset
    high = fence.max_corner - inset
    return [
        ("sw", np.array([low[0], low[1], height])),
        ("se", np.array([high[0], low[1], height])),
        ("ne", np.array([high[0], high[1], height])),
        ("nw", np.array([low[0], high[1], height])),
        ("centre", np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, height])),
    ]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
    config = AircraftConfig.from_yaml(REPO_ROOT / args.config)
    camera = CameraModel.tello_nominal()
    room = Room.from_marker_map(marker_map)
    detector = QrDetector()
    T_bc = config.mount.transform()
    rng = np.random.default_rng(args.seed)

    rows = []
    estimates = []
    surveyed = []
    for name, point in check_points(config, args.height):
        yaw = heading_for_markers(point, marker_map, config, camera)
        samples = []
        markers_seen = set()
        for trial in range(args.samples):
            # A held hover is not perfectly still. Two centimetres and a degree
            # of jitter is about what a Tello holds station to indoors and
            # averaging over it is what the hardware procedure does too.
            position = point + rng.normal(0.0, 0.02, 3)
            heading = yaw + rng.normal(0.0, np.deg2rad(1.0))
            pose = make_transform(euler_to_matrix(float(heading)), position) @ T_bc
            renderer = Renderer(room, camera, RenderSettings(seed=args.seed + trial))
            image = renderer.render(pose)
            fix = estimate_pose(
                detector.detect(Frame(timestamp=0.0, image=image)),
                marker_map,
                camera,
                T_bc,
            )
            if fix is None:
                continue
            samples.append(fix.position - (position - point))
            markers_seen.update(fix.marker_ids)

        if not samples:
            rows.append({"point": name, "fixes": 0})
            print(f"  {name:<7} no fix at any of {args.samples} samples")
            continue

        mean = np.mean(samples, axis=0)
        spread = np.std(samples, axis=0)
        error = mean - point
        estimates.append(mean)
        surveyed.append(point)
        rows.append(
            {
                "point": name,
                "surveyed_x": round(float(point[0]), 4),
                "surveyed_y": round(float(point[1]), 4),
                "surveyed_z": round(float(point[2]), 4),
                "heading_deg": round(float(np.rad2deg(yaw)), 1),
                "fixes": len(samples),
                "estimate_x": round(float(mean[0]), 4),
                "estimate_y": round(float(mean[1]), 4),
                "estimate_z": round(float(mean[2]), 4),
                "error_x": round(float(error[0]), 4),
                "error_y": round(float(error[1]), 4),
                "error_z": round(float(error[2]), 4),
                "error_m": round(float(np.linalg.norm(error)), 4),
                "spread_m": round(float(np.linalg.norm(spread)), 4),
                "markers": "|".join(sorted(markers_seen)),
            }
        )
        print(
            f"  {name:<7} error {np.linalg.norm(error) * 100:5.1f} cm  "
            f"({error[0] * 100:+5.1f}, {error[1] * 100:+5.1f}, {error[2] * 100:+5.1f})  "
            f"spread {np.linalg.norm(spread) * 100:4.1f} cm  "
            f"from {len(markers_seen)} marker(s), {len(samples)}/{args.samples} fixes"
        )

    summary = static_survey_error(estimates, surveyed)
    summary["survey_method"] = marker_map.survey.method
    summary["survey_stated_accuracy_m"] = float(marker_map.survey.stated_accuracy)
    summary["camera_source"] = camera.source
    summary["samples_per_point"] = args.samples
    summary["height_m"] = args.height

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with open(out / f"{args.stem}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with open(out / f"{args.stem}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print()
        print(f"points         {summary.get('samples', 0)}")
        if summary.get("samples"):
            print(f"error rmse     {summary['error_rmse'] * 100:.1f} cm")
            print(f"error max      {summary['error_max'] * 100:.1f} cm")
            bias = summary["bias_m"]
            print(
                f"bias           ({bias[0] * 100:+.1f}, {bias[1] * 100:+.1f}, "
                f"{bias[2] * 100:+.1f}) cm"
            )
        print(f"survey method  {marker_map.survey.method}")
        if marker_map.survey.method == "designed":
            print(
                "               The layout has never been measured, so this is the "
                "estimator's agreement with a file, not with a room."
            )
        print(f"wrote          {out / (args.stem + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
