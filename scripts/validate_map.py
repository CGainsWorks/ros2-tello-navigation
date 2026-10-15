#!/usr/bin/env python3
"""Check a marker map before anything flies at it.

Three kinds of finding and the difference matters:

**Errors** stop the map loading at all: a duplicate identifier, two markers with
the same payload, a marker with no size. The loader raises on these and this
script reports them and exits non-zero.

**Warnings** are geometry that will make localisation worse without making it
impossible: markers too close together, a layout that is nearly coplanar, a
marker below the floor. Printed, not fatal, because a map can be awkward and
still be a correct description of a real room.

**Coverage** is the useful part. It flies a grid of virtual poses through the
room and reports, for each, how many markers a camera there could actually read.
A layout with a hole in it will localise beautifully for most of a mission and
lose the aircraft in one corner and that is much easier to fix on paper than in
the air.

Run it after every change to ``maps/marker_map.yaml``, which CI does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.geometry import euler_to_matrix, make_transform
from tello_core.localisation.markers import MarkerMap, MarkerMapError, markers_in_view
from tello_core.sim.qr import decodable_range, symbol_modules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument("--config", default="src/tello_bringup/config/aircraft.yaml")
    parser.add_argument("--step", type=float, default=0.4, help="coverage grid spacing, metres")
    parser.add_argument("--height", type=float, default=1.2, help="coverage grid height, metres")
    parser.add_argument(
        "--headings", type=int, default=16, help="headings tried at each grid point"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="fraction of the flyable area that must see at least one marker",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path = REPO_ROOT / args.marker_map

    try:
        marker_map = MarkerMap.from_yaml(path)
    except MarkerMapError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    config = AircraftConfig.from_yaml(REPO_ROOT / args.config)
    camera = CameraModel.tello_nominal()

    warnings = marker_map.check_geometry()
    coverage = _coverage(marker_map, config, camera, args)

    if args.json:
        print(json.dumps({"warnings": warnings, **coverage}, indent=2))
    else:
        _report(marker_map, camera, warnings, coverage, args)

    if coverage["covered_fraction"] < args.min_coverage:
        print(
            f"\nerror: only {coverage['covered_fraction'] * 100:.1f} per cent of the "
            f"flyable area can see a marker, against a required "
            f"{args.min_coverage * 100:.0f} per cent",
            file=sys.stderr,
        )
        return 1
    return 0


def _coverage(marker_map: MarkerMap, config: AircraftConfig, camera: CameraModel, args) -> dict:
    """How much of the flyable volume can localise and from which headings.

    A point counts as covered if *some* heading sees a marker and doubly covered
    if some heading sees two. The distinction matters: one marker gives a pose
    whose lateral component is weakly observed and two on different walls fix
    what neither fixes alone.
    """
    fence = config.geofence
    xs = np.arange(fence.min_corner[0], fence.max_corner[0] + 1e-9, args.step)
    ys = np.arange(fence.min_corner[1], fence.max_corner[1] + 1e-9, args.step)
    headings = np.linspace(-np.pi, np.pi, args.headings, endpoint=False)
    T_bc = config.mount.transform()

    total = 0
    covered = 0
    doubly = 0
    blind: list = []
    best_counts = []
    for x in xs:
        for y in ys:
            total += 1
            position = np.array([float(x), float(y), float(args.height)])
            best = 0
            for yaw in headings:
                T_wc = make_transform(euler_to_matrix(float(yaw)), position) @ T_bc
                best = max(best, len(markers_in_view(marker_map, T_wc, camera)))
                if best >= 2:
                    break
            best_counts.append(best)
            if best >= 1:
                covered += 1
            if best >= 2:
                doubly += 1
            else:
                blind.append([round(float(x), 2), round(float(y), 2), best])

    return {
        "grid_points": total,
        "covered_fraction": covered / max(total, 1),
        "two_marker_fraction": doubly / max(total, 1),
        "mean_best_markers": float(np.mean(best_counts)) if best_counts else 0.0,
        "weak_points": blind[:40],
        "height_m": args.height,
        "step_m": args.step,
    }


def _report(marker_map: MarkerMap, camera: CameraModel, warnings, coverage, args) -> None:
    survey = marker_map.survey
    print(f"map            {marker_map.source_path}")
    print(f"markers        {len(marker_map)}")
    print(
        f"survey         {survey.method} ({survey.instrument}), stated accuracy "
        f"{survey.stated_accuracy * 1000:.0f} mm"
    )
    if survey.method == "designed":
        print(
            "               NOTE: 'designed' means this layout has never been measured. "
            "Every accuracy figure derived from it is a statement about the "
            "simulator, not about a room."
        )

    low, high = marker_map.bounds()
    print(
        f"extent         x {low[0]:.2f} to {high[0]:.2f}, y {low[1]:.2f} to {high[1]:.2f}, "
        f"z {low[2]:.2f} to {high[2]:.2f} m"
    )

    print("\nmarkers")
    for marker in marker_map:
        reach = decodable_range(marker.payload, marker.size, camera.fx)
        print(
            f"  {marker.marker_id:<5} {marker.payload:<8} {marker.size * 1000:.0f} mm  "
            f"{symbol_modules(marker.payload)} modules  "
            f"at ({marker.position[0]:.2f}, {marker.position[1]:.2f}, {marker.position[2]:.2f})  "
            f"facing {np.rad2deg(marker.normal_yaw):>6.1f} deg  "
            f"reach about {reach:.1f} m"
        )

    print("\ncoverage")
    print(f"  grid           {coverage['grid_points']} points at z = {args.height:.2f} m")
    print(f"  sees a marker  {coverage['covered_fraction'] * 100:.1f} per cent")
    print(f"  sees two       {coverage['two_marker_fraction'] * 100:.1f} per cent")
    if coverage["weak_points"]:
        print("  weak points (x, y, best markers from any heading):")
        for x, y, count in coverage["weak_points"][:12]:
            print(f"    ({x:.2f}, {y:.2f})  {count}")
        if len(coverage["weak_points"]) > 12:
            print(f"    ... and {len(coverage['weak_points']) - 12} more")

    if warnings:
        print("\nwarnings")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("\nno geometry warnings")


if __name__ == "__main__":
    raise SystemExit(main())
