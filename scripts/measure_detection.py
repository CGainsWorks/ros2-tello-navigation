#!/usr/bin/env python3
"""Measure what the QR detector can actually read and under what conditions.

Three sweeps, all against the renderer in ``src/tello_core/sim``, all seeded:

``range``
    Detection rate against distance, viewing the marker square on. Fixes the
    marker size in ``maps/marker_map.yaml`` and the geometry of the test area:
    a layout whose waypoints sit beyond this curve's cliff will not localise and
    that is a design decision to make on the ground rather than a surprise in the
    air.

``yaw_rate``
    Detection rate against how fast the aircraft is turning, at a fixed 1.5 m.
    The one that mattered most in practice. The first autonomous flight in this
    repository lost localisation nine seconds after take-off and the cause was
    not the estimator: the controller was allowed to yaw at 90 degrees per second,
    which at 1/60 s exposure smears a marker across roughly eighteen pixels and
    destroys the module grid the decoder needs. The controller's yaw limit is set
    from this curve.

``incidence``
    Detection rate against viewing angle, at a fixed 1.5 m. Decides the maximum
    incidence used when planning headings.

Writes ``results/detection_range.csv``, ``results/detection_yaw_rate.csv`` and
``results/detection_incidence.csv``. Every number the README or the marker map
quotes about detection comes from these files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.geometry import euler_to_matrix, make_transform
from tello_core.localisation.markers import MarkerMap
from tello_core.localisation.multi import estimate_pose
from tello_core.localisation.qr import QrDetector
from tello_core.sim.render import Renderer, RenderSettings
from tello_core.sim.world import Room
from tello_core.types import Frame

# The marker the sweeps look at and the wall it is on. M01 is on the x = 0 wall
# facing along +x, so a camera at (d, 0.9, 1.4) looking at -x sees it square on
# at range d.
TARGET = "M01"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument("--config", default="src/tello_bringup/config/aircraft.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--trials", type=int, default=12, help="renders per operating point")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--sweeps", nargs="*", default=["range", "yaw_rate", "incidence"])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
    config = AircraftConfig.from_yaml(REPO_ROOT / args.config)
    camera = CameraModel.tello_nominal()
    marker = marker_map.get(TARGET)
    if marker is None:
        raise SystemExit(f"marker {TARGET} is not in the map")

    room = Room.from_marker_map(marker_map)
    detector = QrDetector()
    T_bc = config.mount.transform()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"detector backend: {detector.backend}, marker size {marker.size:.3f} m, "
          f"focal {camera.fx:.1f} px")

    if "range" in args.sweeps:
        rows = sweep_range(room, camera, detector, marker, marker_map, T_bc, args)
        write(out / "detection_range.csv", rows)
    if "yaw_rate" in args.sweeps:
        rows = sweep_yaw_rate(room, camera, detector, marker, marker_map, T_bc, args)
        write(out / "detection_yaw_rate.csv", rows)
    if "incidence" in args.sweeps:
        rows = sweep_incidence(room, camera, detector, marker, marker_map, T_bc, args)
        write(out / "detection_incidence.csv", rows)
    return 0


def _jitter(position: np.ndarray, heading: float, rng) -> tuple:
    """Perturb a nominal pose by a realistic amount.

    Every operating point is sampled over a small spread of poses rather than
    measured at one exact alignment. Without it the curves come out sharply
    non-monotonic and the reason is an artefact rather than a finding: at one
    particular range the marker's module grid happens to land on the pixel grid
    in a way the decoder handles badly and two centimetres further away it does
    not. Nothing flies to two centimetres, so a curve measured to two centimetres
    is measuring the wrong thing.
    """
    return (
        position + rng.normal(0.0, 0.02, 3),
        heading + rng.normal(0.0, np.deg2rad(2.0)),
    )


def _trial(
    renderer: Renderer,
    detector: QrDetector,
    marker_map: MarkerMap,
    camera: CameraModel,
    T_bc: np.ndarray,
    T_wb: np.ndarray,
    truth: np.ndarray,
    angular_velocity: np.ndarray,
    linear_velocity: np.ndarray,
):
    """One render, one detection, one pose solve.

    Returns ``(decoded, fixed, error, rmse)``. Decoding and fixing are reported
    separately because they fail for different reasons and the difference is
    actionable: a marker that decodes but yields no fix is a geometry problem
    and one that does not decode at all is a resolution or blur problem.
    """
    image = renderer.render(
        T_wb @ T_bc,
        angular_velocity=angular_velocity,
        linear_velocity=linear_velocity,
    )
    observations = detector.detect(Frame(timestamp=0.0, image=image))
    decoded = any(o.decoded == marker_map.get(TARGET).payload for o in observations)
    if not decoded:
        return False, False, float("nan"), float("nan")
    fix = estimate_pose(observations, marker_map, camera, T_bc)
    if fix is None:
        return True, False, float("nan"), float("nan")
    return True, True, float(np.linalg.norm(fix.position - truth)), float(fix.reprojection_rmse)


def sweep_range(room, camera, detector, marker, marker_map, T_bc, args):
    """Square on, stationary, increasing distance."""
    rows = []
    for distance in np.arange(0.6, 4.01, 0.2):
        nominal = np.array([float(distance), marker.position[1], marker.position[2]])
        rng = np.random.default_rng(args.seed + int(distance * 100))
        decoded, fixed, errors, residuals = [], [], [], []
        for trial in range(args.trials):
            position, heading = _jitter(nominal, np.pi, rng)
            T_wb = make_transform(euler_to_matrix(float(heading)), position)
            renderer = Renderer(room, camera, RenderSettings(seed=args.seed + trial))
            ok, has_fix, error, rmse = _trial(
                renderer, detector, marker_map, camera, T_bc, T_wb, position,
                np.zeros(3), np.zeros(3),
            )
            decoded.append(ok)
            fixed.append(has_fix)
            if has_fix:
                errors.append(error)
                residuals.append(rmse)
        span = camera.fx * marker.size / float(distance)
        rows.append({
            "range_m": round(float(distance), 3),
            "marker_span_px": round(float(span), 1),
            "pixels_per_module": round(float(span / 21.0), 2),
            "decode_rate": round(float(np.mean(decoded)), 3),
            "fix_rate": round(float(np.mean(fixed)), 3),
            "position_error_m": round(float(np.mean(errors)), 4) if errors else "",
            "reprojection_rmse_px": round(float(np.mean(residuals)), 3) if residuals else "",
            "trials": args.trials,
        })
        print(f"  range {distance:4.1f} m  span {span:5.0f} px  "
              f"decoded {np.mean(decoded) * 100:5.1f}  fixed {np.mean(fixed) * 100:5.1f} per cent")
    return rows


def sweep_yaw_rate(room, camera, detector, marker, marker_map, T_bc, args):
    """Square on at 1.5 m, turning. The curve the controller's limit comes from."""
    rows = []
    distance = 1.5
    position = np.array([distance, marker.position[1], marker.position[2]])
    for rate_dps in [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90, 120]:
        omega = np.array([0.0, 0.0, float(np.deg2rad(rate_dps))])
        rng = np.random.default_rng(args.seed + 1000 + rate_dps)
        decoded, fixed, errors = [], [], []
        for trial in range(args.trials):
            jittered, heading = _jitter(position, np.pi, rng)
            T_jitter = make_transform(euler_to_matrix(float(heading)), jittered)
            renderer = Renderer(room, camera, RenderSettings(seed=args.seed + trial))
            ok, has_fix, error, _ = _trial(
                renderer, detector, marker_map, camera, T_bc, T_jitter, jittered,
                omega, np.zeros(3),
            )
            decoded.append(ok)
            fixed.append(has_fix)
            if has_fix:
                errors.append(error)
        blur = camera.fx * np.deg2rad(rate_dps) / 60.0
        rows.append({
            "yaw_rate_dps": rate_dps,
            "blur_px": round(float(blur), 2),
            "decode_rate": round(float(np.mean(decoded)), 3),
            "fix_rate": round(float(np.mean(fixed)), 3),
            "position_error_m": round(float(np.mean(errors)), 4) if errors else "",
            "range_m": distance,
            "trials": args.trials,
        })
        print(f"  yaw {rate_dps:4d} deg/s  blur {blur:5.1f} px  "
              f"decoded {np.mean(decoded) * 100:5.1f}  fixed {np.mean(fixed) * 100:5.1f} per cent")
    return rows


def sweep_incidence(room, camera, detector, marker, marker_map, T_bc, args):
    """Fixed 1.5 m from the marker, viewed from increasingly off to one side."""
    rows = []
    distance = 1.5
    for angle_deg in range(0, 81, 10):
        angle = np.deg2rad(angle_deg)
        offset = np.array([np.cos(angle), np.sin(angle), 0.0]) * distance
        nominal = marker.position + offset
        nominal_heading = float(np.arctan2(-offset[1], -offset[0]))
        rng = np.random.default_rng(args.seed + 2000 + angle_deg)
        decoded, fixed, errors = [], [], []
        for trial in range(args.trials):
            position, heading = _jitter(nominal, nominal_heading, rng)
            T_wb = make_transform(euler_to_matrix(float(heading)), position)
            renderer = Renderer(room, camera, RenderSettings(seed=args.seed + trial))
            ok, has_fix, error, _ = _trial(
                renderer, detector, marker_map, camera, T_bc, T_wb, position,
                np.zeros(3), np.zeros(3),
            )
            decoded.append(ok)
            fixed.append(has_fix)
            if has_fix:
                errors.append(error)
        rows.append({
            "incidence_deg": angle_deg,
            "decode_rate": round(float(np.mean(decoded)), 3),
            "fix_rate": round(float(np.mean(fixed)), 3),
            "position_error_m": round(float(np.mean(errors)), 4) if errors else "",
            "range_m": distance,
            "trials": args.trials,
        })
        print(f"  incidence {angle_deg:3d} deg  decoded {np.mean(decoded) * 100:5.1f}  "
              f"fixed {np.mean(fixed) * 100:5.1f} per cent")
    return rows


def write(path: Path, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
