#!/usr/bin/env python3
"""Calibrate the camera from a chessboard recording and check the result.

The intrinsics this repository ships are *derived*, not measured: the focal length
comes from DJI's published 82.6 degree diagonal field of view and the distortion
is set to zero because there is no published figure for it. That is the largest
single unverified assumption in the project, because the focal length scales every
range estimate linearly and uncorrected distortion adds an error that grows
towards the frame edges, which is exactly where a marker is when the aircraft is
about to lose sight of it. See ``docs/HARDWARE.md``.

This script replaces that guess with a measurement.

    python scripts/calibrate_camera.py --images 'calib/*.png' \\
        --rows 6 --columns 9 --square 0.025

Writes the same YAML format ``CameraModel.from_yaml`` reads, so a measured
calibration is a configuration change and not a code change.

How to take the pictures
------------------------
Print a chessboard, stick it to something rigid and flat and record the Tello's
stream while moving the board, not the aircraft, through the frame. Twenty to
thirty frames, with the board:

* filling most of the frame in some and small in others, so the focal length and
  the principal point separate;
* tilted 30 to 45 degrees in several, because a set of head-on views leaves the
  distortion coefficients poorly determined;
* in each corner of the frame at least once, since that is where the distortion
  lives.

What this script does that a bare ``calibrateCamera`` call does not: it reports the
per-frame reprojection error so a bad frame can be dropped and it reports what
the measurement means for *this* application, which is the error in a marker's
range at the distances this system actually flies at. A calibration with an
impressive RMS and a 5 per cent focal-length error is a calibration that will
produce confidently wrong positions.

Without a recording
-------------------
``--self-test`` runs the whole pipeline against synthetically rendered
chessboards with known intrinsics, which is how the code here is exercised in the
absence of a camera. It checks that the recovered focal length matches the one
used to generate the images. That is a test of this script, not a calibration.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import cv2
import numpy as np
import yaml

from tello_core.camera import CameraModel

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", default=None, help="glob of chessboard frames")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--columns", type=int, default=9, help="inner corners per row")
    parser.add_argument("--square", type=float, default=0.025, help="square size, metres")
    parser.add_argument(
        "--out", default="src/tello_bringup/config/camera_tello.yaml"
    )
    parser.add_argument(
        "--max-frame-error",
        type=float,
        default=1.0,
        help="drop frames whose reprojection error exceeds this, in pixels",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run against synthetic chessboards with known intrinsics",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return _self_test(args)

    if not args.images:
        print(
            "no --images given. Either pass a glob of chessboard frames or use "
            "--self-test; see the docstring for how to take the pictures.",
            file=sys.stderr,
        )
        return 2

    paths = sorted(glob.glob(args.images))
    if not paths:
        print(f"no images matched {args.images!r}", file=sys.stderr)
        return 2

    images = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  could not read {path}")
            continue
        images.append((path, image))

    result = calibrate(images, args)
    if result is None:
        return 1
    model, report = result
    _write(REPO_ROOT / args.out, model, report)
    return 0


def find_corners(image: np.ndarray, rows: int, columns: int):
    """Chessboard corners to sub-pixel accuracy, or ``None``.

    ``findChessboardCornersSB`` where available: it is more robust to blur and to
    uneven lighting than the older detector, which matters on a frame from a
    small drone camera under a ceiling light.
    """
    pattern = (columns, rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            image, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if found:
            return corners
    found, corners = cv2.findChessboardCorners(
        image,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    return cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)


def calibrate(images, args):
    """Run the calibration, dropping frames that do not fit."""
    board = np.zeros((args.rows * args.columns, 3), dtype=np.float32)
    board[:, :2] = np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
    board *= args.square

    object_points = []
    image_points = []
    used = []
    size = None
    for path, image in images:
        corners = find_corners(image, args.rows, args.columns)
        if corners is None:
            print(f"  no board found in {Path(path).name}")
            continue
        object_points.append(board.copy())
        image_points.append(corners)
        used.append(path)
        size = (image.shape[1], image.shape[0])

    if len(object_points) < 6:
        print(
            f"only {len(object_points)} usable frames, which is not enough. "
            "Twenty to thirty is the target.",
            file=sys.stderr,
        )
        return None

    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, size, None, None
    )

    # Per-frame error, so a bad frame can be identified rather than quietly
    # dragging the whole calibration.
    per_frame = []
    for index in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[index], rvecs[index], tvecs[index], matrix, distortion
        )
        error = float(
            np.sqrt(
                np.mean(
                    np.sum((projected.reshape(-1, 2) - image_points[index].reshape(-1, 2)) ** 2,
                           axis=1)
                )
            )
        )
        per_frame.append((error, used[index]))

    print(f"\n{len(object_points)} frames, overall RMS {rms:.3f} px")
    for error, path in sorted(per_frame, reverse=True)[:5]:
        flag = "  <-- consider dropping" if error > args.max_frame_error else ""
        print(f"  {error:5.3f} px  {Path(path).name}{flag}")

    model = CameraModel(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        width=int(size[0]),
        height=int(size[1]),
        distortion=np.asarray(distortion, dtype=float).ravel()[:5],
        source=f"chessboard-{len(object_points)}-frames",
    )
    report = {
        "rms_px": round(float(rms), 4),
        "frames": len(object_points),
        "worst_frame_px": round(float(max(e for e, _ in per_frame)), 4),
    }
    _compare_with_nominal(model)
    return model, report


def _compare_with_nominal(model: CameraModel) -> None:
    """Say what the measurement means for this application, not just its RMS."""
    nominal = CameraModel.tello_nominal()
    if (model.width, model.height) != (nominal.width, nominal.height):
        print(
            f"\nnote: calibrated at {model.width}x{model.height} and the shipped "
            f"nominal model is {nominal.width}x{nominal.height}. Use "
            "CameraModel.scaled() rather than mixing them."
        )
        return

    drift = (model.fx - nominal.fx) / nominal.fx
    print("\nagainst the derived nominal model")
    print(f"  focal length  {nominal.fx:.1f} px derived, {model.fx:.1f} px measured "
          f"({drift * 100:+.1f} per cent)")
    print(f"  principal pt  ({nominal.cx:.1f}, {nominal.cy:.1f}) assumed, "
          f"({model.cx:.1f}, {model.cy:.1f}) measured")
    print(f"  distortion    zero assumed, k1 = {model.distortion[0]:+.4f} measured")
    print(
        f"\n  a {abs(drift) * 100:.1f} per cent focal-length error is "
        f"{abs(drift) * 2.0 * 100:.1f} cm of range error at 2 m, which is the "
        "distance this system flies markers at"
    )
    if abs(drift) > 0.03:
        print(
            "  the derived model was out by more than three per cent, so every "
            "position estimate in results/ was scaled by that much. Re-run the "
            "evaluation with the measured model."
        )


def _write(path: Path, model: CameraModel, report: dict) -> None:
    payload = model.to_dict()
    payload["calibration"] = report
    payload["note"] = (
        "Measured with scripts/calibrate_camera.py. Point the nodes and the "
        "offline scripts at this file to use it instead of the derived nominal "
        "model."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


# ------------------------------------------------------------------- self test


def _render_chessboard(camera: CameraModel, rows: int, columns: int, square: float,
                       pose, rng) -> np.ndarray:
    """A chessboard seen through ``camera`` at ``pose``, by warping a flat board."""
    cell = 60
    board = np.zeros(((rows + 1) * cell, (columns + 1) * cell), dtype=np.uint8)
    for row in range((rows + 1)):
        for column in range((columns + 1)):
            if (row + column) % 2 == 0:
                board[row * cell : (row + 1) * cell, column * cell : (column + 1) * cell] = 255

    # Board corners in board metres and where they land in the image.
    width = (columns + 1) * square
    height = (rows + 1) * square
    corners_board = np.array(
        [[0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, height, 0.0], [0.0, height, 0.0]]
    )
    rotation, translation = pose
    corners_camera = (rotation @ corners_board.T).T + translation
    pixels = camera.project(corners_camera).astype(np.float32)
    if not np.isfinite(pixels).all():
        return None

    source = np.array(
        [[0.0, 0.0], [board.shape[1] - 1.0, 0.0],
         [board.shape[1] - 1.0, board.shape[0] - 1.0], [0.0, board.shape[0] - 1.0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, pixels)
    canvas = np.full((camera.height, camera.width), 128, dtype=np.uint8)
    warped = cv2.warpPerspective(
        board, transform, (camera.width, camera.height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=128,
    )
    mask = cv2.warpPerspective(
        np.full(board.shape, 255, np.uint8), transform,
        (camera.width, camera.height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    np.copyto(canvas, warped, where=mask.astype(bool))
    canvas = canvas.astype(np.float32) + rng.normal(0.0, 1.5, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _self_test(args) -> int:
    """Calibrate synthetic boards rendered with known intrinsics."""
    from tello_core.geometry import euler_to_matrix

    truth = CameraModel.tello_nominal()
    rng = np.random.default_rng(5)
    rows, columns, square = 6, 9, 0.025
    width = (columns + 1) * square
    height = (rows + 1) * square

    images = []
    for index in range(24):
        yaw = rng.uniform(-0.45, 0.45)
        pitch = rng.uniform(-0.45, 0.45)
        roll = rng.uniform(-0.25, 0.25)
        depth = rng.uniform(0.35, 0.95)
        rotation = euler_to_matrix(yaw, pitch, roll)
        centre = np.array(
            [rng.uniform(-0.12, 0.12), rng.uniform(-0.09, 0.09), depth]
        )
        translation = centre - rotation @ np.array([width / 2, height / 2, 0.0])
        image = _render_chessboard(
            truth, rows, columns, square, (rotation, translation), rng
        )
        if image is not None:
            images.append((f"synthetic_{index:02d}.png", image))

    print(f"rendered {len(images)} synthetic boards with fx = {truth.fx:.1f} px")
    namespace = argparse.Namespace(
        rows=rows, columns=columns, square=square,
        max_frame_error=args.max_frame_error,
    )
    result = calibrate(images, namespace)
    if result is None:
        print("the self test could not calibrate its own images", file=sys.stderr)
        return 1
    model, _ = result

    error = abs(model.fx - truth.fx) / truth.fx
    print(f"\nself test: recovered fx within {error * 100:.2f} per cent of the truth")
    if error > 0.02:
        print(
            "that is worse than two per cent, which means this script would not "
            "produce a usable calibration from real images either",
            file=sys.stderr,
        )
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
