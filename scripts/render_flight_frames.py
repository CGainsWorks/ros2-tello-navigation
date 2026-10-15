#!/usr/bin/env python3
"""What the camera saw: a contact sheet of frames from a flight, with the
detections drawn on.

The brief asks for a flight video. There is no aircraft, so there is no footage
and a video of a simulation would be a video of this repository's own renderer
rather than evidence about anything. This is the honest substitute: real frames
from the reference flight, re-rendered at the poses the aircraft actually
occupied, with the detector run over each one and what it found drawn on top.

It is worth more than a video would be, for two reasons. A reader can see the
image quality the detector is working with, which is the thing that decides
whether any of this works and they can see *which* markers were read at each
point in the flight rather than watching a dot move across a plot.

Frames are re-rendered rather than saved during the flight. Saving them would mean
either keeping several hundred megabytes of PNG out of a thirty-second flight or
making the mission runner decide what to keep; re-rendering from the logged pose
is exact, costs a few seconds and keeps the flight log a few tens of kilobytes.

Writes ``analysis/figures/camera_frames.png``. One PNG, no Git LFS needed, which
is the other reason not to produce a video.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import cv2
import numpy as np

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.geometry import euler_to_matrix, make_transform
from tello_core.localisation.markers import MarkerMap
from tello_core.localisation.multi import estimate_pose
from tello_core.localisation.qr import QrDetector
from tello_core.logs import FlightLog
from tello_core.sim.render import Renderer, RenderSettings
from tello_core.sim.world import Room
from tello_core.types import Frame

FLYING = {"TAKEOFF", "NAVIGATE", "HOLD", "RETURN", "LAND", "EMERGENCY_LAND"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results/latest")
    parser.add_argument("--stem", default="flight")
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument("--config", default="src/tello_bringup/config/aircraft.yaml")
    parser.add_argument("--out", default="analysis/figures/camera_frames.png")
    parser.add_argument("--frames", type=int, default=12, help="frames on the sheet")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=400, help="thumbnail width, pixels")
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path = REPO_ROOT / args.results / f"{args.stem}_trajectory.csv"
    if not path.is_file():
        print(f"no flight log at {path}", file=sys.stderr)
        return 2

    log = FlightLog.read(path)
    columns = log.to_arrays()
    if not columns or "truth_x" not in columns or not np.isfinite(columns["truth_x"]).any():
        print(
            "this log has no ground-truth poses, so the frames cannot be "
            "re-rendered. Only a simulated flight carries them.",
            file=sys.stderr,
        )
        return 2

    marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
    config = AircraftConfig.from_yaml(REPO_ROOT / args.config)
    camera = CameraModel.tello_nominal()
    room = Room.from_marker_map(marker_map)
    renderer = Renderer(room, camera, RenderSettings(seed=args.seed))
    detector = QrDetector()
    T_bc = config.mount.transform()

    indices = _choose(columns, args.frames)
    panels = []
    for index in indices:
        panels.append(
            _panel(columns, index, renderer, detector, marker_map, camera, T_bc, args.width)
        )

    sheet = _grid(panels, args.columns)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(panels)} frames)")
    return 0


def _choose(columns, count: int):
    """Evenly spaced moments while the aircraft was flying.

    Evenly in *time* rather than evenly over interesting events, because a sheet
    that quietly picked the frames where detection worked would be a sales
    brochure rather than evidence.
    """
    flying = np.array([str(state) in FLYING for state in columns["state"]])
    candidates = np.flatnonzero(flying)
    if candidates.size == 0:
        candidates = np.arange(len(columns["time_s"]))
    if candidates.size <= count:
        return list(candidates)
    picks = np.linspace(0, candidates.size - 1, count).round().astype(int)
    return [int(candidates[p]) for p in picks]


def _panel(columns, index, renderer, detector, marker_map, camera, T_bc, width):
    """One frame, re-rendered at the logged pose, with the detections drawn."""
    position = np.array(
        [columns["truth_x"][index], columns["truth_y"][index], columns["truth_z"][index]]
    )
    yaw = float(np.deg2rad(columns["truth_yaw_deg"][index]))
    pose = make_transform(euler_to_matrix(yaw), position) @ T_bc

    image = renderer.render(pose)
    observations = detector.detect(Frame(timestamp=0.0, image=image))
    fix = estimate_pose(observations, marker_map, camera, T_bc)

    annotated = image.copy()
    for observation in observations:
        corners = np.asarray(observation.corners, dtype=np.int32).reshape(-1, 1, 2)
        known = marker_map.by_payload(observation.decoded) is not None
        colour = (0, 200, 0) if known else (0, 160, 255)
        cv2.polylines(annotated, [corners], True, colour, 3)
        label = observation.decoded if known else f"{observation.decoded}?"
        origin = tuple(int(v) for v in np.asarray(observation.corners)[0])
        cv2.putText(
            annotated, label, (origin[0], max(origin[1] - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA,
        )

    scale = width / annotated.shape[1]
    thumb = cv2.resize(
        annotated, (width, int(round(annotated.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )

    caption_height = 52
    panel = np.full((thumb.shape[0] + caption_height, width, 3), 24, dtype=np.uint8)
    panel[: thumb.shape[0]] = thumb

    error = columns["truth_error_m"][index]
    lines = [
        f"t={columns['time_s'][index]:5.1f}s  {str(columns['state'][index])}  "
        f"at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})",
        (
            f"{len(observations)} read, {len(fix.marker_ids) if fix else 0} used"
            + (f", {fix.reprojection_rmse:.2f} px" if fix else ", no fix")
            + (f", off by {error * 100:.0f} cm" if np.isfinite(error) else "")
        ),
    ]
    for offset, text in enumerate(lines):
        cv2.putText(
            panel, text, (8, thumb.shape[0] + 20 + offset * 20),
            cv2.FONT_HERSHEY_SIMPLEX, _fitted(text, width - 16), (235, 235, 235), 1,
            cv2.LINE_AA,
        )
    return panel


def _fitted(text: str, available: int, largest: float = 0.42) -> float:
    """The largest font scale at which a caption fits the panel.

    Measured rather than assumed. The captions vary in length by a dozen
    characters depending on how many markers were read and a fixed scale chosen
    for the short ones silently truncates the long ones, which are exactly the
    frames worth reading.
    """
    scale = largest
    while scale > 0.2:
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if text_width <= available:
            return scale
        scale -= 0.02
    return scale


def _grid(panels, columns: int) -> np.ndarray:
    if not panels:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    rows = int(np.ceil(len(panels) / columns))
    height, width = panels[0].shape[:2]
    gap = 6
    sheet = np.full(
        (rows * height + (rows + 1) * gap, columns * width + (columns + 1) * gap, 3),
        12,
        dtype=np.uint8,
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        y = gap + row * (height + gap)
        x = gap + column * (width + gap)
        sheet[y : y + height, x : x + width] = panel
    return sheet


if __name__ == "__main__":
    raise SystemExit(main())
