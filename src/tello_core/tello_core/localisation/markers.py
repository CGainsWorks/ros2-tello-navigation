"""The marker map: file format, validation and world-frame corner geometry.

The map is the survey. Everything this system claims about where it is rests on
the numbers in ``maps/marker_map.yaml`` being right, so the loader is strict:
it refuses duplicate identifiers, refuses payloads that two markers share and
refuses a marker whose size is absent from both the entry and the file default.
A map that loads is a map whose geometry is fully determined.

Marker frame
------------
A marker's own frame is OpenCV's, the one ``estimatePoseSingleMarkers`` and
``SOLVEPNP_IPPE_SQUARE`` document:

* ``x`` to the right as printed,
* ``y`` **upwards** as printed,
* ``z`` out of the paper, towards whoever is looking at it.

Corners are ordered top-left, top-right, bottom-right, bottom-left, which is also
the order ``cv2.QRCodeDetector`` and ``cv2.QRCodeDetectorAruco`` return them in,
so the correspondence between image corners and object corners is fixed and no
orientation disambiguation is needed. That is the main practical reason this
project uses QR symbols rather than a marker family whose four-fold symmetry has
to be resolved.

It is worth being explicit about why ``y`` is up rather than down, because "x
right, y down, z out of the page" is the combination that looks natural if you
are used to image coordinates and it is not a right-handed frame with ``z``
towards the viewer: with ``x`` right and ``y`` down, ``z`` points *into* the page.
Using it anyway produces markers that render mirrored and a mirrored marker
still decodes, still solves and gives a pose that is wrong in a way no residual
reveals, because the mirrored geometry is self-consistent. This project got that
wrong once and it cost an afternoon.

Placing a marker in the world
-----------------------------
Markers go on walls, so the natural way to describe one is the compass bearing
its front face looks along. ``normal_yaw_deg`` is the yaw of the marker's ``z``
axis, that is, the direction the printed face points *into the room*. A marker on
the wall at ``x = 0`` facing along positive x has ``normal_yaw_deg: 0``.

``tilt_deg`` rotates the marker about its own ``x`` axis, for a marker that
leans; ``roll_deg`` rotates about its own ``z``, for one that is not hung square.
Both default to zero and both exist because a real survey never comes out
perfectly and forcing the surveyor to pretend it did puts the error into the
localisation instead of into the map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

from ..geometry import make_transform, rot_x, rot_z


class MarkerMapError(ValueError):
    """Raised for any map that is not fully determined or not self-consistent."""


@dataclass(frozen=True)
class Marker:
    """One surveyed marker."""

    marker_id: str
    payload: str
    """The string encoded in the QR symbol. Distinct from ``marker_id`` because
    the payload has to survive being printed and read back, so it is restricted
    to plain ASCII, while the identifier is free to be whatever the survey
    sheet calls it."""

    size: float
    """Side length of the QR symbol's finder-pattern square, metres, excluding
    the white quiet zone. Getting this wrong scales every position estimate by
    exactly the same factor, which is the single easiest way to produce a system
    that looks precise and is entirely wrong."""

    position: np.ndarray
    """Centre of the marker in the world frame, metres."""

    normal_yaw: float
    """Radians. Yaw of the outward normal, that is, of the marker's z axis."""

    tilt: float = 0.0
    """Radians about the marker's own x axis."""

    roll: float = 0.0
    """Radians about the marker's own z axis."""

    survey_sigma: float = 0.005
    """One-sigma survey uncertainty of ``position``, metres. Enters the fix
    covariance, so a marker the surveyor was unsure about correctly produces a
    less confident fix rather than a confidently wrong one."""

    metadata: dict = field(default_factory=dict)

    @property
    def rotation(self) -> np.ndarray:
        """Rotation taking marker-frame vectors to world-frame vectors.

        Built from the outward normal rather than from Euler angles about the
        world axes, because that is how a marker on a wall is actually
        described. With ``normal_yaw = 0`` the marker faces along world ``+x``:
        its ``z`` is world ``+x``, its ``y`` (up as printed) is world ``+z`` and
        its ``x`` is world ``+y``, which is what you get standing in the room
        looking at the wall.
        """
        z_axis = np.array([np.cos(self.normal_yaw), np.sin(self.normal_yaw), 0.0])
        y_axis = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(y_axis, z_axis)
        base = np.column_stack([x_axis, y_axis, z_axis])
        # Tilt is about the marker's own x, roll about its own z, so both are
        # right-multiplications: they act in the marker frame, not the world.
        return base @ rot_x(self.tilt) @ rot_z(self.roll)

    @property
    def pose(self) -> np.ndarray:
        """4x4 transform from the marker frame to the world frame."""
        return make_transform(self.rotation, self.position)

    def corners_marker(self) -> np.ndarray:
        """The four corners in the marker's own frame, ``(4, 3)``.

        Order is top-left, top-right, bottom-right, bottom-left as printed,
        matching both ``cv2.QRCodeDetector`` and ``SOLVEPNP_IPPE_SQUARE``.
        """
        h = self.size / 2.0
        return np.array(
            [
                [-h, h, 0.0],
                [h, h, 0.0],
                [h, -h, 0.0],
                [-h, -h, 0.0],
            ]
        )

    def corners_world(self) -> np.ndarray:
        """The four corners in the world frame, ``(4, 3)``."""
        return (self.rotation @ self.corners_marker().T).T + self.position

    def normal_world(self) -> np.ndarray:
        """Unit outward normal in the world frame."""
        return self.rotation @ np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class SurveyInfo:
    """Provenance of the survey, carried into every results file.

    A position error quoted against a survey is only meaningful alongside how
    that survey was made. Two centimetres of error against a laser-measured
    layout and two centimetres against one paced out with a tape are not the
    same claim, so the map states which it is and the evaluation prints it.
    """

    method: str = "unspecified"
    instrument: str = "unspecified"
    date: str = "unspecified"
    stated_accuracy: float = float("nan")
    """One-sigma, metres."""

    notes: str = ""


@dataclass
class MarkerMap:
    """A validated collection of surveyed markers."""

    markers: List[Marker]
    frame_id: str = "map"
    survey: SurveyInfo = field(default_factory=SurveyInfo)
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        self._by_id: Dict[str, Marker] = {m.marker_id: m for m in self.markers}
        self._by_payload: Dict[str, Marker] = {m.payload: m for m in self.markers}

    def __len__(self) -> int:
        return len(self.markers)

    def __iter__(self):
        return iter(self.markers)

    def get(self, marker_id: str) -> Optional[Marker]:
        return self._by_id.get(marker_id)

    def by_payload(self, payload: str) -> Optional[Marker]:
        """Look a marker up by what the detector actually read off the symbol."""
        return self._by_payload.get(payload)

    @property
    def ids(self) -> List[str]:
        return [m.marker_id for m in self.markers]

    def positions(self) -> np.ndarray:
        """``(N, 3)`` of marker centres, in the order they appear in the file."""
        return np.array([m.position for m in self.markers], dtype=float)

    def bounds(self) -> tuple:
        """Axis-aligned bounding box of the marker centres, ``(min, max)``."""
        p = self.positions()
        return p.min(axis=0), p.max(axis=0)

    @staticmethod
    def from_yaml(path) -> "MarkerMap":
        path = Path(path)
        if not path.is_file():
            raise MarkerMapError(f"marker map not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise MarkerMapError(f"{path}: top level must be a mapping")
        return MarkerMap.from_dict(data, source_path=str(path))

    @staticmethod
    def from_dict(data: dict, source_path: Optional[str] = None) -> "MarkerMap":
        entries = data.get("markers")
        if not entries:
            raise MarkerMapError("map contains no markers")

        default_size = data.get("default_size_m")
        default_sigma = float(data.get("default_survey_sigma_m", 0.005))
        survey_raw = data.get("survey", {}) or {}
        survey = SurveyInfo(
            method=str(survey_raw.get("method", "unspecified")),
            instrument=str(survey_raw.get("instrument", "unspecified")),
            date=str(survey_raw.get("date", "unspecified")),
            stated_accuracy=float(survey_raw.get("stated_accuracy_m", float("nan"))),
            notes=str(survey_raw.get("notes", "")),
        )

        markers: List[Marker] = []
        seen_ids = set()
        seen_payloads = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise MarkerMapError(f"marker {index}: entry must be a mapping")
            marker_id = str(entry.get("id", "")).strip()
            if not marker_id:
                raise MarkerMapError(f"marker {index}: missing 'id'")
            if marker_id in seen_ids:
                raise MarkerMapError(f"duplicate marker id: {marker_id}")
            seen_ids.add(marker_id)

            payload = str(entry.get("payload", marker_id))
            if payload in seen_payloads:
                raise MarkerMapError(
                    f"marker {marker_id}: payload {payload!r} is already used by another "
                    "marker. Two markers that decode to the same string cannot be told "
                    "apart in flight."
                )
            if not payload.isascii():
                raise MarkerMapError(
                    f"marker {marker_id}: payload {payload!r} is not ASCII. Non-ASCII "
                    "payloads encode as byte mode and decode inconsistently across "
                    "readers."
                )
            seen_payloads.add(payload)

            size = entry.get("size_m", default_size)
            if size is None:
                raise MarkerMapError(
                    f"marker {marker_id}: no 'size_m' and the file sets no "
                    "'default_size_m'. Marker size scales every position estimate, so "
                    "it is not something to guess a default for."
                )
            size = float(size)
            if not size > 0:
                raise MarkerMapError(f"marker {marker_id}: size_m must be positive")

            position = np.asarray(entry.get("position", []), dtype=float).ravel()
            if position.size != 3:
                raise MarkerMapError(f"marker {marker_id}: 'position' must be [x, y, z]")
            if not np.isfinite(position).all():
                raise MarkerMapError(f"marker {marker_id}: 'position' has a non-finite value")

            markers.append(
                Marker(
                    marker_id=marker_id,
                    payload=payload,
                    size=size,
                    position=position,
                    normal_yaw=float(np.deg2rad(float(entry.get("normal_yaw_deg", 0.0)))),
                    tilt=float(np.deg2rad(float(entry.get("tilt_deg", 0.0)))),
                    roll=float(np.deg2rad(float(entry.get("roll_deg", 0.0)))),
                    survey_sigma=float(entry.get("survey_sigma_m", default_sigma)),
                    metadata={
                        k: v
                        for k, v in entry.items()
                        if k
                        not in {
                            "id",
                            "payload",
                            "size_m",
                            "position",
                            "normal_yaw_deg",
                            "tilt_deg",
                            "roll_deg",
                            "survey_sigma_m",
                        }
                    },
                )
            )

        return MarkerMap(
            markers=markers,
            frame_id=str(data.get("frame_id", "map")),
            survey=survey,
            source_path=source_path,
        )

    def to_dict(self) -> dict:
        """Round-trips through :meth:`from_dict`."""
        return {
            "frame_id": self.frame_id,
            "survey": {
                "method": self.survey.method,
                "instrument": self.survey.instrument,
                "date": self.survey.date,
                "stated_accuracy_m": float(self.survey.stated_accuracy),
                "notes": self.survey.notes,
            },
            "markers": [
                {
                    "id": m.marker_id,
                    "payload": m.payload,
                    "size_m": float(m.size),
                    "position": [float(v) for v in m.position],
                    "normal_yaw_deg": float(np.rad2deg(m.normal_yaw)),
                    "tilt_deg": float(np.rad2deg(m.tilt)),
                    "roll_deg": float(np.rad2deg(m.roll)),
                    "survey_sigma_m": float(m.survey_sigma),
                }
                for m in self.markers
            ],
        }

    def check_geometry(self, min_separation: float = 0.05) -> List[str]:
        """Return a list of human-readable warnings about the layout.

        Not errors: a map can be geometrically awkward and still be a correct
        description of a real room. These are the things that will make
        localisation worse and that the surveyor can usually fix cheaply, so
        ``scripts/validate_map.py`` prints them and does not fail on them.
        """
        warnings: List[str] = []
        positions = self.positions()
        for i in range(len(self.markers)):
            for j in range(i + 1, len(self.markers)):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d < min_separation:
                    warnings.append(
                        f"{self.markers[i].marker_id} and {self.markers[j].marker_id} are "
                        f"{d:.3f} m apart, closer than the {min_separation:.2f} m minimum. "
                        "Nearly coincident markers add points to the pose solve without "
                        "adding geometry, so the solve looks better conditioned than it is."
                    )
        if len(self.markers) >= 3:
            centred = positions - positions.mean(axis=0)
            singular = np.linalg.svd(centred, compute_uv=False)
            if singular[-1] < 1e-3 * max(singular[0], 1e-9):
                warnings.append(
                    "all markers are very close to a single plane. Position along the "
                    "plane normal will be poorly observed whenever only one marker is in "
                    "view."
                )
        for marker in self.markers:
            if marker.position[2] < 0.0:
                warnings.append(
                    f"{marker.marker_id} is below the floor plane at z={marker.position[2]:.3f}"
                )
        return warnings


def markers_in_view(
    markers: Iterable[Marker],
    camera_pose_world: np.ndarray,
    camera,
    margin: float = 8.0,
    max_incidence: float = np.deg2rad(50.0),
    max_range: float = 3.5,
    min_span_px: float = 60.0,
) -> List[Marker]:
    """Which markers a camera at ``camera_pose_world`` could plausibly detect.

    Three tests, in the order that rejects the most for the least work:

    1. the marker's outward normal must face the camera at an incidence below
       ``max_incidence``. Fifty degrees, from ``results/detection_incidence.csv``:
       the measured detection rate is 100 per cent out to 50 degrees and 5 per
       cent at 60, because the finder patterns stop being square and the modules
       compress along one axis. A planner that counts markers it cannot actually
       read will happily pick a heading that sees four of them and localises off
       none;
    2. the centre must be within ``max_range`` and the projected symbol must span
       at least ``min_span_px``. Both from ``results/detection_range.csv``: the
       measured decode rate is 75 to 100 per cent out to 3.8 m, where the symbol
       spans 54 pixels and 15 per cent at 4.0 m. The defaults sit just inside
       that with margin for a marker seen at an angle;
    3. all four corners must project inside the image with ``margin`` pixels to
       spare, since the detector needs the quiet zone.

    This is used by the simulator to decide what to draw and by the mission
    planner to choose a heading and deliberately not by the detector: the real
    detector is given the whole image and finds what it finds.
    """
    T_wc = np.asarray(camera_pose_world, dtype=float)
    camera_position = T_wc[:3, 3]
    R_cw = T_wc[:3, :3].T
    visible: List[Marker] = []
    for marker in markers:
        to_camera = camera_position - marker.position
        distance = float(np.linalg.norm(to_camera))
        if distance > max_range or distance < 1e-6:
            continue
        cos_incidence = float(np.dot(marker.normal_world(), to_camera / distance))
        if cos_incidence < np.cos(max_incidence):
            continue
        corners_camera = (R_cw @ (marker.corners_world() - camera_position).T).T
        pixels = camera.project(corners_camera)
        if not camera.in_view(pixels, margin=margin).all():
            continue
        span = max(
            float(np.linalg.norm(pixels[i] - pixels[(i + 1) % 4])) for i in range(4)
        )
        if span < min_span_px:
            continue
        visible.append(marker)
    return visible


def pixel_span(marker: Marker, camera_pose_world: np.ndarray, camera) -> float:
    """Longest edge of the projected marker, in pixels.

    OpenCV's QR decoder needs roughly five pixels per module and the smallest
    symbol used here is version 1 at 21 modules, so somewhere near 100 pixels of
    span is the practical detection floor. The simulator uses this to decide
    whether a marker it has drawn is realistically decodable and the mission
    planner uses it to prefer headings that put markers close.
    """
    T_wc = np.asarray(camera_pose_world, dtype=float)
    R_cw = T_wc[:3, :3].T
    corners_camera = (R_cw @ (marker.corners_world() - T_wc[:3, 3]).T).T
    pixels = camera.project(corners_camera)
    if not np.isfinite(pixels).all():
        return 0.0
    edges = [
        float(np.linalg.norm(pixels[i] - pixels[(i + 1) % 4])) for i in range(4)
    ]
    return max(edges)


def marker_from_pose(
    marker_id: str,
    payload: str,
    size: float,
    pose_world: np.ndarray,
    survey_sigma: float = 0.005,
) -> Marker:
    """Build a :class:`Marker` from a 4x4 pose rather than from angles.

    Used by the map-estimation half of the evaluation, which recovers where the
    markers appear to be from flight data and compares that with the survey.
    """
    T = np.asarray(pose_world, dtype=float)
    R = T[:3, :3]
    z_axis = R @ np.array([0.0, 0.0, 1.0])
    normal_yaw = float(np.arctan2(z_axis[1], z_axis[0]))
    # Recover tilt and roll by removing the yaw-only part of the nominal
    # construction and reading what is left.
    z_nominal = np.array([np.cos(normal_yaw), np.sin(normal_yaw), 0.0])
    y_nominal = np.array([0.0, 0.0, 1.0])
    base = np.column_stack([np.cross(y_nominal, z_nominal), y_nominal, z_nominal])
    residual = base.T @ R
    tilt = float(np.arctan2(-residual[1, 2], residual[2, 2]))
    roll = float(np.arctan2(-residual[0, 1], residual[0, 0]))
    return Marker(
        marker_id=marker_id,
        payload=payload,
        size=float(size),
        position=T[:3, 3].copy(),
        normal_yaw=normal_yaw,
        tilt=tilt,
        roll=roll,
        survey_sigma=float(survey_sigma),
    )
