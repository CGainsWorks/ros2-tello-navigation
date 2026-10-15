"""Finding and decoding QR markers in a frame.

Two OpenCV detectors exist, they are not equivalent and this module uses both.

The older ``cv2.QRCodeDetector`` finds symbols by thresholding and run-length
analysis. The newer ``cv2.QRCodeDetectorAruco``, added in OpenCV 4.7, locates the
three finder patterns with the ArUco marker detector's adaptive-threshold and
contour machinery and only then decodes.

**Range.** The ArUco-based detector wins and not marginally. Measured on this
project's own renderer, with a 0.30 m symbol, the Tello's nominal 683-pixel focal
length and the default noise and compression, the classic detector stops reading
at about 2.0 m and the ArUco-based one still reads 85 per cent of frames at 3.8 m.
That is the difference between a marker layout that covers a 4 x 3 m room and one
that does not.

**Pairs.** The classic detector wins and also not marginally. The marker layout
puts markers in pairs 45 cm apart, because two markers in one view resolve a pose
ambiguity that one cannot. Shown a pair at 1.5 m, the ArUco-based detector finds
*a* symbol and returns an empty payload for it, reading neither; the classic
detector reads both. That was found by a test that had been passing for an hour
and started failing the moment the second marker was added to the map.

So the ArUco-based detector runs first and the classic one runs as well whenever
the first found a quadrilateral it could not decode. On a frame where everything
decodes, which is most of them, the second detector never runs and costs nothing.
On a frame where it does run, both sets of detections are merged by payload.

``results/detection_range.csv`` holds the measured range curve and
``scripts/measure_detection.py`` regenerates it.

Corner accuracy
---------------
Both detectors return corners to the nearest pixel or so. A pixel of corner error
at 2 m on a 0.30 m marker is roughly 6 mm of position error and it is the
dominant error term in the whole pipeline, so the corners are refined with
``cv2.cornerSubPix`` before they go anywhere near the pose solver. That is worth
about a factor of three on the reprojection residual and it costs well under a
millisecond.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..types import Frame, MarkerObservation


def _make_detector(prefer_aruco: bool = True):
    """Return the primary detector and the name of what was chosen."""
    if prefer_aruco and hasattr(cv2, "QRCodeDetectorAruco"):
        return cv2.QRCodeDetectorAruco(), "aruco"
    return cv2.QRCodeDetector(), "classic"


@dataclass(frozen=True)
class DetectorSettings:
    """Tuning for the detection stage."""

    prefer_aruco: bool = True
    fallback: bool = True
    """Run the second detector when the first returns an undecoded symbol.

    Costs nothing on a frame where everything decodes, which is most of them.
    Turning it off loses closely-spaced marker pairs, which is most of the value
    of the marker layout."""

    subpixel: bool = True
    subpixel_window: int = 5
    """Half-width in pixels of the ``cornerSubPix`` search window. Five is about
    a third of a module at the shortest usable range, which keeps the window
    inside the finder pattern rather than straddling the quiet zone."""

    min_side_px: float = 12.0
    """Reject a detection whose shortest projected edge is below this. The
    decoder occasionally reports a payload for a symbol far too small to have
    genuinely carried one and those detections carry corner errors large enough
    to drag a multi-marker solve badly off."""

    max_aspect: float = 6.0
    """Reject a detection whose longest edge is more than this multiple of its
    shortest. A QR symbol is square, so a wildly non-square quadrilateral is a
    detection artefact, not a marker seen at a steep angle: 6:1 corresponds to
    an incidence of about 80 degrees, beyond which the pose is worthless
    anyway."""


class QrDetector:
    """Detects and decodes every QR symbol in a frame."""

    def __init__(self, settings: Optional[DetectorSettings] = None) -> None:
        self.settings = settings or DetectorSettings()
        self._detector, self.backend = _make_detector(self.settings.prefer_aruco)
        self._fallback = (
            cv2.QRCodeDetector()
            if self.settings.fallback and self.backend == "aruco"
            else None
        )
        self.fallback_frames = 0
        """How many frames needed the second detector. Reported in the flight
        metadata: a run where it fires constantly is a run where the primary
        detector is struggling and that is worth knowing before the accuracy
        numbers are read."""

    def detect(self, frame: Frame) -> List[MarkerObservation]:
        """Return one :class:`~tello_core.types.MarkerObservation` per symbol read.

        Symbols that are found but not decoded are dropped. There is no useful
        way to use a marker whose identity is unknown: without the payload there
        is no map entry, without a map entry there is no world position and a
        quadrilateral with no world position contributes nothing to a pose.
        """
        image = frame.image
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, undecoded = self._run(self._detector, gray, frame.timestamp)
        if undecoded and self._fallback is not None:
            # The primary detector saw a symbol it could not read. That is the
            # signature of a closely-spaced pair, which is exactly what this
            # marker layout is built from, so it is worth the second pass.
            self.fallback_frames += 1
            extra, _ = self._run(self._fallback, gray, frame.timestamp)
            known = {observation.decoded for observation in found}
            found.extend(o for o in extra if o.decoded not in known)
        return found

    def _run(self, detector, gray: np.ndarray, timestamp: float):
        """One detector over one frame. Returns ``(observations, undecoded)``."""
        try:
            ok, decoded, points, _ = detector.detectAndDecodeMulti(gray)
        except cv2.error:
            # Some OpenCV builds raise rather than returning False on a frame
            # with nothing in it.
            return [], 0
        if not ok or points is None:
            return [], 0

        observations: List[MarkerObservation] = []
        undecoded = 0
        for payload, quad in zip(decoded, np.asarray(points, dtype=np.float32)):
            if not payload:
                undecoded += 1
                continue
            corners = quad.reshape(4, 2).astype(np.float32)
            if not self._plausible(corners):
                continue
            if self.settings.subpixel:
                corners = self._refine(gray, corners)
            observations.append(
                MarkerObservation(
                    marker_id=payload,
                    corners=corners.astype(float),
                    timestamp=timestamp,
                    decoded=payload,
                )
            )
        return observations, undecoded

    def _plausible(self, corners: np.ndarray) -> bool:
        edges = np.array(
            [float(np.linalg.norm(corners[i] - corners[(i + 1) % 4])) for i in range(4)]
        )
        if edges.min() < self.settings.min_side_px:
            return False
        if edges.min() <= 0 or edges.max() / edges.min() > self.settings.max_aspect:
            return False
        # A self-intersecting quadrilateral has a signed area far smaller than
        # its edge lengths imply. Cheap check, catches the occasional
        # corner-order failure that would otherwise produce a mirrored pose.
        area = 0.5 * abs(
            float(
                np.dot(corners[:, 0], np.roll(corners[:, 1], -1))
                - np.dot(np.roll(corners[:, 0], -1), corners[:, 1])
            )
        )
        return area > 0.25 * edges.min() ** 2

    def _refine(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        window = int(self.settings.subpixel_window)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        refined = corners.reshape(-1, 1, 2).astype(np.float32).copy()
        try:
            cv2.cornerSubPix(gray, refined, (window, window), (-1, -1), criteria)
        except cv2.error:
            return corners
        refined = refined.reshape(4, 2)
        # cornerSubPix can walk a corner a long way when the window straddles a
        # strong unrelated edge. Anything that moves more than the window is not
        # a refinement, it is a different corner.
        moved = np.linalg.norm(refined - corners, axis=1)
        return np.where((moved < window)[:, None], refined, corners)


def observation_span(observation: MarkerObservation) -> float:
    """Longest edge of a detection in pixels, the usual proxy for how close it is."""
    c = np.asarray(observation.corners, dtype=float)
    return max(float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4))


def filter_known(
    observations: Sequence[MarkerObservation], marker_map
) -> tuple:
    """Split detections into those the map knows about and those it does not.

    Returned as ``(known, unknown_payloads)``. The unknown ones are reported
    rather than discarded silently: a QR code on somebody's laptop lid in the
    corner of the room is a perfectly ordinary thing to see and so is a marker
    whose payload was mistyped in the map and the operator needs to be able to
    tell those two apart.
    """
    known: List[MarkerObservation] = []
    unknown: List[str] = []
    for observation in observations:
        if marker_map.by_payload(observation.decoded) is not None:
            known.append(observation)
        else:
            unknown.append(observation.decoded)
    return known, unknown
