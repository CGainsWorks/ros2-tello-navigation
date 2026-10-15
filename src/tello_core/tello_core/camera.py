"""The pinhole camera model and where the Tello's numbers come from.

One class, :class:`CameraModel`, wraps the intrinsic matrix and the distortion
coefficients and is the only place in the repository that knows the image size.
Both the renderer in :mod:`tello_core.sim.camera` and the pose solver in
:mod:`tello_core.localisation.pnp` take the same instance, which is what makes
the simulated pipeline a genuine test of the real one: if the intrinsics are
wrong the pose is wrong and the error shows up as reprojection residual exactly
as it would on hardware.

Provenance of the shipped intrinsics
------------------------------------
The default returned by :meth:`CameraModel.tello_nominal` is *derived, not
measured*. The Tello's forward camera streams 960x720 and DJI publish an 82.6
degree diagonal field of view. For a pinhole model with square pixels and the
principal point at the image centre that fixes the focal length::

    diagonal = sqrt(960^2 + 720^2) = 1200 px
    f = (1200 / 2) / tan(82.6 deg / 2) = 683.0 px

Distortion is set to zero in the nominal model because there is no published
figure for it and inventing one would be worse than admitting there is none. The
Tello's lens does have visible barrel distortion at the frame edges, so a real
flight should use a measured calibration: ``scripts/calibrate_camera.py`` takes a
chessboard recording and writes the same YAML format. Until that is run against
hardware the nominal model is what is used and ``CameraModel.source`` says so
in every log line that quotes a pose.

See ``docs/HARDWARE.md`` for the full list of what this repository asserts about
the aircraft without having been able to measure it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics, distortion and image size for one camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray
    """OpenCV's ``(k1, k2, p1, p2, k3)``, length 5."""

    source: str = "unknown"
    """Where these numbers came from. ``"nominal-fov"`` for the derived model,
    ``"chessboard"`` for a measured one, or the path of the YAML they were read
    from. Carried into the results metadata so a reported accuracy can never be
    read without knowing which calibration produced it."""

    @staticmethod
    def tello_nominal() -> "CameraModel":
        """The derived model described in the module docstring."""
        diagonal = float(np.hypot(960.0, 720.0))
        f = (diagonal / 2.0) / np.tan(np.deg2rad(82.6) / 2.0)
        return CameraModel(
            fx=f,
            fy=f,
            cx=(960.0 - 1.0) / 2.0,
            cy=(720.0 - 1.0) / 2.0,
            width=960,
            height=720,
            distortion=np.zeros(5),
            source="nominal-fov",
        )

    @staticmethod
    def from_yaml(path) -> "CameraModel":
        """Load a calibration in the format ``scripts/calibrate_camera.py`` writes."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        matrix = np.asarray(data["camera_matrix"], dtype=float).reshape(3, 3)
        distortion = np.asarray(data.get("distortion_coefficients", [0.0] * 5), dtype=float)
        if distortion.size < 5:
            distortion = np.pad(distortion.ravel(), (0, 5 - distortion.size))
        return CameraModel(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            width=int(data["image_width"]),
            height=int(data["image_height"]),
            distortion=distortion.ravel()[:5],
            source=str(data.get("source", str(path))),
        )

    def to_dict(self) -> dict:
        """The YAML-serialisable form, matching what :meth:`from_yaml` reads."""
        return {
            "image_width": int(self.width),
            "image_height": int(self.height),
            "camera_matrix": [float(v) for v in self.matrix.ravel()],
            "distortion_coefficients": [float(v) for v in np.asarray(self.distortion).ravel()],
            "source": self.source,
        }

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 intrinsic matrix ``K``."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def horizontal_fov(self) -> float:
        """Radians."""
        return 2.0 * float(np.arctan(self.width / (2.0 * self.fx)))

    @property
    def vertical_fov(self) -> float:
        """Radians."""
        return 2.0 * float(np.arctan(self.height / (2.0 * self.fy)))

    def scaled(self, factor: float) -> "CameraModel":
        """The same camera at a different resolution.

        The simulator renders at a reduced resolution when a test does not need
        full frames and a scaled *image* with unscaled intrinsics is a silent
        way to get poses that are wrong by exactly the scale factor. Going
        through this method makes that impossible to do by accident.
        """
        if factor <= 0:
            raise ValueError("scale factor must be positive")
        return CameraModel(
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=(self.cx + 0.5) * factor - 0.5,
            cy=(self.cy + 0.5) * factor - 0.5,
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
            distortion=np.asarray(self.distortion, dtype=float).copy(),
            source=f"{self.source}+scale{factor:g}",
        )

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        """Project ``(N, 3)`` camera-frame points to ``(N, 2)`` pixels.

        Points at or behind the image plane are projected to NaN rather than to
        a plausible-looking pixel. A marker behind the camera that projects to a
        valid coordinate is the classic way to get a confident, badly wrong
        pose and NaN propagates into the visibility test instead of hiding.
        """
        points = np.atleast_2d(np.asarray(points_camera, dtype=float))
        z = points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = points[:, 0] / z
            y = points[:, 1] / z
        behind = z <= 1e-6
        x = np.where(behind, np.nan, x)
        y = np.where(behind, np.nan, y)
        x, y = self._distort(x, y)
        return np.stack([self.fx * x + self.cx, self.fy * y + self.cy], axis=1)

    def _distort(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Apply the Brown-Conrady model to normalised coordinates."""
        k1, k2, p1, p2, k3 = [float(v) for v in np.asarray(self.distortion).ravel()[:5]]
        if k1 == k2 == p1 == p2 == k3 == 0.0:
            return x, y
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return x_d, y_d

    def in_view(self, pixels: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of which ``(N, 2)`` pixels fall inside the image.

        ``margin`` shrinks the accepted region, in pixels. The detector needs a
        quiet zone around a QR symbol, so a marker whose corners are exactly on
        the frame edge will be found by the projector and not by OpenCV; the
        callers that care pass a margin rather than discovering this as a
        mysterious detection-rate shortfall.
        """
        pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
        inside = (
            (pixels[:, 0] >= margin)
            & (pixels[:, 0] <= self.width - 1 - margin)
            & (pixels[:, 1] >= margin)
            & (pixels[:, 1] <= self.height - 1 - margin)
        )
        return inside & np.isfinite(pixels).all(axis=1)
