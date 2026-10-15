"""Visual odometry: body-frame velocity from de-rotated optical flow.

This is a velocity front end, not a pose front end and that is a deliberate
choice about what a monocular camera on a Tello can actually contribute.

A conventional monocular VO pipeline estimates an essential matrix, decomposes it
into a rotation and a *unit* translation and accumulates poses. It has no scale
of its own, it drifts and its drift is unbounded. Here there is already something
that measures absolute pose properly, the markers and something that measures
attitude properly, the aircraft. What is missing is a velocity measurement good
enough to carry the estimate through the gaps between marker fixes, because the
only alternative is integrating a 10 Hz accelerometer and that is not good
enough. So the front end is built to produce exactly that and nothing more.

The method
----------
Track corners between consecutive frames with pyramidal Lucas-Kanade. For a
camera moving with linear velocity ``v`` and angular velocity ``omega``, the flow
at a point with normalised image coordinates ``(x, y)`` and depth ``Z`` is::

    xdot = (-vx + x vz) / Z  +  x y wx - (1 + x^2) wy + y wz
    ydot = (-vy + y vz) / Z  +  (1 + y^2) wx - x y wy - x wz

The rotational half depends only on the angular velocity, which the aircraft
measures, so it is subtracted rather than estimated. What is left is linear in
``v`` given ``Z`` and is solved for by iteratively reweighted least squares over
every track at once.

Depth
-----
Scale has to come from somewhere and this is where. The caller supplies the
distance to the dominant plane and, optionally, its normal in the camera frame,
both of which the system already knows: the markers are on walls, the map says
where the walls are and the filter says where the aircraft is. Per-track depth
then follows from the plane, which handles a wall seen at an angle correctly
rather than pretending the scene is fronto-parallel.

``VoEstimate.scale_source`` records where the depth came from, so a velocity
resting on a stale depth can be identified in the log afterwards rather than
quietly widening the error bars on everything.

What this does not do
---------------------
It does not detect that the scene is not the plane it was told about. Fly at a
wall with a table in front of it and the estimate is biased by roughly the ratio
of the two depths. The inlier ratio catches gross violations, because a second
depth layer produces flow the single-plane model cannot fit and the estimate is
down-weighted accordingly. It is not a substitute for knowing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..camera import CameraModel
from ..geometry import R_BC
from ..types import Frame, VoEstimate


@dataclass(frozen=True)
class VoSettings:
    """Tuning for the front end."""

    max_features: int = 220
    min_features: int = 40
    """Below this the tracker is re-seeded from a fresh corner detection."""

    quality_level: float = 0.01
    min_distance: float = 12.0
    """Pixels between detected corners. Large enough that the tracks spread over
    the frame rather than clustering on the one high-contrast object in it,
    which matters because a cluster of tracks in one corner gives a velocity
    estimate that is well conditioned in the least-squares sense and wrong."""

    lk_window: int = 21
    lk_levels: int = 3
    forward_backward_threshold: float = 1.0
    """Pixels. A track is kept only if tracking it back to the previous frame
    lands within this of where it started. This is the single most effective
    outlier rejection available here and it costs one more Lucas-Kanade call."""

    max_flow_px: float = 90.0
    """Flow beyond this in one frame interval is not a track, it is a mismatch."""

    huber_delta: float = 1.5
    """Robust-loss transition, in units of the residual's own median absolute
    deviation."""

    iterations: int = 4
    min_tracks: int = 12
    max_dt: float = 0.25
    """Seconds. A gap longer than this between frames means dropped video and
    Lucas-Kanade will have lost the tracks anyway."""


class VisualOdometry:
    """Frame-to-frame velocity estimation.

    Stateful across frames by necessity: it holds the previous image and the
    tracked points. It holds nothing else and in particular no accumulated pose,
    so there is nothing in it that can drift.
    """

    def __init__(self, camera: CameraModel, settings: Optional[VoSettings] = None) -> None:
        self.camera = camera
        self.settings = settings or VoSettings()
        self._previous: Optional[np.ndarray] = None
        self._previous_time: Optional[float] = None
        self._points: Optional[np.ndarray] = None
        self._lk_params = dict(
            winSize=(self.settings.lk_window, self.settings.lk_window),
            maxLevel=self.settings.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self) -> None:
        """Forget the previous frame. Called after a video dropout."""
        self._previous = None
        self._previous_time = None
        self._points = None

    def process(
        self,
        frame: Frame,
        angular_velocity_body: np.ndarray,
        depth: float,
        plane_normal_camera: Optional[np.ndarray] = None,
        scale_source: str = "filter",
    ) -> Optional[VoEstimate]:
        """One frame in, one velocity out, or ``None`` on the first frame.

        ``angular_velocity_body`` is in the body frame, rad/s. ``depth`` is the
        distance along the optical axis to the dominant plane, metres.
        """
        gray = frame.image if frame.image.ndim == 2 else cv2.cvtColor(
            frame.image, cv2.COLOR_BGR2GRAY
        )

        if self._previous is None or self._previous_time is None:
            self._seed(gray, frame.timestamp)
            return None

        dt = float(frame.timestamp) - float(self._previous_time)
        if dt <= 0.0 or dt > self.settings.max_dt:
            self._seed(gray, frame.timestamp)
            return None

        if self._points is None or len(self._points) < self.settings.min_tracks:
            self._detect(self._previous)
        if self._points is None or len(self._points) < self.settings.min_tracks:
            self._seed(gray, frame.timestamp)
            return None

        previous_points, current_points = self._track(self._previous, gray)
        n_tracked = 0 if current_points is None else len(current_points)
        if n_tracked < self.settings.min_tracks:
            self._seed(gray, frame.timestamp)
            return VoEstimate(
                timestamp=float(frame.timestamp),
                velocity_body=np.zeros(3),
                n_tracks=n_tracked,
                inlier_ratio=0.0,
                scale_source=scale_source,
                valid=False,
            )

        velocity_camera, inlier_ratio = self._solve(
            previous_points,
            current_points,
            dt,
            np.asarray(angular_velocity_body, dtype=float),
            float(depth),
            plane_normal_camera,
        )

        self._previous = gray
        self._previous_time = float(frame.timestamp)
        self._points = current_points.reshape(-1, 1, 2)
        if len(self._points) < self.settings.min_features:
            self._detect(gray)

        if velocity_camera is None:
            return VoEstimate(
                timestamp=float(frame.timestamp),
                velocity_body=np.zeros(3),
                n_tracks=n_tracked,
                inlier_ratio=0.0,
                scale_source=scale_source,
                valid=False,
            )

        return VoEstimate(
            timestamp=float(frame.timestamp),
            velocity_body=R_BC @ velocity_camera,
            n_tracks=n_tracked,
            inlier_ratio=float(inlier_ratio),
            scale_source=scale_source,
            valid=True,
        )

    # ------------------------------------------------------------------ tracking

    def _seed(self, gray: np.ndarray, timestamp: float) -> None:
        self._previous = gray
        self._previous_time = float(timestamp)
        self._detect(gray)

    def _detect(self, gray: np.ndarray) -> None:
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.settings.max_features,
            qualityLevel=self.settings.quality_level,
            minDistance=self.settings.min_distance,
            blockSize=7,
        )
        self._points = None if corners is None else corners.astype(np.float32)

    def _track(self, previous: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points = self._points.astype(np.float32)
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points, None, **self._lk_params
        )
        if forward is None:
            return np.empty((0, 2)), np.empty((0, 2))
        backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, forward, None, **self._lk_params
        )
        if backward is None:
            return np.empty((0, 2)), np.empty((0, 2))

        p0 = points.reshape(-1, 2)
        p1 = forward.reshape(-1, 2)
        p0b = backward.reshape(-1, 2)
        keep = (
            (status.ravel() == 1)
            & (status_back.ravel() == 1)
            & (np.linalg.norm(p0b - p0, axis=1) < self.settings.forward_backward_threshold)
            & (np.linalg.norm(p1 - p0, axis=1) < self.settings.max_flow_px)
        )
        return p0[keep], p1[keep]

    # -------------------------------------------------------------------- solve

    def _solve(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        dt: float,
        omega_body: np.ndarray,
        depth: float,
        plane_normal_camera: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], float]:
        if depth <= 0.05 or len(p0) < self.settings.min_tracks:
            return None, 0.0

        fx, fy = self.camera.fx, self.camera.fy
        x0 = (p0[:, 0] - self.camera.cx) / fx
        y0 = (p0[:, 1] - self.camera.cy) / fy
        x1 = (p1[:, 0] - self.camera.cx) / fx
        y1 = (p1[:, 1] - self.camera.cy) / fy

        # Flow in normalised coordinates per second.
        xdot = (x1 - x0) / dt
        ydot = (y1 - y0) / dt

        # Rotational component, removed rather than estimated.
        omega = R_BC.T @ np.asarray(omega_body, dtype=float)
        wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
        xdot -= x0 * y0 * wx - (1.0 + x0**2) * wy + y0 * wz
        ydot -= (1.0 + y0**2) * wx - x0 * y0 * wy - x0 * wz

        # Per-track depth from the plane. With a fronto-parallel default this is
        # a constant, which is the usual assumption; with a real normal it is the
        # correct depth for a wall seen at an angle.
        if plane_normal_camera is None:
            inverse_depth = np.full(x0.shape, 1.0 / depth)
        else:
            normal = np.asarray(plane_normal_camera, dtype=float)
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            rays = np.stack([x0, y0, np.ones_like(x0)], axis=1)
            projection = rays @ normal
            with np.errstate(divide="ignore", invalid="ignore"):
                z = depth / projection
            valid = np.isfinite(z) & (z > 0.05) & (z < 25.0)
            if valid.sum() < self.settings.min_tracks:
                return None, 0.0
            x0, y0, xdot, ydot, z = x0[valid], y0[valid], xdot[valid], ydot[valid], z[valid]
            inverse_depth = 1.0 / z

        n = x0.shape[0]
        A = np.zeros((2 * n, 3))
        A[0::2, 0] = -inverse_depth
        A[0::2, 2] = x0 * inverse_depth
        A[1::2, 1] = -inverse_depth
        A[1::2, 2] = y0 * inverse_depth
        b = np.empty(2 * n)
        b[0::2] = xdot
        b[1::2] = ydot

        weights = np.ones(2 * n)
        velocity = np.zeros(3)
        for _ in range(self.settings.iterations):
            sqrt_w = np.sqrt(weights)
            try:
                velocity, *_ = np.linalg.lstsq(A * sqrt_w[:, None], b * sqrt_w, rcond=None)
            except np.linalg.LinAlgError:
                return None, 0.0
            residual = np.abs(A @ velocity - b)
            scale = 1.4826 * np.median(residual) + 1e-9
            normalised = residual / scale
            weights = np.where(
                normalised <= self.settings.huber_delta,
                1.0,
                self.settings.huber_delta / np.maximum(normalised, 1e-9),
            )

        inlier_ratio = float(np.mean(weights > 0.999))
        if not np.all(np.isfinite(velocity)):
            return None, 0.0
        return velocity, inlier_ratio


def plane_depth_from_marker(
    marker, body_position: np.ndarray, body_yaw: float, T_bc: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Depth and camera-frame normal of the wall a marker is on.

    The obvious source of scale during a mission: the aircraft is looking at a
    marker, the marker is on a wall, the map says where that wall is. Returns
    ``(depth, normal_camera)`` where depth is measured along the optical axis.
    """
    from ..geometry import euler_to_matrix, invert_transform, make_transform

    T_wb = make_transform(euler_to_matrix(float(body_yaw)), np.asarray(body_position, dtype=float))
    T_cw = invert_transform(T_wb @ np.asarray(T_bc, dtype=float))
    normal_camera = T_cw[:3, :3] @ marker.normal_world()
    centre_camera = T_cw[:3, :3] @ marker.position + T_cw[:3, 3]
    depth = float(np.dot(centre_camera, normal_camera))
    return abs(depth), normal_camera * np.sign(depth if depth != 0 else 1.0)
