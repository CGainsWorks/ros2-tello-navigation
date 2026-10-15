"""Rendering a camera frame from the planar room.

An exact ray caster rather than a rasteriser. For every pixel it intersects the
ray with every plane, keeps the nearest hit in front of the camera and samples
that plane's texture bilinearly. With six planes and a few hundred thousand
pixels that is a handful of vectorised numpy operations and one ``cv2.remap`` per
plane, which comes out in the low tens of milliseconds and is exact: no polygon
clipping, no z-fighting and no seams at plane boundaries.

Correct occlusion matters more here than speed. The most common way a marker
fails to be seen in a real room is that something is in the way and a renderer
that draws every marker regardless would make the estimator look much better than
it is.

After geometry, the frame goes through a small chain of image degradations, in
the order a real camera applies them:

1. **Motion blur**, from the camera's own movement during the exposure. This is
   the one that actually costs detections on a Tello: the aircraft yaws quickly
   and a 1/60 s exposure smears a marker across several pixels.
2. **Vignetting and exposure**, a slow multiplicative field.
3. **Sensor noise**, Gaussian, scaled by the square root of the signal so that
   dark regions are noisier, which is what a small sensor does.
4. **Compression**, as a JPEG round trip. The Tello's video arrives as H.264 at a
   low bitrate and its blocking artefacts sit right on top of the QR module grid.
   JPEG is not H.264 but it puts 8x8 block artefacts in the same place and the
   alternative is either a real encoder in the loop or pretending the stream is
   lossless.

All four are configurable and all four are on by default, because a simulator
whose frames are cleaner than the real thing produces a detection rate that means
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..camera import CameraModel
from .world import Room


@dataclass(frozen=True)
class RenderSettings:
    """The image-degradation chain. Defaults are the ones used for all results."""

    motion_blur: bool = True
    max_blur_px: float = 12.0
    """Cap on the blur kernel, so a spin does not turn into a 200-pixel smear
    that costs more to render than the rest of the frame."""

    vignette: float = 0.35
    """Fractional falloff at the image corners. Zero disables."""

    exposure: float = 1.0
    noise_sigma: float = 3.0
    """Grey levels at mid-signal."""

    jpeg_quality: int = 62
    """0 disables the compression stage."""

    seed: int = 0


class Renderer:
    """Renders frames of a :class:`~tello_core.sim.world.Room`."""

    def __init__(
        self,
        room: Room,
        camera: CameraModel,
        settings: Optional[RenderSettings] = None,
    ) -> None:
        self.room = room
        self.camera = camera
        self.settings = settings or RenderSettings()
        self._rng = np.random.default_rng(self.settings.seed)
        self._rays_camera = self._build_ray_grid(camera)
        self._vignette = self._build_vignette(camera, self.settings.vignette)

    @staticmethod
    def _build_ray_grid(camera: CameraModel) -> np.ndarray:
        """Unit-z direction for every pixel, in the camera frame, ``(H, W, 3)``.

        Built once. This is the expensive part of a ray caster and it does not
        depend on the pose, so paying for it per frame would be most of the
        render time for nothing.

        Distortion is inverted numerically by ``cv2.undistortPoints`` so that the
        rendered image carries the same lens distortion the pose solver will
        later have to undo. With the nominal zero-distortion model this is an
        identity but it means switching to a measured calibration changes the
        rendered image as well as the solver, which is the only way the pair can
        stay consistent.
        """
        h, w = camera.height, camera.width
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(-1, 1, 2)
        normalised = cv2.undistortPoints(
            pixels,
            camera.matrix.astype(np.float64),
            np.asarray(camera.distortion, dtype=np.float64),
        ).reshape(-1, 2)
        rays = np.concatenate([normalised, np.ones((normalised.shape[0], 1))], axis=1)
        return rays.reshape(h, w, 3).astype(np.float64)

    @staticmethod
    def _build_vignette(camera: CameraModel, strength: float) -> Optional[np.ndarray]:
        if strength <= 0.0:
            return None
        h, w = camera.height, camera.width
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(xx - camera.cx, yy - camera.cy)
        r = r / r.max()
        return (1.0 - strength * r**2).astype(np.float32)

    def render(
        self,
        pose_world: np.ndarray,
        angular_velocity: Optional[np.ndarray] = None,
        linear_velocity: Optional[np.ndarray] = None,
        exposure_time: float = 1.0 / 60.0,
    ) -> np.ndarray:
        """Render one BGR frame with the camera at ``pose_world``.

        ``pose_world`` is the 4x4 camera-to-world transform, so its translation
        is the optical centre and its rotation takes camera-frame vectors to the
        world.
        """
        T = np.asarray(pose_world, dtype=float)
        R_wc = T[:3, :3]
        centre = T[:3, 3]

        gray = self._cast(R_wc, centre)

        if self.settings.motion_blur:
            gray = self._apply_motion_blur(
                gray, R_wc, angular_velocity, linear_velocity, exposure_time
            )

        image = gray.astype(np.float32)
        if self._vignette is not None:
            image *= self._vignette
        image *= self.settings.exposure

        if self.settings.noise_sigma > 0:
            # Signal-dependent: sigma scales with the square root of the level,
            # normalised so that the configured figure applies at mid grey.
            #
            # Drawn as float32. The default float64 draw allocates twice the
            # memory and then has to be narrowed again to add to the image and
            # at 700,000 pixels a frame that alone was a tenth of the render time.
            scale = np.sqrt(np.clip(image, 1.0, 255.0) / 128.0, dtype=np.float32)
            noise = self._rng.standard_normal(image.shape, dtype=np.float32)
            image += noise * (np.float32(self.settings.noise_sigma) * scale)

        gray = np.clip(image, 0, 255).astype(np.uint8)

        if self.settings.jpeg_quality > 0:
            ok, encoded = cv2.imencode(
                ".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.jpeg_quality)]
            )
            if ok:
                gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _cast(self, R_wc: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """The geometry pass: nearest-hit ray casting into the plane textures.

        Written to avoid ever materialising the intersection points. The obvious
        form computes ``X = C + t * d`` for every pixel and then dots ``X`` with
        the plane axes, which allocates two more float arrays the size of the
        image per plane. Substituting and collecting terms gives the plane
        coordinates directly::

            u = (C - origin) . u_axis + t * (d . u_axis)

        so the two ray-axis dot products are computed once per plane and the rest
        is elementwise. On a 960x720 frame with six planes that is the difference
        between 90 ms and 30 ms, which over a two-minute mission at 30 Hz is four
        minutes of wall clock.
        """
        h, w = self.camera.height, self.camera.width
        rays_world = (self._rays_camera @ R_wc.T).astype(np.float32)

        best_depth = np.full((h, w), np.inf, dtype=np.float32)
        out = np.zeros((h, w), dtype=np.uint8)

        for plane in self.room.planes:
            normal = plane.normal.astype(np.float32)
            denominator = rays_world @ normal
            numerator = np.float32(np.dot(plane.origin - centre, plane.normal))
            with np.errstate(divide="ignore", invalid="ignore"):
                t = numerator / denominator
            hit = np.isfinite(t) & (t > 1e-3) & (t < best_depth)
            if not hit.any():
                continue

            offset = (centre - plane.origin).astype(np.float32)
            u = np.float32(np.dot(offset, plane.u_axis)) + t * (
                rays_world @ plane.u_axis.astype(np.float32)
            )
            v = np.float32(np.dot(offset, plane.v_axis)) + t * (
                rays_world @ plane.v_axis.astype(np.float32)
            )
            hit &= (u >= 0.0) & (u <= plane.extent_u) & (v >= 0.0) & (v <= plane.extent_v)
            if not hit.any():
                continue

            map_x = np.ascontiguousarray(u * np.float32(plane.pixels_per_metre))
            map_y = np.ascontiguousarray(v * np.float32(plane.pixels_per_metre))
            sampled = cv2.remap(
                plane.texture,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            out = np.where(hit, sampled, out)
            best_depth = np.where(hit, t, best_depth)

        return out

    def _apply_motion_blur(
        self,
        image: np.ndarray,
        R_wc: np.ndarray,
        angular_velocity: Optional[np.ndarray],
        linear_velocity: Optional[np.ndarray],
        exposure_time: float,
    ) -> np.ndarray:
        """Linear blur along the image-space motion at the principal point.

        A single global kernel, which is an approximation: true motion blur from
        a translating camera varies with depth across the frame. It is the right
        approximation for this world, where the dominant motion during a 1/60 s
        exposure is yaw and rotational blur genuinely is uniform to first order.
        """
        if angular_velocity is None and linear_velocity is None:
            return image

        omega_world = np.zeros(3) if angular_velocity is None else np.asarray(angular_velocity)
        velocity_world = np.zeros(3) if linear_velocity is None else np.asarray(linear_velocity)
        omega_camera = R_wc.T @ omega_world
        velocity_camera = R_wc.T @ velocity_world

        # Image-plane shift at the principal point over one exposure. The
        # rotational term is exact at the centre; the translational term needs a
        # depth and the mean scene depth is a fair stand-in given the blur is
        # applied globally anyway.
        depth = 2.0
        shift_x = (
            -self.camera.fx * omega_camera[1] + self.camera.fx * velocity_camera[0] / depth
        ) * exposure_time
        shift_y = (
            self.camera.fy * omega_camera[0] + self.camera.fy * velocity_camera[1] / depth
        ) * exposure_time
        length = float(np.hypot(shift_x, shift_y))
        if length < 1.0:
            return image
        length = min(length, self.settings.max_blur_px)

        size = int(np.ceil(length)) | 1
        kernel = np.zeros((size, size), dtype=np.float32)
        centre = size // 2
        direction = np.array([shift_x, shift_y])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        for step in np.linspace(-length / 2.0, length / 2.0, num=max(3, size)):
            x = int(round(centre + direction[0] * step))
            y = int(round(centre + direction[1] * step))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] += 1.0
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
