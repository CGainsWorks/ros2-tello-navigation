"""The simulated room: textured planes with the surveyed markers printed on them.

The world is six rectangles. That is not a limitation worth apologising for: the
test area this project targets is a room, the markers are taped to its walls
and a planar world is what lets the renderer be an exact ray caster rather than
an approximation. Everything the estimator has to cope with, perspective, oblique
incidence, occlusion of a marker by the frame edge, parallax between near and far
features, is present in a planar world.

Wall texture is procedural and seeded. It exists so that the visual-odometry
front end has something to track: a bare white wall gives Lucas-Kanade nothing
and a VO front end tested against a bare white wall would report a perfect zero
velocity and look excellent. The texture is a mixture of band-limited noise and
hard-edged rectangles, which is a fair caricature of a room with skirting boards,
door frames and posters in it. It is not photorealistic and does not need to be;
what it needs to be is *cornered*, at a spatial frequency a real room has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..localisation.markers import Marker, MarkerMap
from .qr import render_sheet


@dataclass
class Plane:
    """A finite textured rectangle in the world.

    ``origin`` is the world point that the texture's pixel (0, 0) sits on,
    ``u_axis`` runs along the texture's columns and ``v_axis`` along its rows.
    Both axes are unit vectors and they are orthogonal, so the mapping from a
    world point on the plane to a texture pixel is two dot products.
    """

    name: str
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    extent_u: float
    extent_v: float
    texture: np.ndarray
    pixels_per_metre: float

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.u_axis, self.v_axis)
        return n / np.linalg.norm(n)

    def contains_uv(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (u >= 0.0) & (u <= self.extent_u) & (v >= 0.0) & (v <= self.extent_v)

    def world_to_uv(self, points: np.ndarray) -> tuple:
        d = np.asarray(points, dtype=float) - self.origin
        return d @ self.u_axis, d @ self.v_axis

    def uv_to_world(self, u, v) -> np.ndarray:
        return self.origin + np.outer(np.atleast_1d(u), self.u_axis) + np.outer(
            np.atleast_1d(v), self.v_axis
        )

    def paste(self, bitmap: np.ndarray, corners_uv: np.ndarray) -> None:
        """Warp ``bitmap`` onto the texture so its corners land on ``corners_uv``.

        ``corners_uv`` are in metres along the plane axes, in the bitmap's own
        corner order: top-left, top-right, bottom-right, bottom-left. A full
        perspective transform rather than an affine one, because a sheet taped up
        with a roll in it is still exactly planar and its corners in plane
        coordinates are still a parallelogram but going through
        ``getPerspectiveTransform`` costs nothing and removes a whole class of
        "why is it slightly sheared" question.
        """
        src = np.array(
            [
                [0.0, 0.0],
                [bitmap.shape[1] - 1.0, 0.0],
                [bitmap.shape[1] - 1.0, bitmap.shape[0] - 1.0],
                [0.0, bitmap.shape[0] - 1.0],
            ],
            dtype=np.float32,
        )
        dst = (np.asarray(corners_uv, dtype=np.float32) * self.pixels_per_metre).astype(
            np.float32
        )
        transform = cv2.getPerspectiveTransform(src, dst)
        h, w = self.texture.shape[:2]
        warped = cv2.warpPerspective(
            bitmap,
            transform,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpPerspective(
            np.full(bitmap.shape[:2], 255, dtype=np.uint8),
            transform,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        np.copyto(self.texture, warped, where=mask.astype(bool))


def wall_texture(
    height_m: float,
    width_m: float,
    pixels_per_metre: float,
    seed: int,
    kind: str = "wall",
) -> np.ndarray:
    """A seeded procedural texture for one surface.

    Three layers, in increasing spatial frequency:

    * a slow luminance gradient, which is what a room lit from one side looks
      like and what makes a fixed-threshold detector fail on the dark side;
    * hard-edged rectangles, the corners Lucas-Kanade actually locks onto;
    * band-limited noise, which stops large flat regions from being aperture
      problems.

    Reproducible from ``seed`` alone, so the world a result was measured in can
    be rebuilt exactly.
    """
    rng = np.random.default_rng(seed)
    h = max(8, int(round(height_m * pixels_per_metre)))
    w = max(8, int(round(width_m * pixels_per_metre)))

    # Slow gradient, built at low resolution and stretched.
    coarse = rng.uniform(0.55, 0.85, size=(4, 4))
    gradient = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    # Band-limited noise: white noise at a sixteenth scale, bicubic back up. This
    # gives energy at a spatial period of roughly 4 cm on a wall at the default
    # resolution, which is about the scale of the texture in painted plaster.
    noise_small = rng.normal(0.0, 1.0, size=(max(4, h // 16), max(4, w // 16)))
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = noise / (np.abs(noise).max() + 1e-9)

    image = np.clip(gradient + 0.06 * noise, 0.0, 1.0)

    # Hard-edged features. On the floor these are floorboard lines; on walls and
    # ceiling they are panels, posters and a skirting board.
    if kind == "floor":
        spacing = max(6, int(0.15 * pixels_per_metre))
        for y in range(0, h, spacing):
            image[y : y + max(1, spacing // 20), :] *= rng.uniform(0.6, 0.8)
    else:
        n_rects = int(rng.integers(5, 9))
        for _ in range(n_rects):
            rw = int(rng.integers(int(0.15 * pixels_per_metre), int(0.6 * pixels_per_metre)))
            rh = int(rng.integers(int(0.15 * pixels_per_metre), int(0.6 * pixels_per_metre)))
            x0 = int(rng.integers(0, max(1, w - rw)))
            y0 = int(rng.integers(0, max(1, h - rh)))
            image[y0 : y0 + rh, x0 : x0 + rw] *= rng.uniform(0.55, 0.9)
        if kind == "wall":
            skirting = max(2, int(0.08 * pixels_per_metre))
            image[h - skirting :, :] *= 0.55

    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


@dataclass
class Room:
    """A rectangular room with the marker map's markers printed on its surfaces."""

    length_x: float
    length_y: float
    height_z: float
    planes: List[Plane] = field(default_factory=list)
    marker_map: Optional[MarkerMap] = None
    pixels_per_metre: float = 600.0
    seed: int = 7

    @staticmethod
    def from_marker_map(
        marker_map: MarkerMap,
        length_x: float = 4.0,
        length_y: float = 3.0,
        height_z: float = 2.4,
        pixels_per_metre: float = 600.0,
        seed: int = 7,
        module_px: int = 6,
    ) -> "Room":
        """Build the room and print every marker in the map onto it.

        A marker is printed onto whichever surface it lies in, to within 3 cm and
        with its plane normal within 5 degrees of the surface normal. Anything
        else gets its own free-standing quad, which is how a marker on a tripod
        in the middle of the floor would be handled. If a marker matches no
        surface and cannot be placed, that is an error rather than a silently
        invisible marker: a map entry the aircraft can never see is a bug in the
        map and finding it at build time is much cheaper than finding it as an
        unexplained localisation gap halfway through a mission.
        """
        room = Room(
            length_x=float(length_x),
            length_y=float(length_y),
            height_z=float(height_z),
            marker_map=marker_map,
            pixels_per_metre=float(pixels_per_metre),
            seed=int(seed),
        )
        room.planes = room._build_surfaces()
        for index, marker in enumerate(marker_map):
            room._print_marker(marker, module_px=module_px, index=index)
        return room

    def _build_surfaces(self) -> List[Plane]:
        lx, ly, lz = self.length_x, self.length_y, self.height_z
        ppm = self.pixels_per_metre
        ex, ey, ez = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        specs = [
            # name,        origin,                 u_axis, v_axis, extent_u, extent_v, kind
            ("wall_x0", np.array([0.0, 0.0, lz]), ey, -ez, ly, lz, "wall"),
            ("wall_x1", np.array([lx, ly, lz]), -ey, -ez, ly, lz, "wall"),
            ("wall_y0", np.array([lx, 0.0, lz]), -ex, -ez, lx, lz, "wall"),
            ("wall_y1", np.array([0.0, ly, lz]), ex, -ez, lx, lz, "wall"),
            ("floor", np.array([0.0, 0.0, 0.0]), ex, ey, lx, ly, "floor"),
            ("ceiling", np.array([0.0, ly, lz]), ex, -ey, lx, ly, "ceiling"),
        ]
        planes: List[Plane] = []
        for offset, (name, origin, u, v, eu, ev, kind) in enumerate(specs):
            planes.append(
                Plane(
                    name=name,
                    origin=np.asarray(origin, dtype=float),
                    u_axis=np.asarray(u, dtype=float),
                    v_axis=np.asarray(v, dtype=float),
                    extent_u=float(eu),
                    extent_v=float(ev),
                    texture=wall_texture(ev, eu, ppm, self.seed + 101 * offset, kind),
                    pixels_per_metre=ppm,
                )
            )
        return planes

    def _print_marker(self, marker: Marker, module_px: int, index: int) -> None:
        bitmap, symbol_px, total_px = render_sheet(marker.payload, module_px=module_px)
        sheet_size = marker.size * total_px / symbol_px
        half = sheet_size / 2.0
        # Bitmap corner order is top-left, top-right, bottom-right, bottom-left,
        # in the marker frame of tello_core.localisation.markers: x right, y up.
        sheet_corners_marker = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
        )
        corners_world = (marker.rotation @ sheet_corners_marker.T).T + marker.position

        for plane in self.planes:
            u, v = plane.world_to_uv(corners_world)
            out_of_plane = (corners_world - plane.origin) @ plane.normal
            if np.abs(out_of_plane).max() > 0.03:
                continue
            if float(np.abs(np.dot(marker.normal_world(), plane.normal))) < np.cos(
                np.deg2rad(5.0)
            ):
                continue
            # A 5 cm tolerance rather than a hard edge: a sheet whose quiet zone
            # runs a centimetre off the end of a wall is a real thing that
            # happens and the symbol itself is still fully on the surface.
            tol = 0.05
            on_surface = (
                (u >= -tol)
                & (u <= plane.extent_u + tol)
                & (v >= -tol)
                & (v <= plane.extent_v + tol)
            )
            if not on_surface.all():
                continue
            plane.paste(bitmap, np.stack([u, v], axis=1))
            return

        # Not on any surface: give it its own quad, sized to the sheet.
        # Texture rows run downwards, the marker's own y runs upwards, so v is
        # the marker's -y.
        u_axis = marker.rotation @ np.array([1.0, 0.0, 0.0])
        v_axis = marker.rotation @ np.array([0.0, -1.0, 0.0])
        texture = np.full(
            (int(round(sheet_size * self.pixels_per_metre)),) * 2, 255, dtype=np.uint8
        )
        plane = Plane(
            name=f"marker_{marker.marker_id}",
            origin=marker.position - half * u_axis - half * v_axis,
            u_axis=u_axis,
            v_axis=v_axis,
            extent_u=sheet_size,
            extent_v=sheet_size,
            texture=texture,
            pixels_per_metre=self.pixels_per_metre,
        )
        plane.paste(bitmap, np.array([[0.0, 0.0], [sheet_size, 0.0], [sheet_size, sheet_size],
                                      [0.0, sheet_size]]))
        self.planes.append(plane)

    def contains(self, position: Sequence[float], margin: float = 0.0) -> bool:
        """True if a point is inside the room with ``margin`` metres to spare."""
        p = np.asarray(position, dtype=float)
        return bool(
            margin <= p[0] <= self.length_x - margin
            and margin <= p[1] <= self.length_y - margin
            and margin <= p[2] <= self.height_z - margin
        )
