"""Shared fixtures.

Session-scoped where construction is expensive. Building the room bakes ten QR
symbols into six wall textures at 600 pixels per metre, which takes about half a
second; doing it per test would dominate the suite's runtime and would not test
anything, because the room is immutable once built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src" / "tello_core", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tello_core.camera import CameraModel  # noqa: E402
from tello_core.config import AircraftConfig  # noqa: E402
from tello_core.localisation.markers import MarkerMap  # noqa: E402
from tello_core.sim.render import Renderer, RenderSettings  # noqa: E402
from tello_core.sim.world import Room  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def marker_map() -> MarkerMap:
    return MarkerMap.from_yaml(REPO_ROOT / "maps" / "marker_map.yaml")


@pytest.fixture(scope="session")
def config() -> AircraftConfig:
    return AircraftConfig.from_yaml(
        REPO_ROOT / "src" / "tello_bringup" / "config" / "aircraft.yaml"
    )


@pytest.fixture(scope="session")
def camera() -> CameraModel:
    return CameraModel.tello_nominal()


@pytest.fixture(scope="session")
def room(marker_map: MarkerMap) -> Room:
    return Room.from_marker_map(marker_map)


@pytest.fixture(scope="session")
def renderer(room: Room, camera: CameraModel) -> Renderer:
    """The default renderer, with every image degradation on.

    Deliberately not a clean one. A detection test against a noiseless,
    unblurred, uncompressed image measures nothing that would survive contact
    with a real camera and the point of these tests is to catch the day the
    detector stops working on realistic frames.
    """
    return Renderer(room, camera, RenderSettings(seed=11))


@pytest.fixture(scope="session")
def clean_renderer(room: Room, camera: CameraModel) -> Renderer:
    """No noise, no blur, no compression. For geometry assertions only.

    Used where the quantity under test is the projection itself, so that a
    failure means the geometry is wrong rather than that a random draw was
    unlucky.
    """
    return Renderer(
        room,
        camera,
        RenderSettings(
            motion_blur=False, vignette=0.0, noise_sigma=0.0, jpeg_quality=0, seed=0
        ),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
