"""The visual-odometry front end.

Driven through the renderer, because the thing being tested is whether real
optical flow over a real textured scene recovers the right velocity. A synthetic
flow field would test the least-squares solve and nothing else and the
least-squares solve was never the risky part.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.geometry import euler_to_matrix, make_transform
from tello_core.localisation.vo import VisualOdometry, VoSettings, plane_depth_from_marker
from tello_core.types import Frame


def _frames(renderer, config, positions, yaw, dt: float = 0.1):
    """Render a sequence of frames along a path."""
    frames = []
    for index, position in enumerate(positions):
        pose = make_transform(euler_to_matrix(yaw), np.asarray(position, float)) @ (
            config.mount.transform()
        )
        frames.append(
            Frame(timestamp=index * dt, image=renderer.render(pose), index=index)
        )
    return frames


def test_first_frame_produces_nothing(clean_renderer, camera, config):
    vo = VisualOdometry(camera)
    frames = _frames(clean_renderer, config, [[2.0, 1.5, 1.3]], np.pi)
    assert vo.process(frames[0], np.zeros(3), depth=2.0) is None


def test_a_stationary_camera_reports_no_motion(clean_renderer, camera, config):
    vo = VisualOdometry(camera)
    positions = [[2.0, 1.5, 1.3]] * 4
    estimate = None
    for frame in _frames(clean_renderer, config, positions, np.pi):
        estimate = vo.process(frame, np.zeros(3), depth=2.0)
    assert estimate is not None and estimate.valid
    assert np.linalg.norm(estimate.velocity_body) < 0.12


@pytest.mark.parametrize(
    "axis,speed",
    [(0, 0.4), (0, -0.4), (1, 0.3), (2, 0.3)],
)
def test_translation_is_recovered_with_the_right_sign(
    clean_renderer, camera, config, axis, speed
):
    """The sign matters more than the magnitude.

    A velocity estimate with the wrong sign is worse than none at all: the filter
    integrates it, so the estimate accelerates away from the truth exactly when
    the aircraft is moving fastest.
    """
    dt = 0.1
    yaw = np.pi
    start = np.array([2.0, 1.5, 1.3])
    # World-frame motion, so a body-forward move at yaw = pi is world -x.
    direction = np.zeros(3)
    direction[axis] = 1.0
    world_direction = euler_to_matrix(yaw) @ direction
    positions = [start + world_direction * speed * dt * step for step in range(5)]

    vo = VisualOdometry(camera)
    estimate = None
    for frame in _frames(clean_renderer, config, positions, yaw, dt=dt):
        estimate = vo.process(frame, np.zeros(3), depth=2.0)
    assert estimate is not None and estimate.valid
    recovered = estimate.velocity_body[axis]
    assert np.sign(recovered) == np.sign(speed)
    assert abs(recovered) > 0.4 * abs(speed)


def test_rotation_alone_is_not_reported_as_translation(clean_renderer, camera, config):
    """De-rotation is the whole trick.

    A camera turning on the spot produces exactly the flow field a camera
    translating sideways produces, to first order. The aircraft measures its own
    yaw rate, so the rotational part is subtracted rather than estimated and
    without that subtraction a turn reads as a two metre per second sideways
    dash.
    """
    dt = 0.1
    rate = np.deg2rad(20.0)
    positions = [[2.0, 1.5, 1.3]] * 5
    yaws = [np.pi + rate * dt * step for step in range(5)]

    vo = VisualOdometry(camera)
    estimate = None
    for index, (position, yaw) in enumerate(zip(positions, yaws)):
        pose = make_transform(euler_to_matrix(yaw), np.asarray(position, float)) @ (
            config.mount.transform()
        )
        frame = Frame(timestamp=index * dt, image=clean_renderer.render(pose), index=index)
        estimate = vo.process(frame, np.array([0.0, 0.0, rate]), depth=2.0)
    assert estimate is not None and estimate.valid
    assert np.linalg.norm(estimate.velocity_body) < 0.25


def test_scale_comes_from_the_depth_hint(clean_renderer, camera, config):
    """Monocular flow gives velocity over depth; the depth is supplied.

    Doubling the stated depth must double the reported speed. That is not a bug,
    it is the entire reason ``VoEstimate.scale_source`` exists: a velocity resting
    on a stale depth is wrong in proportion and the log has to say where the
    depth came from.
    """
    dt = 0.1
    yaw = np.pi
    start = np.array([2.0, 1.5, 1.3])
    positions = [start + np.array([-0.4 * dt * step, 0.0, 0.0]) for step in range(4)]
    frames = _frames(clean_renderer, config, positions, yaw, dt=dt)

    def speed_with(depth):
        vo = VisualOdometry(camera)
        estimate = None
        for frame in frames:
            estimate = vo.process(frame, np.zeros(3), depth=depth)
        return float(np.linalg.norm(estimate.velocity_body))

    assert speed_with(4.0) == pytest.approx(2.0 * speed_with(2.0), rel=0.25)


def test_a_long_gap_resets_the_tracker(clean_renderer, camera, config):
    vo = VisualOdometry(camera, VoSettings(max_dt=0.2))
    frames = _frames(clean_renderer, config, [[2.0, 1.5, 1.3]] * 2, np.pi, dt=5.0)
    vo.process(frames[0], np.zeros(3), depth=2.0)
    assert vo.process(frames[1], np.zeros(3), depth=2.0) is None


def test_reset_forgets_the_previous_frame(clean_renderer, camera, config):
    vo = VisualOdometry(camera)
    frames = _frames(clean_renderer, config, [[2.0, 1.5, 1.3]] * 2, np.pi)
    vo.process(frames[0], np.zeros(3), depth=2.0)
    vo.reset()
    assert vo.process(frames[1], np.zeros(3), depth=2.0) is None


def test_an_absurd_depth_is_refused(clean_renderer, camera, config):
    vo = VisualOdometry(camera)
    estimate = None
    for frame in _frames(clean_renderer, config, [[2.0, 1.5, 1.3]] * 3, np.pi):
        estimate = vo.process(frame, np.zeros(3), depth=0.0)
    assert estimate is not None
    assert not estimate.valid


def test_estimate_records_where_its_scale_came_from(clean_renderer, camera, config):
    vo = VisualOdometry(camera)
    estimate = None
    for frame in _frames(clean_renderer, config, [[2.0, 1.5, 1.3]] * 3, np.pi):
        estimate = vo.process(frame, np.zeros(3), depth=2.0, scale_source="marker")
    assert estimate.scale_source == "marker"
    assert estimate.n_tracks > 0
    assert 0.0 <= estimate.inlier_ratio <= 1.0


def test_plane_depth_from_marker_matches_the_geometry(marker_map, config):
    """The depth hint the mission runner actually uses."""
    marker = marker_map.get("M01")
    position = np.array([marker.position[0] + 1.5, marker.position[1], marker.position[2]])
    depth, normal = plane_depth_from_marker(
        marker, position, np.pi, config.mount.transform()
    )
    assert depth == pytest.approx(1.5 - config.mount.x, abs=0.05)
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)
