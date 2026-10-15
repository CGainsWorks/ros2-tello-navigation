"""Frames, rotations and the conventions everything else assumes.

Small tests but the ones most worth having. Every sign error in a system like
this ends up here and a sign error in a rotation does not produce an obvious
failure: it produces an aircraft that flies well in one axis and diverges in
another, which costs a day to find.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.geometry import (
    R_BC,
    euler_to_matrix,
    invert_transform,
    is_rotation,
    make_transform,
    matrix_to_euler,
    rot_x,
    rot_y,
    rot_z,
    transform_points,
    wrap_angle,
    wrap_angles,
)


def test_body_to_camera_is_a_proper_rotation():
    assert is_rotation(R_BC)


def test_body_to_camera_maps_optical_axis_to_nose():
    """Camera +z is forward along the optical axis; body +x is out of the nose."""
    assert np.allclose(R_BC @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])


def test_body_to_camera_maps_image_right_to_body_left():
    """Camera +x is to the right in the image, which is the body's -y.

    This one sign is why a marker to the aircraft's left appears on the left of
    the frame. Flipping it produces a controller that steers away from every
    marker it sees.
    """
    assert np.allclose(R_BC @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0])


def test_body_to_camera_maps_image_down_to_body_down():
    assert np.allclose(R_BC @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0])


@pytest.mark.parametrize(
    "angle,expected",
    [
        (0.0, 0.0),
        # The interval is half open at the top: exactly pi comes back as -pi.
        (np.pi, -np.pi),
        (-np.pi, -np.pi),
        (6.2, 6.2 - 2 * np.pi),
        (-6.2, 2 * np.pi - 6.2),
    ],
)
def test_wrap_angle(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected, abs=1e-12)


def test_wrap_angle_stays_in_range():
    values = np.linspace(-50.0, 50.0, 997)
    wrapped = np.array([wrap_angle(v) for v in values])
    assert np.all(wrapped >= -np.pi - 1e-12)
    assert np.all(wrapped < np.pi)


def test_wrap_angle_matches_vectorised_version():
    values = np.linspace(-20.0, 20.0, 101)
    assert np.allclose([wrap_angle(v) for v in values], wrap_angles(values))


def test_euler_round_trip():
    for yaw, pitch, roll in [
        (0.0, 0.0, 0.0),
        (0.3, -0.2, 0.1),
        (-2.9, 0.05, -0.4),
        (np.pi / 2, 0.0, 0.0),
    ]:
        R = euler_to_matrix(yaw, pitch, roll)
        assert is_rotation(R)
        back = matrix_to_euler(R)
        assert np.allclose(back, (wrap_angle(yaw), pitch, roll), atol=1e-9)


def test_euler_is_intrinsic_zyx():
    yaw, pitch, roll = 0.4, -0.3, 0.2
    assert np.allclose(
        euler_to_matrix(yaw, pitch, roll), rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    )


def test_matrix_to_euler_is_defined_at_gimbal_lock():
    """Straight up is a degenerate attitude, not an exception.

    A Tello pitched 90 degrees has already failed in a way this function cannot
    rescue but returning NaN there would propagate into the controller and turn
    a bad attitude into a bad command.
    """
    R = euler_to_matrix(0.7, np.pi / 2, 0.0)
    yaw, pitch, roll = matrix_to_euler(R)
    assert np.isfinite([yaw, pitch, roll]).all()
    assert pitch == pytest.approx(np.pi / 2, abs=1e-6)
    assert roll == 0.0


def test_invert_transform_is_the_inverse():
    T = make_transform(euler_to_matrix(0.3, 0.2, -0.1), np.array([1.0, -2.0, 0.5]))
    assert np.allclose(T @ invert_transform(T), np.eye(4), atol=1e-12)


def test_invert_transform_matches_general_inverse():
    T = make_transform(euler_to_matrix(-1.1, 0.4, 0.9), np.array([0.2, 0.3, 1.7]))
    assert np.allclose(invert_transform(T), np.linalg.inv(T), atol=1e-12)


def test_transform_points_round_trip():
    T = make_transform(euler_to_matrix(0.9, -0.1, 0.25), np.array([2.0, 1.0, -0.5]))
    points = np.random.default_rng(0).normal(size=(20, 3))
    assert np.allclose(transform_points(invert_transform(T), transform_points(T, points)), points)


def test_is_rotation_rejects_reflections():
    reflection = np.diag([1.0, 1.0, -1.0])
    assert not is_rotation(reflection)


def test_is_rotation_rejects_scaled_rotations():
    assert not is_rotation(1.01 * euler_to_matrix(0.2))
