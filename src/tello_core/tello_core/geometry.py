"""Frame conventions and the rotation maths everything else builds on.

Three frames are used and they are not the same handedness, which is the single
most common source of sign errors in this kind of system. They are spelled out
here once so that no other module has to guess.

``world`` (also called ``map``)
    Right-handed, z up. The origin and the x axis are defined by the marker
    survey: see ``maps/marker_map.yaml``. x is "along the long wall", y is 90
    degrees to the left of it, z is up from the floor. This is REP-103 ENU with
    a local origin rather than a geodetic one, which is the correct choice for a
    GPS-denied indoor frame.

``body``
    Attached to the airframe. x forward out of the nose, y to the left, z up.
    Also REP-103. The Tello's own telemetry is *not* in this frame and the
    driver converts, see :mod:`tello_core.link.state`.

``camera``
    OpenCV's convention, because every OpenCV function used here assumes it: z
    forward along the optical axis, x right, y down. The Tello's forward camera
    is rigidly mounted and looks along the body x axis, so the body-to-camera
    rotation is a constant, ``R_BC`` below.

Poses are stored as (position, yaw, pitch, roll) or as 4x4 homogeneous matrices.
The intrinsic Z-Y-X (yaw, then pitch, then roll) Tait-Bryan convention is used
throughout, matching what the Tello reports and what ROS 2's
``tf_transformations.euler_from_quaternion`` returns for ``'rzyx'``.
"""

from __future__ import annotations

import numpy as np

# Rotation taking a vector expressed in the camera frame to the body frame.
#
# Body x (forward) is camera z (forward), body y (left) is camera -x (left is
# minus right), body z (up) is camera -y (up is minus down). Written as columns:
# the first column is where camera x lands in body coordinates and so on.
R_BC = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# The inverse, body to camera. Orthonormal, so the transpose.
R_CB = R_BC.T


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi).

    Half-open at the top, so exactly ``pi`` comes back as ``-pi``. Which end is
    closed does not matter for anything here and the arithmetic is the standard
    one-liner; what matters is that it is stated, because a test that asserts the
    other convention will pass on some inputs and fail on one.

    Every angular difference in this repository goes through here. Yaw error is
    the input to a PID whose output is a yaw rate and a controller that sees a
    6.2 radian error instead of a -0.08 radian one will spin the aircraft all
    the way round the long way.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Vectorised :func:`wrap_angle`."""
    return (np.asarray(angles, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def rot_z(yaw: float) -> np.ndarray:
    """Rotation about the world z axis by ``yaw`` radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(pitch: float) -> np.ndarray:
    """Rotation about the y axis by ``pitch`` radians."""
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(roll: float) -> np.ndarray:
    """Rotation about the x axis by ``roll`` radians."""
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def euler_to_matrix(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Intrinsic Z-Y-X Tait-Bryan angles to a rotation matrix.

    The result rotates a vector from the rotated frame into the reference frame,
    so ``euler_to_matrix(yaw) @ v_body`` gives ``v_world``.
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_euler(R: np.ndarray) -> tuple:
    """Inverse of :func:`euler_to_matrix`, returning ``(yaw, pitch, roll)``.

    Near the gimbal lock at ``|pitch| = pi/2`` the yaw and roll split is not
    unique; roll is set to zero there and the whole rotation about the vertical
    is reported as yaw. A Tello that is pitched 90 degrees has already failed in
    a way this function is not going to rescue, so a defined answer matters more
    than an exact one.
    """
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) > 0.99999:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a homogeneous transform without a general matrix inverse.

    ``T`` is assumed rigid, so the rotation block is orthonormal and the inverse
    is the transpose. Using ``np.linalg.inv`` here would work but would quietly
    absorb a non-rigid matrix instead of producing nonsense and a non-rigid
    matrix at this point is always a bug worth seeing.
    """
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return (np.asarray(T, dtype=float)[:3, :3] @ points.T).T + np.asarray(T, dtype=float)[:3, 3]


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """True if ``R`` is a proper rotation matrix to within ``tol``."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    orthonormal = np.allclose(R.T @ R, np.eye(3), atol=tol)
    proper = abs(float(np.linalg.det(R)) - 1.0) < tol
    return bool(orthonormal and proper)
