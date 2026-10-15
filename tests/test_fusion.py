"""The fusion filter.

Driven with synthetic telemetry and synthetic fixes rather than through the
renderer, so that a failure here is a failure of the filter rather than of
anything upstream of it. The renderer-driven end-to-end check is in
``test_mission.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.geometry import euler_to_matrix
from tello_core.localisation.fusion import GRAVITY, Ekf, FusionSettings
from tello_core.types import MarkerFix, TelloState, VoEstimate


def _telemetry(
    timestamp: float,
    accel_body=None,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    velocity_body=(0.0, 0.0, 0.0),
    height: float = 1.2,
    battery: float = 0.9,
) -> TelloState:
    """A packet for a level aircraft, with gravity already in the accelerometer."""
    if accel_body is None:
        accel_body = euler_to_matrix(yaw, pitch, roll).T @ np.array([0.0, 0.0, GRAVITY])
    return TelloState(
        timestamp=timestamp,
        accel=np.asarray(accel_body, dtype=float),
        gyro_z=0.0,
        attitude=np.array([yaw, pitch, roll]),
        velocity_body=np.asarray(velocity_body, dtype=float),
        height=height,
        barometer=height,
        battery=battery,
        temperature=65.0,
        valid=True,
    )


def _fix(timestamp: float, position, yaw: float = 0.0, sigma: float = 0.02) -> MarkerFix:
    return MarkerFix(
        timestamp=timestamp,
        position=np.asarray(position, dtype=float),
        yaw=float(yaw),
        marker_ids=("M01",),
        reprojection_rmse=0.5,
        position_covariance=np.eye(3) * sigma**2,
        yaw_variance=np.deg2rad(2.0) ** 2,
        n_points=4,
    )


def _settle(filt: Ekf, position, duration: float = 2.0, dt: float = 0.1) -> None:
    """Feed matched telemetry and fixes until the filter has converged."""
    t = filt.timestamp or 0.0
    steps = int(duration / dt)
    for _ in range(steps):
        t += dt
        filt.predict(_telemetry(t))
        filt.update_marker_fix(_fix(t, position))


# ------------------------------------------------------------- initialisation


def test_starts_unlocalised():
    filt = Ekf()
    assert not filt.is_localised()
    assert filt.time_since_fix() == float("inf")


def test_first_fix_initialises_rather_than_updates():
    filt = Ekf()
    filt.update_marker_fix(_fix(1.0, [2.0, 1.5, 1.2], yaw=0.4))
    assert np.allclose(filt.position, [2.0, 1.5, 1.2])
    assert filt.yaw == pytest.approx(0.4)
    assert filt.is_localised()


def test_localised_goes_false_after_the_timeout():
    filt = Ekf(FusionSettings(fix_timeout=1.0))
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    filt.predict(_telemetry(0.5))
    assert filt.is_localised()
    for t in np.arange(0.6, 3.0, 0.1):
        filt.predict(_telemetry(float(t)))
    assert not filt.is_localised()


def test_localised_goes_false_when_the_covariance_grows_too_far():
    filt = Ekf(FusionSettings(fix_timeout=1e6, max_position_sigma=0.05))
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2], sigma=0.02))
    for t in np.arange(0.1, 6.0, 0.1):
        filt.predict(_telemetry(float(t)))
    assert filt.position_sigma() > 0.05
    assert not filt.is_localised()


# ------------------------------------------------------------------ prediction


def test_a_stationary_level_aircraft_does_not_drift_much():
    """Gravity must be removed exactly, or the estimate accelerates into the floor."""
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    for t in np.arange(0.1, 3.0, 0.1):
        filt.predict(_telemetry(float(t)))
    assert np.linalg.norm(filt.position - np.array([2.0, 1.5, 1.2])) < 0.15


def test_forward_acceleration_moves_the_estimate_forward():
    """Telemetry consistent with the motion, not just the acceleration part of it.

    The Tello's own body-velocity estimate arrives in the same packet and the
    filter uses it, so a packet claiming half a g of forward acceleration and
    zero velocity is not a test of the prediction step, it is a test of what the
    filter does with two measurements that contradict each other.
    """
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [1.0, 1.0, 1.2]))
    accel = np.array([0.5, 0.0, GRAVITY])
    for t in np.arange(0.1, 1.01, 0.1):
        filt.predict(
            _telemetry(
                float(t), accel_body=accel, velocity_body=(0.5 * float(t), 0.0, 0.0),
                height=1.2,
            )
        )
    assert filt.position[0] > 1.05
    assert filt.velocity[0] > 0.2


def test_acceleration_is_rotated_by_the_estimated_yaw():
    """The same body-frame acceleration must move the aircraft in the direction
    it is actually pointing."""
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [1.0, 1.0, 1.2], yaw=np.pi / 2))
    accel = np.array([0.5, 0.0, GRAVITY])
    for t in np.arange(0.1, 1.01, 0.1):
        filt.predict(
            _telemetry(
                float(t), accel_body=accel, yaw=np.pi / 2,
                velocity_body=(0.5 * float(t), 0.0, 0.0), height=1.2,
            )
        )
    assert filt.position[1] > 1.05
    assert abs(filt.position[0] - 1.0) < 0.05


def test_covariance_grows_without_fixes():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    before = filt.position_sigma()
    for t in np.arange(0.1, 2.0, 0.1):
        filt.predict(_telemetry(float(t)))
    assert filt.position_sigma() > before


def test_a_packet_from_the_past_is_ignored():
    filt = Ekf()
    filt.update_marker_fix(_fix(1.0, [2.0, 1.5, 1.2]))
    filt.predict(_telemetry(1.5))
    predictions = filt.diagnostics.predictions
    filt.predict(_telemetry(1.2))
    assert filt.diagnostics.predictions == predictions


def test_an_invalid_packet_is_ignored():
    filt = Ekf()
    filt.update_marker_fix(_fix(1.0, [2.0, 1.5, 1.2]))
    invalid = TelloState(
        timestamp=2.0,
        accel=np.zeros(3),
        gyro_z=0.0,
        attitude=np.zeros(3),
        velocity_body=np.zeros(3),
        height=0.0,
        barometer=float("nan"),
        battery=0.0,
        temperature=float("nan"),
        valid=False,
    )
    filt.predict(invalid)
    assert filt.timestamp == pytest.approx(1.0)


# --------------------------------------------------------------------- updates


def test_a_fix_pulls_the_estimate_towards_it():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    filt.predict(_telemetry(0.5))
    assert filt.update_marker_fix(_fix(0.5, [2.3, 1.5, 1.2]))
    assert 2.0 < filt.position[0] <= 2.3


def test_an_outlier_fix_is_gated_out():
    """A fix ten metres away is not a surprise to be absorbed, it is a wrong answer."""
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    filt.predict(_telemetry(0.5))
    assert not filt.update_marker_fix(_fix(0.5, [12.0, 1.5, 1.2], sigma=0.02))
    assert filt.diagnostics.fixes_rejected == 1
    assert np.linalg.norm(filt.position - np.array([2.0, 1.5, 1.2])) < 0.3


def test_a_delayed_fix_is_applied_at_its_own_timestamp():
    """The out-of-order path, which is the normal case rather than the exception.

    A marker fix is only available after detection and the pose solve have run,
    tens of milliseconds after the frame arrived, by which time more telemetry
    has been processed. Applying it at the current time instead injects however
    far the aircraft moved in between.
    """
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [1.0, 1.0, 1.2]))
    for t in np.arange(0.1, 0.61, 0.1):
        filt.predict(_telemetry(float(t)))
    accepted = filt.update_marker_fix(_fix(0.35, [1.1, 1.0, 1.2]))
    assert accepted
    assert filt.timestamp == pytest.approx(0.6)
    assert 1.0 <= filt.position[0] <= 1.15


def test_a_fix_older_than_the_buffer_is_dropped():
    filt = Ekf(FusionSettings(max_buffer_age=0.3))
    filt.update_marker_fix(_fix(0.0, [1.0, 1.0, 1.2]))
    for t in np.arange(0.1, 2.01, 0.1):
        filt.predict(_telemetry(float(t)))
    assert not filt.update_marker_fix(_fix(0.2, [1.5, 1.0, 1.2]))
    assert filt.diagnostics.fixes_stale >= 1


def test_height_update_corrects_the_vertical_axis():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.5]))
    for t in np.arange(0.1, 2.0, 0.1):
        filt.predict(_telemetry(float(t), height=1.0))
    assert filt.position[2] == pytest.approx(1.0, abs=0.1)


def test_height_outside_the_sensor_range_is_ignored():
    filt = Ekf(FusionSettings(height_valid_range=(0.5, 2.0)))
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    for t in np.arange(0.1, 2.0, 0.1):
        filt.predict(_telemetry(float(t), height=0.0))
    assert filt.position[2] > 1.0


def test_visual_odometry_updates_velocity():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    filt.predict(_telemetry(0.1))
    assert filt.update_visual_odometry(
        VoEstimate(0.1, np.array([0.4, 0.0, 0.0]), n_tracks=80, inlier_ratio=0.9,
                   scale_source="marker")
    )
    assert filt.velocity[0] > 0.05


def test_visual_odometry_with_too_few_tracks_is_refused():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    assert not filt.update_visual_odometry(
        VoEstimate(0.1, np.array([1.0, 0.0, 0.0]), n_tracks=3, inlier_ratio=0.9,
                   scale_source="filter")
    )
    assert filt.diagnostics.vo_rejected == 1


def test_invalid_visual_odometry_is_refused():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    assert not filt.update_visual_odometry(
        VoEstimate(0.1, np.zeros(3), n_tracks=80, inlier_ratio=0.0,
                   scale_source="filter", valid=False)
    )


# ------------------------------------------------------------------ invariants


def test_covariance_stays_symmetric_and_positive_definite():
    """Joseph form and the reason for it.

    The short ``(I - KH)P`` update is algebraically identical and numerically is
    not: over a whole mission at 10 Hz it loses symmetry and eventually positive
    definiteness and a filter with an indefinite covariance produces a Kalman
    gain that amplifies noise instead of rejecting it.
    """
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    rng = np.random.default_rng(3)
    t = 0.0
    for _ in range(600):
        t += 0.1
        filt.predict(_telemetry(t))
        if rng.random() < 0.7:
            filt.update_marker_fix(_fix(t, np.array([2.0, 1.5, 1.2]) + rng.normal(0, 0.02, 3)))
    assert np.allclose(filt.covariance, filt.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(filt.covariance) > 0)


def test_covariance_floor_stops_the_filter_becoming_overconfident():
    """Consecutive fixes are not independent, so averaging cannot beat the
    systematic error. Without a floor the filter reports millimetres."""
    settings = FusionSettings(min_position_sigma=0.03)
    filt = Ekf(settings)
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    _settle(filt, [2.0, 1.5, 1.2], duration=30.0)
    assert filt.position_sigma() >= settings.min_position_sigma * 0.99


def test_divergence_is_detected():
    """Height out of the sensor's range, so nothing constrains the estimate.

    With the time-of-flight sensor in range the vertical axis stays pinned no
    matter how badly the horizontal drifts, which is exactly why the reported
    sigma is the worst axis rather than the average of the three.
    """
    filt = Ekf(FusionSettings(diverged_position_sigma=0.5))
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    for t in np.arange(0.1, 60.0, 0.1):
        filt.predict(_telemetry(float(t), height=0.0))
    # Forty-five seconds of dead reckoning to pass half a metre of uncertainty.
    # Slower than it sounds because the aircraft's own velocity estimate is still
    # arriving and bounds the drift; the accelerometer alone would get there in a
    # fraction of the time.
    assert filt.has_diverged()


def test_nav_state_is_a_faithful_snapshot():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2], yaw=0.3))
    filt.predict(_telemetry(0.5, yaw=0.3))
    state = filt.nav_state()
    assert np.allclose(state.position, filt.position)
    assert state.yaw == pytest.approx(filt.yaw)
    assert state.covariance.shape == (11, 11)
    assert state.position_sigma() == pytest.approx(filt.position_sigma())
    assert state.localised == filt.is_localised()


def test_diagnostics_count_what_happened():
    filt = Ekf()
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    filt.predict(_telemetry(0.5))
    filt.update_marker_fix(_fix(0.5, [2.01, 1.5, 1.2]))
    filt.update_marker_fix(_fix(0.5, [20.0, 1.5, 1.2]))
    counts = filt.diagnostics.as_dict()
    assert counts["fixes_accepted"] >= 2
    assert counts["fixes_rejected"] == 1
    assert counts["predictions"] == 1


def test_accelerometer_bias_is_estimated():
    """A constant bias must end up in the bias state, not in the position.

    Ten-hertz accelerometer data integrated twice turns a 0.05 m/s^2 bias into
    25 cm of position error in ten seconds, which is longer than this aircraft
    routinely goes without seeing a marker.
    """
    filt = Ekf()
    bias = np.array([0.25, -0.15, 0.0])
    filt.update_marker_fix(_fix(0.0, [2.0, 1.5, 1.2]))
    t = 0.0
    for _ in range(400):
        t += 0.1
        filt.predict(_telemetry(t, accel_body=np.array([0.0, 0.0, GRAVITY]) + bias))
        filt.update_marker_fix(_fix(t, [2.0, 1.5, 1.2]))
    assert np.linalg.norm(filt.state[7:10] - bias) < np.linalg.norm(bias)
