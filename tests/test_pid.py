"""The PID controller and the three behaviours it exists to get right."""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.control.pid import ControlGains, Pid, PidGains, VectorPid


def test_proportional_response_has_the_right_sign_and_size():
    pid = Pid(PidGains(kp=2.0))
    assert pid.update(setpoint=1.0, measurement=0.0, dt=0.05) == pytest.approx(2.0)
    assert pid.update(setpoint=-1.0, measurement=0.0, dt=0.05) == pytest.approx(-2.0)


def test_output_is_clamped():
    pid = Pid(PidGains(kp=10.0, output_limit=0.5))
    assert pid.update(1.0, 0.0, 0.05) == pytest.approx(0.5)
    assert pid.state.saturated


def test_no_derivative_kick_on_a_setpoint_step():
    """A step in the setpoint must not produce a derivative impulse.

    Differentiating the error would put a full-scale stick deflection on the
    output for one cycle every time a waypoint changes. Differentiating the
    measurement gives the same damping with none of that.
    """
    pid = Pid(PidGains(kp=1.0, kd=5.0))
    pid.update(setpoint=0.0, measurement=0.0, dt=0.05)
    stepped = pid.update(setpoint=1.0, measurement=0.0, dt=0.05)
    assert stepped == pytest.approx(1.0, abs=1e-9)


def test_derivative_damps_a_moving_measurement():
    pid = Pid(PidGains(kp=0.0, kd=1.0, derivative_tau=0.0))
    pid.update(setpoint=0.0, measurement=0.0, dt=0.1)
    moving = pid.update(setpoint=0.0, measurement=0.1, dt=0.1)
    assert moving < 0.0


def test_derivative_is_filtered():
    """The filter must attenuate a one-sample jump, not pass it straight through."""
    unfiltered = Pid(PidGains(kp=0.0, kd=1.0, derivative_tau=0.0))
    filtered = Pid(PidGains(kp=0.0, kd=1.0, derivative_tau=0.2))
    for pid in (unfiltered, filtered):
        pid.update(0.0, 0.0, 0.05)
    assert abs(filtered.update(0.0, 0.1, 0.05)) < abs(unfiltered.update(0.0, 0.1, 0.05))


def test_integral_removes_a_steady_error():
    pid = Pid(PidGains(kp=0.0, ki=1.0))
    for _ in range(20):
        output = pid.update(setpoint=1.0, measurement=0.0, dt=0.05)
    assert output == pytest.approx(1.0, rel=0.1)


def test_integral_is_clamped():
    pid = Pid(PidGains(kp=0.0, ki=10.0, integral_limit=0.2))
    for _ in range(100):
        output = pid.update(1.0, 0.0, 0.05)
    assert abs(output) <= 0.2 + 1e-9


def test_back_calculation_stops_windup_while_saturated():
    """The classic failure: a saturated actuator and an integrator that keeps growing.

    Drive a large error until the output saturates, then reverse the error. A
    controller with back-calculation responds within a cycle or two. One without
    spends seconds unwinding an integral that never did anything.
    """
    gains = dict(kp=1.0, ki=4.0, output_limit=1.0, integral_limit=100.0)
    with_awu = Pid(PidGains(back_calculation=1.0, **gains))
    without = Pid(PidGains(back_calculation=0.0, **gains))
    for _ in range(60):
        with_awu.update(5.0, 0.0, 0.05)
        without.update(5.0, 0.0, 0.05)
    assert abs(with_awu.state.integral) < abs(without.state.integral)

    # Now reverse the error and count cycles until the output changes sign.
    def cycles_to_reverse(pid):
        for cycle in range(200):
            if pid.update(-1.0, 0.0, 0.05) < 0:
                return cycle
        return 200

    assert cycles_to_reverse(with_awu) < cycles_to_reverse(without)


def test_reset_clears_everything():
    pid = Pid(PidGains(kp=1.0, ki=1.0, kd=1.0))
    for _ in range(10):
        pid.update(1.0, 0.0, 0.05)
    pid.reset()
    assert pid.state.integral == 0.0
    assert pid.state.last_measurement is None
    # After a reset the first call must behave like the very first call.
    assert pid.update(1.0, 0.0, 0.05) == pytest.approx(1.0)


def test_zero_dt_returns_the_previous_output():
    pid = Pid(PidGains(kp=1.0))
    first = pid.update(1.0, 0.0, 0.05)
    assert pid.update(1.0, 0.0, 0.0) == first


def test_vector_pid_axes_are_independent():
    vector = VectorPid.from_gains(PidGains(kp=1.0), PidGains(kp=2.0))
    output = vector.update(np.array([1.0, 1.0, 1.0]), np.zeros(3), 0.05)
    assert output == pytest.approx([1.0, 1.0, 2.0])


def test_vector_pid_reset():
    vector = VectorPid.from_gains(PidGains(kp=1.0, ki=1.0), PidGains(kp=1.0, ki=1.0))
    for _ in range(10):
        vector.update(np.ones(3), np.zeros(3), 0.05)
    assert np.any(vector.integrals() != 0)
    vector.reset()
    assert np.all(vector.integrals() == 0)


def test_gains_load_from_the_shipped_config(repo_root):
    gains = ControlGains.from_yaml(
        repo_root / "src" / "tello_bringup" / "config" / "control.yaml"
    )
    assert gains.horizontal.kp > 0
    assert gains.vertical.kp > 0
    assert gains.yaw.ki == 0.0


def test_shipped_yaw_limit_keeps_motion_blur_under_control(repo_root, camera):
    """The yaw limit is a perception constraint, not a control one.

    Blur is ``fx * omega * exposure`` pixels. Beyond about eight pixels the
    decoder stops reading a marker at any useful range, which cost this project
    its first autonomous flight. The measured curve is in
    ``results/detection_yaw_rate.csv``; this test pins the conclusion so that
    raising the limit for a snappier turn cannot quietly reintroduce it.
    """
    gains = ControlGains.from_yaml(
        repo_root / "src" / "tello_bringup" / "config" / "control.yaml"
    )
    blur_px = camera.fx * gains.yaw.output_limit / 60.0
    assert blur_px < 8.0


def test_gains_from_dict_falls_back_to_defaults():
    gains = ControlGains.from_dict({"horizontal": {"kp": 9.0}})
    assert gains.horizontal.kp == 9.0
    assert gains.horizontal.ki == ControlGains().horizontal.ki
    assert gains.vertical.kp == ControlGains().vertical.kp
