"""The failsafe machine: every trigger, the priority order and the invariants.

The most important test file in the repository. Everything else decides where the
aircraft goes; this decides whether it is allowed to go anywhere and what happens
when it is not. Each trigger gets a test, the priority order gets a test and the
two structural invariants, monotonicity and the silent link-loss descent, get
tests of their own, because they are the properties that stop a flickering sensor
talking the machine out of landing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.control.failsafe import (
    DescentProfile,
    FailsafeInputs,
    FailsafeMachine,
    FailsafeSettings,
)
from tello_core.link.base import LinkStatus
from tello_core.types import FailsafeCause, FlightState, NavState


def _link(state_age: float = 0.05, frame_age: float = 0.05, connected: bool = True) -> LinkStatus:
    return LinkStatus(
        connected=connected,
        last_state_age=state_age,
        last_frame_age=frame_age,
        packets_received=100,
    )


def _nav(
    position=(2.0, 1.5, 1.2),
    sigma: float = 0.03,
    localised: bool = True,
    time_since_fix: float = 0.1,
) -> NavState:
    covariance = np.eye(11) * 1e-4
    covariance[:3, :3] = np.eye(3) * sigma**2
    return NavState(
        timestamp=0.0,
        position=np.asarray(position, dtype=float),
        velocity=np.zeros(3),
        yaw=0.0,
        accel_bias=np.zeros(3),
        gyro_bias=0.0,
        covariance=covariance,
        localised=localised,
        time_since_fix=time_since_fix,
    )


def _inputs(now: float = 1.0, **overrides) -> FailsafeInputs:
    defaults = dict(
        now=now,
        link=_link(),
        estimate=_nav(),
        battery=0.9,
        temperature=65.0,
        height=1.2,
        flight_time=now,
        airborne=True,
    )
    defaults.update(overrides)
    return FailsafeInputs(**defaults)


def _airborne_machine(config) -> FailsafeMachine:
    """A machine already navigating, which is where every trigger test starts."""
    machine = FailsafeMachine(config, FailsafeSettings())
    machine.update(_inputs(now=0.0, takeoff_requested=True, airborne=False, height=0.0))
    machine.update(_inputs(now=0.5, height=0.8))
    assert machine.state is FlightState.NAVIGATE
    return machine


# ------------------------------------------------------------------ lifecycle


def test_starts_idle_and_stays_there(config):
    machine = FailsafeMachine(config, FailsafeSettings())
    decision = machine.update(_inputs(now=0.0, airborne=False, height=0.0))
    assert decision.state is FlightState.IDLE
    assert not decision.send_rc


def test_takeoff_requires_a_request(config):
    machine = FailsafeMachine(config, FailsafeSettings())
    assert machine.update(_inputs(airborne=False, height=0.0)).state is FlightState.IDLE
    assert (
        machine.update(_inputs(takeoff_requested=True, airborne=False, height=0.0)).state
        is FlightState.TAKEOFF
    )


def test_takeoff_becomes_navigate_once_airborne(config):
    assert _airborne_machine(config).state is FlightState.NAVIGATE


@pytest.mark.parametrize(
    "override,cause",
    [
        ({"battery": 0.20}, FailsafeCause.LOW_BATTERY),
        ({"battery": 0.10}, FailsafeCause.CRITICAL_BATTERY),
        ({"estimator_diverged": True}, FailsafeCause.ESTIMATOR_DIVERGED),
        ({"link": _link(state_age=5.0)}, FailsafeCause.LINK_LOSS),
        ({"temperature": 92.0}, FailsafeCause.OVERHEAT),
        ({"operator_abort": True}, FailsafeCause.OPERATOR_ABORT),
    ],
)
def test_a_condition_that_would_land_the_aircraft_refuses_the_take_off(
    config, override, cause
):
    """Anything worth landing for is worth not taking off for.

    And the aircraft is already where a landing would put it, so the refusal is
    terminal rather than a descent. Getting this wrong is not a cosmetic
    difference: a battery below the return threshold used to send the machine to
    RETURN while the aircraft sat on the floor with its motors off and RETURN
    does not command a take-off, so nothing moved and nothing landed until the
    wall-clock limit expired ten minutes later.
    """
    machine = FailsafeMachine(config, FailsafeSettings())
    inputs = _inputs(now=0.0, takeoff_requested=True, airborne=False, height=0.0, **override)
    decision = machine.update(inputs)
    assert decision.state is FlightState.LANDED
    assert decision.cause is cause
    assert not decision.send_rc
    assert "refused to take off" in machine.events[-1].detail


def test_a_healthy_aircraft_is_not_refused(config):
    machine = FailsafeMachine(config, FailsafeSettings())
    decision = machine.update(
        _inputs(now=0.0, takeoff_requested=True, airborne=False, height=0.0)
    )
    assert decision.state is FlightState.TAKEOFF


def test_losing_localisation_on_the_ground_is_not_a_refusal(config):
    """It is a take-off that has not happened yet.

    How long to wait for a first fix is the mission runner's decision, not the
    failsafe's; see MissionSettings.preflight_timeout.
    """
    machine = FailsafeMachine(config, FailsafeSettings())
    decision = machine.update(
        _inputs(
            now=0.0, takeoff_requested=True, airborne=False, height=0.0,
            estimate=_nav(localised=False, time_since_fix=9.0),
        )
    )
    assert decision.state is FlightState.TAKEOFF


def test_mission_complete_ends_in_a_landing(config):
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, mission_complete=True))
    assert machine.state is FlightState.LAND
    assert machine.cause is FailsafeCause.NONE


def test_landing_completes_when_the_aircraft_is_down(config):
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, mission_complete=True))
    machine.update(_inputs(now=3.0, height=0.02, airborne=False))
    decision = machine.update(_inputs(now=4.5, height=0.02, airborne=False))
    assert decision.state is FlightState.LANDED
    assert decision.terminal
    assert not decision.send_rc


def test_a_bounce_does_not_count_as_a_landing(config):
    """The dwell requirement exists so a touch and a rebound is not a landing."""
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, mission_complete=True))
    machine.update(_inputs(now=3.0, height=0.02, airborne=False))
    machine.update(_inputs(now=3.2, height=0.4, airborne=True))
    assert machine.state is FlightState.LAND


# -------------------------------------------------------------------- triggers


def test_link_loss_lands(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, link=_link(state_age=2.0)))
    assert decision.state is FlightState.LAND
    assert decision.cause is FailsafeCause.LINK_LOSS


def test_link_loss_stops_sending_stick_commands(config):
    """The counter-intuitive one and the reason the field exists.

    A Tello lands itself fifteen seconds after the last command it hears. On a
    half-broken link, an rc command that gets through resets that timer, so a
    controller that keeps transmitting can keep an aircraft it has no control
    over airborne indefinitely. Falling silent hands the aircraft to its own
    failsafe, which is the only thing still able to act.
    """
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, link=_link(state_age=2.0)))
    assert not decision.send_rc
    assert decision.command_land


def test_an_ordinary_landing_does_keep_sending_stick_commands(config):
    """Silence is specific to link loss, not to landing.

    A controlled descent needs rc to hold the horizontal channels at zero.
    """
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, mission_complete=True))
    assert decision.send_rc
    assert decision.command_land


def test_critical_battery_lands_immediately(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, battery=0.10))
    assert decision.state is FlightState.EMERGENCY_LAND
    assert decision.cause is FailsafeCause.CRITICAL_BATTERY


def test_low_battery_returns_home(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, battery=0.22))
    assert decision.state is FlightState.RETURN
    assert decision.cause is FailsafeCause.LOW_BATTERY


def test_low_battery_lands_where_it_is_when_there_is_no_reserve(config):
    """Not enough left to fly home means do not start the journey."""
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, battery=0.045))
    assert decision.state in (FlightState.LAND, FlightState.EMERGENCY_LAND)


def test_return_ends_in_a_landing(config):
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, battery=0.22))
    decision = machine.update(_inputs(now=6.0, battery=0.22, home_reached=True))
    assert decision.state is FlightState.LAND
    assert decision.cause is FailsafeCause.LOW_BATTERY


def test_estimator_divergence_lands_immediately(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, estimator_diverged=True))
    assert decision.state is FlightState.EMERGENCY_LAND
    assert decision.cause is FailsafeCause.ESTIMATOR_DIVERGED


def test_geofence_breach_returns(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, estimate=_nav(position=(3.9, 1.5, 1.2))))
    assert decision.state is FlightState.RETURN
    assert decision.cause is FailsafeCause.GEOFENCE_BREACH


def test_large_geofence_breach_lands(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, estimate=_nav(position=(4.5, 1.5, 1.2))))
    assert decision.state is FlightState.LAND
    assert decision.cause is FailsafeCause.GEOFENCE_BREACH


def test_geofence_floor_does_not_trip_during_take_off(config):
    """The fence's floor is a minimum height, which a take-off starts below.

    Checking it during a climb trips the geofence half a second into every
    flight, which is exactly what the first version did.
    """
    machine = FailsafeMachine(config, FailsafeSettings())
    machine.update(_inputs(now=0.0, takeoff_requested=True, airborne=False, height=0.0))
    decision = machine.update(
        _inputs(now=0.6, height=0.2, estimate=_nav(position=(2.0, 1.5, 0.2)))
    )
    assert decision.state is FlightState.TAKEOFF


def test_geofence_is_ignored_when_not_localised(config):
    """An unlocalised estimate saying "outside" is not evidence of anything."""
    machine = _airborne_machine(config)
    decision = machine.update(
        _inputs(now=2.0, estimate=_nav(position=(9.0, 9.0, 1.2), localised=False))
    )
    assert decision.cause is not FailsafeCause.GEOFENCE_BREACH


def test_overheat_returns(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, temperature=92.0))
    assert decision.state is FlightState.RETURN
    assert decision.cause is FailsafeCause.OVERHEAT


def test_mission_timeout_returns(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=400.0, flight_time=400.0))
    assert decision.state is FlightState.RETURN
    assert decision.cause is FailsafeCause.MISSION_TIMEOUT


def test_operator_abort_lands(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, operator_abort=True))
    assert decision.state is FlightState.LAND
    assert decision.cause is FailsafeCause.OPERATOR_ABORT


# --------------------------------------------------------------- localisation


def test_lost_localisation_holds(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, estimate=_nav(localised=False, time_since_fix=3.0)))
    assert decision.state is FlightState.HOLD
    assert decision.cause is FailsafeCause.LOCALISATION_LOST


def test_hold_recovers_when_localisation_returns(config):
    """The only recoverable state, because it is the only recoverable condition."""
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, estimate=_nav(localised=False, time_since_fix=3.0)))
    decision = machine.update(_inputs(now=3.0))
    assert decision.state is FlightState.NAVIGATE
    assert decision.cause is FailsafeCause.NONE


def test_hold_lands_if_localisation_does_not_come_back(config):
    machine = _airborne_machine(config)
    for now in (2.0, 4.0, 6.0, 9.0):
        decision = machine.update(
            _inputs(now=now, estimate=_nav(localised=False, time_since_fix=now))
        )
    assert decision.state is FlightState.LAND
    assert decision.cause is FailsafeCause.LOCALISATION_LOST


def test_stale_video_counts_as_lost_localisation(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, link=_link(frame_age=5.0)))
    assert decision.state is FlightState.HOLD


def test_not_localised_on_the_ground_is_not_a_failsafe(config):
    """Before take-off, no fix means the flight does not start, not that it aborts."""
    machine = FailsafeMachine(config, FailsafeSettings())
    machine.update(_inputs(now=0.0, takeoff_requested=True, airborne=False, height=0.0))
    decision = machine.update(
        _inputs(now=0.5, airborne=False, height=0.0, estimate=_nav(localised=False))
    )
    assert decision.state is FlightState.TAKEOFF
    assert decision.cause is FailsafeCause.NONE


# ------------------------------------------------------------------ invariants


def test_priority_battery_beats_localisation(config):
    machine = _airborne_machine(config)
    decision = machine.update(
        _inputs(now=2.0, battery=0.10, estimate=_nav(localised=False, time_since_fix=5.0))
    )
    assert decision.cause is FailsafeCause.CRITICAL_BATTERY


def test_priority_divergence_beats_link_loss(config):
    machine = _airborne_machine(config)
    decision = machine.update(
        _inputs(now=2.0, estimator_diverged=True, link=_link(state_age=5.0))
    )
    assert decision.cause is FailsafeCause.ESTIMATOR_DIVERGED


def test_priority_link_loss_beats_low_battery(config):
    machine = _airborne_machine(config)
    decision = machine.update(_inputs(now=2.0, link=_link(state_age=5.0), battery=0.20))
    assert decision.cause is FailsafeCause.LINK_LOSS


def test_severity_is_monotone(config):
    """Once landing, always landing.

    A machine that can be talked out of a landing decision by a sensor that
    flickers is not a failsafe. The one exception is HOLD, which exists for the
    condition that clears on its own.
    """
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, battery=0.10))
    assert machine.state is FlightState.EMERGENCY_LAND
    machine.update(_inputs(now=3.0, battery=0.95))
    assert machine.state is FlightState.EMERGENCY_LAND


def test_landed_is_terminal(config):
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, mission_complete=True))
    machine.update(_inputs(now=3.0, height=0.02, airborne=False))
    machine.update(_inputs(now=4.5, height=0.02, airborne=False))
    assert machine.state is FlightState.LANDED
    decision = machine.update(_inputs(now=5.0, height=1.5, airborne=True))
    assert decision.state is FlightState.LANDED


def test_every_transition_is_recorded_with_a_cause(config):
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, battery=0.10))
    rows = machine.event_rows()
    assert rows
    for row in rows:
        assert set(row) == {"time_s", "from_state", "to_state", "cause", "detail"}
        assert row["to_state"]
        assert row["cause"]


def test_events_carry_a_readable_detail(config):
    machine = _airborne_machine(config)
    machine.update(_inputs(now=2.0, battery=0.10))
    landing = [r for r in machine.event_rows() if r["cause"] == "CRITICAL_BATTERY"]
    assert landing and "per cent" in landing[0]["detail"]


# --------------------------------------------------------------------- descent


def test_emergency_descends_faster_than_a_controlled_landing():
    profile = DescentProfile(FailsafeSettings())
    assert profile.rate(FlightState.EMERGENCY_LAND) < profile.rate(FlightState.LAND) < 0


@pytest.mark.parametrize(
    "state", [FlightState.LAND, FlightState.EMERGENCY_LAND]
)
def test_descent_is_downward(state):
    assert DescentProfile(FailsafeSettings()).rate(state) < 0
