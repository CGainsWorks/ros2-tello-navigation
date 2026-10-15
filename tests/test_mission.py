"""Missions: parsing, validation and one complete flight end to end.

The flight test is the slow one and the one worth having. It runs the real
detector over real rendered frames, the real filter, the real controller and the
real failsafe against the real simulator and asserts that the aircraft gets
where it was sent and lands itself. Everything else in the suite tests a piece;
this tests that the pieces make a system.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.control.mission import (
    Mission,
    MissionError,
    MissionRunner,
    MissionSettings,
    mission_visibility,
    resolve_yaw,
    summarise,
)
from tello_core.control.pid import ControlGains
from tello_core.eval.metrics import evaluate
from tello_core.sim.link import LinkFaults, SimLink, SimSettings
from tello_core.types import FlightState, WaypointEvent

MISSION = {
    "name": "unit",
    "takeoff_height_m": 1.1,
    "waypoints": [
        {"name": "a", "position": [1.0, 0.9, 1.2]},
        {"name": "b", "position": [2.6, 0.9, 1.2], "yaw_deg": -90.0},
    ],
}


# ------------------------------------------------------------------ parsing


def test_parses_a_mission():
    mission = Mission.from_dict(MISSION)
    assert mission.name == "unit"
    assert len(mission.waypoints) == 2
    assert mission.waypoints[0].yaw is None
    assert mission.waypoints[1].yaw == pytest.approx(np.deg2rad(-90.0))


def test_defaults_apply_to_every_waypoint():
    data = dict(MISSION, default_tolerance_m=0.3, default_dwell_s=2.0, default_timeout_s=45.0)
    mission = Mission.from_dict(data)
    assert all(w.tolerance == 0.3 for w in mission.waypoints)
    assert all(w.dwell == 2.0 for w in mission.waypoints)
    assert all(w.timeout == 45.0 for w in mission.waypoints)


def test_per_waypoint_overrides_win():
    data = dict(MISSION)
    data["waypoints"] = [dict(MISSION["waypoints"][0], tolerance_m=0.05, dwell_s=0.1)]
    waypoint = Mission.from_dict(data).waypoints[0]
    assert waypoint.tolerance == 0.05
    assert waypoint.dwell == 0.1


def test_a_mission_with_no_waypoints_is_rejected():
    with pytest.raises(MissionError, match="no waypoints"):
        Mission.from_dict({"waypoints": []})


def test_a_bad_position_is_rejected():
    with pytest.raises(MissionError, match="position"):
        Mission.from_dict({"waypoints": [{"name": "a", "position": [1.0, 2.0]}]})


def test_a_missing_file_is_rejected(tmp_path):
    with pytest.raises(MissionError, match="not found"):
        Mission.from_yaml(tmp_path / "nope.yaml")


def test_the_shipped_mission_loads_and_validates(repo_root, config, marker_map):
    mission = Mission.from_yaml(repo_root / "missions" / "perimeter.yaml")
    assert mission.validate(config, marker_map) == []


def test_a_waypoint_outside_the_geofence_fails_validation(config, marker_map):
    data = dict(MISSION)
    data["waypoints"] = [{"name": "far", "position": [9.0, 9.0, 1.2]}]
    problems = Mission.from_dict(data).validate(config, marker_map)
    assert problems and "geofence" in problems[0]


def test_a_non_positive_tolerance_fails_validation(config, marker_map):
    data = dict(MISSION)
    data["waypoints"] = [{"name": "a", "position": [1.0, 0.9, 1.2], "tolerance_m": 0.0}]
    problems = Mission.from_dict(data).validate(config, marker_map)
    assert any("tolerance" in problem for problem in problems)


# ------------------------------------------------------------------ planning


def test_resolve_yaw_honours_an_explicit_heading(marker_map, config, camera):
    mission = Mission.from_dict(MISSION)
    assert resolve_yaw(mission.waypoints[1], marker_map, config, camera) == pytest.approx(
        np.deg2rad(-90.0)
    )


def test_resolve_yaw_picks_a_heading_that_sees_something(marker_map, config, camera):
    mission = Mission.from_dict(MISSION)
    rows = mission_visibility(mission, marker_map, config, camera)
    assert rows[0]["auto_yaw"]
    assert rows[0]["n_visible"] >= 1


def test_mission_visibility_reports_every_waypoint(marker_map, config, camera):
    rows = mission_visibility(Mission.from_dict(MISSION), marker_map, config, camera)
    assert [row["name"] for row in rows] == ["a", "b"]
    for row in rows:
        assert len(row["position"]) == 3
        assert -180.0 <= row["yaw_deg"] <= 180.0


def test_summarise_counts_event_kinds():
    events = [
        WaypointEvent(0.0, 0, "a", "arrived"),
        WaypointEvent(1.0, 1, "b", "arrived"),
        WaypointEvent(2.0, 2, "c", "timeout"),
    ]
    assert summarise(events) == {"arrived": 2, "timeout": 1}


# ------------------------------------------------------------------- flying


def _runner(room, config, camera, marker_map, mission, repo_root, **link_kwargs):
    link = SimLink(
        room,
        config,
        camera,
        SimSettings(seed=1, **link_kwargs),
        start_position=np.array([1.0, 0.9, 0.03]),
        start_yaw=0.0,
    )
    settings = MissionSettings(
        control_rate=20.0,
        max_flight_time=60.0,
        gains=ControlGains.from_yaml(
            repo_root / "src" / "tello_bringup" / "config" / "control.yaml"
        ),
    )
    return link, MissionRunner(link, mission, marker_map, config, camera, settings)


@pytest.fixture(scope="module")
def short_flight(request):
    """One complete flight, shared by the assertions that read it.

    Module-scoped because it takes tens of seconds: it renders and detects
    several hundred frames. Splitting it into one flight per assertion would
    multiply that by the number of things worth asserting.
    """
    room = request.getfixturevalue("room")
    config = request.getfixturevalue("config")
    camera = request.getfixturevalue("camera")
    marker_map = request.getfixturevalue("marker_map")
    repo_root = request.getfixturevalue("repo_root")
    mission = Mission.from_dict(MISSION)
    link, runner = _runner(room, config, camera, marker_map, mission, repo_root)
    log = runner.run()
    return runner, log


@pytest.mark.slow
def test_the_aircraft_takes_off_and_lands_itself(short_flight):
    runner, log = short_flight
    assert runner.failsafe.state is FlightState.LANDED
    assert log.rows[-1].state == "LANDED"


@pytest.mark.slow
def test_the_mission_is_completed(short_flight):
    runner, log = short_flight
    arrived = [e for e in log.waypoint_events if e["kind"] == "arrived"]
    assert len(arrived) == len(runner.mission.waypoints)


@pytest.mark.slow
def test_the_estimate_stays_close_to_the_truth(short_flight):
    _, log = short_flight
    metrics = evaluate(log)
    assert metrics["ground_truth"] == "simulator"
    error = metrics["trajectory_error"]
    assert error["samples"] > 100
    assert error["position_rmse_m"] < 0.30
    assert error["yaw_rmse_deg"] < 8.0


@pytest.mark.slow
def test_the_filter_is_not_wildly_overconfident(short_flight):
    """Error over reported sigma, averaged across the flight.

    Under-reporting is the dangerous direction: the failsafe reads the position
    sigma to decide whether the estimate can be acted on and if the filter lies
    to it the failsafe will not trip when it should.
    """
    _, log = short_flight
    assert evaluate(log)["trajectory_error"]["consistency_ratio"] < 5.0


@pytest.mark.slow
def test_localisation_is_available_for_most_of_the_flight(short_flight):
    _, log = short_flight
    availability = evaluate(log)["localisation"]
    assert availability["availability"] > 0.7


@pytest.mark.slow
def test_the_log_records_what_happened(short_flight):
    runner, log = short_flight
    assert log.metadata["marker_map"]
    assert log.metadata["survey_method"] == "designed"
    assert log.metadata["detector_backend"]
    assert log.metadata["final_state"] == "LANDED"
    assert log.metadata["fusion"]["fixes_accepted"] > 20
    assert len(log.rows) > 100
    del runner


@pytest.mark.slow
def test_ground_truth_never_reaches_the_flight_software(short_flight):
    """The estimator must not be able to see the answer.

    Enforced by construction, the truth columns are written by the recorder and
    the flight code never reads them but a test is cheap and the failure mode is
    a system that looks superb and does not work.
    """
    runner, log = short_flight
    del log
    source = runner.__class__.__module__
    assert "sim" not in source
    assert not hasattr(runner.filter, "truth")


@pytest.mark.slow
def test_a_link_loss_produces_a_landing_and_a_cause(
    room, config, camera, marker_map, repo_root
):
    """The headline failsafe test.

    The link drops mid-flight and never comes back. The aircraft must land, the
    cause must be recorded and the controller must stop transmitting so the
    aircraft's own fifteen-second timer is free to act.
    """
    mission = Mission.from_dict(MISSION)
    link, runner = _runner(
        room, config, camera, marker_map, mission, repo_root,
        # Three seconds, so the one second link timeout expires at about four,
        # which is comfortably before this two waypoint mission finishes. An
        # earlier version dropped the link at eight seconds and silently stopped
        # testing anything the day the controller got fast enough to land first.
        faults=LinkFaults(outages=((3.0, 60.0),)),
    )
    log = runner.run()

    causes = {event["cause"] for event in log.failsafe_events}
    assert "LINK_LOSS" in causes
    assert runner.failsafe.state in (FlightState.LAND, FlightState.LANDED)

    # Measured from the log rather than from a hardcoded second, so retuning the
    # controller cannot turn this into a test of nothing.
    dropped = min(
        event["time_s"] for event in log.failsafe_events if event["cause"] == "LINK_LOSS"
    )
    after = [row for row in log.rows if row.time_s > dropped]
    assert after
    assert all(row.rc_sent == 0 for row in after), (
        "the controller kept transmitting after the link dropped, which on a "
        "degraded link would keep resetting the aircraft's own dead-man timer"
    )
    assert not link.dynamics.state.flying


@pytest.mark.slow
def test_a_flat_battery_ends_the_mission_early(
    room, config, camera, marker_map, repo_root
):
    mission = Mission.from_dict(MISSION)
    link, runner = _runner(room, config, camera, marker_map, mission, repo_root)
    link.dynamics.state.battery = 0.13
    log = runner.run()
    causes = {event["cause"] for event in log.failsafe_events}
    assert causes & {"LOW_BATTERY", "CRITICAL_BATTERY"}
    assert runner.failsafe.state in (
        FlightState.LAND,
        FlightState.EMERGENCY_LAND,
        FlightState.LANDED,
    )


@pytest.mark.slow
def test_a_battery_below_the_return_threshold_refuses_the_flight(
    room, config, camera, marker_map, repo_root
):
    """Refused on the ground. The log has to say so.

    This is a different path from the no-fix refusal above: the aircraft
    localises itself, reads its own battery and only then declines, so it never
    goes near ``_abort_preflight``. Both the refusal and the record of it have
    to be asserted. Without the record the log is indistinguishable from a
    mission that flew and landed, since both end in LANDED having reached no
    waypoints. The earlier version of this code sent a grounded aircraft to
    RETURN and left it sitting with its motors off until the clock ran out.
    """
    mission = Mission.from_dict(MISSION)
    link, runner = _runner(room, config, camera, marker_map, mission, repo_root)
    link.dynamics.state.battery = 0.20
    log = runner.run()
    assert log.metadata.get("aborted"), "a refused flight must be marked as aborted"
    assert log.metadata["waypoints_reached"] == 0
    assert not link.dynamics.state.flying
    assert not runner._ever_airborne


@pytest.mark.slow
def test_it_refuses_to_take_off_without_a_fix(room, config, camera, marker_map, repo_root):
    """An aircraft that leaves the ground not knowing where it is has to find out
    while moving, using a camera that is now being shaken."""
    mission = Mission.from_dict(MISSION)
    link = SimLink(
        room,
        config,
        camera,
        SimSettings(seed=1, faults=LinkFaults(frame_loss=1.0)),
        start_position=np.array([1.0, 0.9, 0.03]),
        start_yaw=0.0,
    )
    settings = MissionSettings(control_rate=20.0, preflight_timeout=3.0, max_flight_time=20.0)
    runner = MissionRunner(link, mission, marker_map, config, camera, settings)
    log = runner.run()
    assert log.metadata.get("aborted")
    assert not link.dynamics.state.flying


def test_an_unflyable_mission_is_refused_before_connecting(
    room, config, camera, marker_map, repo_root
):
    data = dict(MISSION)
    data["waypoints"] = [{"name": "far", "position": [9.0, 9.0, 1.2]}]
    mission = Mission.from_dict(data)
    link, runner = _runner(room, config, camera, marker_map, mission, repo_root)
    with pytest.raises(MissionError, match="not flyable"):
        runner.run()
