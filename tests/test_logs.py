"""Flight logging: round trips, rounding and files that exist even when empty."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from tello_core.eval.metrics import (
    absolute_trajectory_error,
    evaluate,
    fix_statistics,
    localisation_availability,
    static_survey_error,
    summarise_errors,
    waypoint_accuracy,
)
from tello_core.logs import FlightLog, FlightRow, provenance_path


def _row(time_s: float, **overrides) -> FlightRow:
    defaults = dict(
        time_s=time_s,
        state="NAVIGATE",
        cause="NONE",
        est_x=1.0,
        est_y=2.0,
        est_z=1.2,
        est_yaw_deg=30.0,
        est_vx=0.1,
        est_vy=0.0,
        est_vz=0.0,
        est_sigma_m=0.04,
        localised=1,
        time_since_fix_s=0.1,
    )
    defaults.update(overrides)
    return FlightRow(**defaults)


def _log(n: int = 20) -> FlightLog:
    log = FlightLog()
    for index in range(n):
        log.append(
            _row(
                index * 0.05,
                truth_x=1.0 + 0.01 * index,
                truth_y=2.0,
                truth_z=1.2,
                truth_yaw_deg=30.0,
                truth_error_m=0.01 * index,
                n_markers=2 if index % 2 else 0,
                fix_rmse_px=0.6 if index % 2 else float("nan"),
                fix_range_m=1.4 if index % 2 else float("nan"),
                fix_accepted=1 if index % 2 else 0,
            )
        )
    log.waypoint_events = [
        {"time_s": 0.5, "index": 0, "name": "a", "kind": "arrived", "position_error_m": 0.08},
        {"time_s": 0.9, "index": 1, "name": "b", "kind": "timeout", "position_error_m": 0.4},
    ]
    log.failsafe_events = [
        {"time_s": 0.1, "from_state": "IDLE", "to_state": "TAKEOFF", "cause": "NONE",
         "detail": "commanded"}
    ]
    log.metadata = {"mission": "test", "survey_method": "designed"}
    return log


def test_write_creates_every_file(tmp_path):
    written = _log().write(tmp_path, stem="flight")
    assert set(written) == {"trajectory", "failsafe", "waypoints", "metadata"}
    for path in written.values():
        assert path.is_file()


def test_empty_event_files_are_still_written(tmp_path):
    """An empty failsafe log says "nothing tripped"; a missing one says nothing."""
    log = FlightLog()
    log.append(_row(0.0))
    written = log.write(tmp_path, stem="flight")
    with open(written["failsafe"], encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows and rows[0][0] == "time_s"
    assert len(rows) == 1


def test_round_trip(tmp_path):
    original = _log()
    written = original.write(tmp_path, stem="flight")
    reloaded = FlightLog.read(written["trajectory"])
    assert len(reloaded.rows) == len(original.rows)
    assert reloaded.rows[3].state == "NAVIGATE"
    assert reloaded.rows[3].est_x == pytest.approx(1.0)
    assert reloaded.metadata["mission"] == "test"
    assert len(reloaded.failsafe_events) == 1
    assert len(reloaded.waypoint_events) == 2


def test_floats_are_rounded_on_the_way_out(tmp_path):
    """Full double precision in a CSV churns on every platform whose libm rounds
    the last bit differently, which would make the reproducibility check in CI a
    coin toss."""
    log = FlightLog()
    log.append(_row(0.0, est_x=1.0 / 3.0))
    written = log.write(tmp_path, stem="flight")
    with open(written["trajectory"], encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["est_x"] == "0.3333"


def test_non_finite_values_are_written_as_empty(tmp_path):
    log = FlightLog()
    log.append(_row(0.0, fix_rmse_px=float("nan")))
    written = log.write(tmp_path, stem="flight")
    with open(written["trajectory"], encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["fix_rmse_px"] == ""


def test_to_arrays_gives_numeric_columns():
    columns = _log().to_arrays()
    assert columns["est_x"].dtype == float
    assert columns["state"].dtype == object
    assert len(columns["time_s"]) == 20


# --------------------------------------------------------------------- metrics


def test_summarise_errors_reports_all_five():
    stats = summarise_errors([0.1, 0.2, 0.3])
    assert set(stats) == {"mean", "median", "rmse", "p95", "max"}
    assert stats["max"] == pytest.approx(0.3)
    assert stats["rmse"] >= stats["mean"]


def test_summarise_errors_handles_an_empty_series():
    stats = summarise_errors([])
    assert all(np.isnan(v) for v in stats.values())


def test_trajectory_error_needs_ground_truth():
    log = FlightLog()
    log.append(_row(0.0))
    assert absolute_trajectory_error(log).samples == 0


def test_trajectory_error_is_computed_when_truth_is_present():
    error = absolute_trajectory_error(_log())
    assert error.samples > 0
    assert error.position["max"] > 0
    assert error.axis_rmse.shape == (3,)


def test_localisation_availability():
    availability = localisation_availability(_log())
    assert availability["availability"] == pytest.approx(1.0)
    assert availability["longest_gap_s"] == 0.0


def test_localisation_availability_finds_the_longest_gap():
    log = FlightLog()
    for index in range(20):
        log.append(_row(index * 0.1, localised=0 if 5 <= index < 12 else 1))
    availability = localisation_availability(log)
    assert availability["availability"] == pytest.approx(13 / 20)
    assert availability["longest_gap_s"] == pytest.approx(0.7, abs=0.05)


def test_fix_statistics():
    stats = fix_statistics(_log())
    assert stats["fixes"] == 10
    assert stats["acceptance"] == pytest.approx(1.0)
    assert stats["mean_markers_per_fix"] == pytest.approx(2.0)


def test_waypoint_accuracy_counts_both_kinds():
    stats = waypoint_accuracy(_log())
    assert stats["waypoints_reached"] == 1
    assert stats["waypoints_timed_out"] == 1
    assert stats["arrival_error_max"] == pytest.approx(0.08)


def test_evaluate_labels_where_the_truth_came_from():
    """A results file from a real flight must not look like it contains an ATE."""
    with_truth = evaluate(_log())
    assert with_truth["ground_truth"] == "simulator"
    assert "trajectory_error" in with_truth

    log = FlightLog()
    log.append(_row(0.0))
    without_truth = evaluate(log)
    assert without_truth["ground_truth"] == "none"
    assert "trajectory_error" not in without_truth


def test_static_survey_error():
    """The hardware-honest metric: estimate minus surveyed point."""
    estimates = [np.array([1.02, 2.01, 1.19]), np.array([2.98, 1.0, 1.21])]
    surveyed = [np.array([1.0, 2.0, 1.2]), np.array([3.0, 1.0, 1.2])]
    result = static_survey_error(estimates, surveyed)
    assert result["samples"] == 2
    assert result["error_max"] < 0.05
    assert len(result["bias_m"]) == 3


def test_static_survey_error_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        static_survey_error([np.zeros(3)], [np.zeros(3), np.zeros(3)])


def test_provenance_path_is_repo_relative_and_posix(tmp_path, monkeypatch):
    """Committed metadata must not carry the author's home directory.

    An absolute Windows path in a results file is wrong on every machine except
    the one that wrote it. It is the kind of wrong that no test notices,
    because nothing reads the field back.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "maps").mkdir()
    target = tmp_path / "maps" / "marker_map.yaml"
    target.touch()
    assert provenance_path(str(target)) == "maps/marker_map.yaml"


def test_provenance_path_leaves_a_path_outside_the_repository_alone(tmp_path, monkeypatch):
    """A map loaded from elsewhere really is elsewhere."""
    inside = tmp_path / "repo"
    inside.mkdir()
    outside = tmp_path / "elsewhere.yaml"
    outside.touch()
    monkeypatch.chdir(inside)
    assert provenance_path(str(outside)) == str(outside.resolve())


def test_provenance_path_passes_nothing_through():
    assert provenance_path(None) is None
    assert provenance_path("") == ""
