"""Trajectory error metrics and which of them survive contact with hardware.

There are two kinds of number in this module and it matters which is which.

**Needs ground truth.** :func:`absolute_trajectory_error` and everything derived
from it compare the estimate with where the aircraft actually was. In simulation
that is available exactly. On a real Tello in a room it is not and no amount of
care makes it so: there is no motion-capture rig in this project's budget and a
tape measure gives you a handful of static points rather than a trajectory. So
every figure of this kind is a statement about the simulator and
``docs/RESULTS.md`` labels it as one.

**Works on hardware.** :func:`fix_residuals`, :func:`localisation_availability`,
:func:`reprojection_statistics` and :func:`static_survey_error` need no truth at
all. The first three come out of any flight log, simulated or real. The last is
the measurement the brief actually asks for: put the aircraft on a surveyed point,
read the estimate, subtract. It is a handful of samples rather than a trajectory
and it is the honest answer to "how accurately does this thing know where it is"
on real hardware.

The distinction is not pedantry. A simulated ATE of 4 cm is a statement about the
error model in ``src/tello_core/sim``; a static survey error of 4 cm on hardware
would be a statement about the aircraft. Reporting them in the same table without
saying which is which is how a portfolio project ends up claiming something it has
not shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..geometry import wrap_angles
from ..logs import FlightLog


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def summarise_errors(errors: Sequence[float], prefix: str = "") -> Dict[str, float]:
    """Mean, median, RMSE, 95th percentile and maximum of an error series.

    All five, deliberately. A mean alone hides a tail, a maximum alone is one
    sample of noise and RMSE is the one that matters for a controller because
    that is what the position loop integrates against.
    """
    values = _finite(np.asarray(errors, dtype=float))
    if values.size == 0:
        return {f"{prefix}{k}": float("nan") for k in ("mean", "median", "rmse", "p95", "max")}
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}p95": float(np.percentile(values, 95)),
        f"{prefix}max": float(np.max(values)),
    }


@dataclass
class TrajectoryError:
    """Position and heading error against ground truth."""

    samples: int
    position: Dict[str, float]
    axis_rmse: np.ndarray
    """Per-axis RMSE in metres. Reported because the axes fail differently: the
    vertical is held by a time-of-flight sensor and the horizontal by markers
    alone and an aggregate hides that."""

    yaw_degrees: Dict[str, float]
    consistency: float
    """Mean of ``error / sigma``. Around one means the filter's own uncertainty is
    honest; well under one means it is pessimistic; well over one means it is
    claiming an accuracy it does not have, which is the dangerous direction
    because the failsafe believes that claim."""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "samples": int(self.samples),
            **{f"position_{k}_m": v for k, v in self.position.items()},
            "axis_rmse_m": [float(v) for v in self.axis_rmse],
            **{f"yaw_{k}_deg": v for k, v in self.yaw_degrees.items()},
            "consistency_ratio": float(self.consistency),
        }


def _empty_trajectory_error() -> "TrajectoryError":
    return TrajectoryError(
        samples=0,
        position=summarise_errors([]),
        axis_rmse=np.full(3, np.nan),
        yaw_degrees=summarise_errors([]),
        consistency=float("nan"),
    )


def absolute_trajectory_error(log: FlightLog, airborne_only: bool = True) -> TrajectoryError:
    """Estimate against truth, for every logged cycle that has both.

    ``airborne_only`` drops the rows before take-off and after landing. Including
    them flatters the result: the aircraft is stationary on the floor with a
    marker in view, which is the easiest localisation problem it will face all
    flight and there are a lot of those rows.
    """
    columns = log.to_arrays()
    if not columns or "truth_x" not in columns:
        return _empty_trajectory_error()

    estimate = np.stack([columns["est_x"], columns["est_y"], columns["est_z"]], axis=1)
    truth = np.stack([columns["truth_x"], columns["truth_y"], columns["truth_z"]], axis=1)
    valid = np.isfinite(truth).all(axis=1) & np.isfinite(estimate).all(axis=1)
    if airborne_only:
        states = columns["state"]
        flying = np.array(
            [str(s) in {"TAKEOFF", "NAVIGATE", "HOLD", "RETURN", "LAND", "EMERGENCY_LAND"}
             for s in states]
        )
        valid &= flying

    if not valid.any():
        return _empty_trajectory_error()

    delta = estimate[valid] - truth[valid]
    distance = np.linalg.norm(delta, axis=1)
    axis_rmse = np.sqrt(np.mean(delta**2, axis=0))

    yaw_delta = columns["est_yaw_deg"][valid] - columns["truth_yaw_deg"][valid]
    yaw_error = np.rad2deg(np.abs(wrap_angles(np.deg2rad(yaw_delta))))

    sigma = _finite(columns["est_sigma_m"][valid])
    consistency = float("nan")
    if sigma.size and np.all(sigma > 0):
        consistency = float(np.mean(distance / np.maximum(sigma, 1e-6)))

    return TrajectoryError(
        samples=int(valid.sum()),
        position=summarise_errors(distance),
        axis_rmse=axis_rmse,
        yaw_degrees=summarise_errors(yaw_error),
        consistency=consistency,
    )


def localisation_availability(log: FlightLog) -> Dict[str, float]:
    """How much of the flight the system knew where it was.

    Needs no ground truth, so it is one of the figures that would carry over to a
    real flight unchanged. Availability is what actually decides whether a mission
    completes: an estimator that is superbly accurate for 60 per cent of the
    flight will trip the localisation failsafe repeatedly and land early.
    """
    columns = log.to_arrays()
    if not columns:
        return {"availability": float("nan"), "longest_gap_s": float("nan"), "samples": 0}

    times = columns["time_s"]
    localised = columns["localised"] > 0.5
    states = columns["state"]
    flying = np.array(
        [str(s) in {"TAKEOFF", "NAVIGATE", "HOLD", "RETURN", "LAND", "EMERGENCY_LAND"}
         for s in states]
    )
    if not flying.any():
        return {"availability": float("nan"), "longest_gap_s": float("nan"), "samples": 0}

    availability = float(np.mean(localised[flying]))

    longest = 0.0
    start: Optional[float] = None
    for time_s, is_localised, is_flying in zip(times, localised, flying):
        if not is_flying:
            continue
        if is_localised:
            if start is not None:
                longest = max(longest, float(time_s - start))
                start = None
        elif start is None:
            start = float(time_s)
    if start is not None:
        longest = max(longest, float(times[-1] - start))

    return {
        "availability": availability,
        "longest_gap_s": longest,
        "samples": int(flying.sum()),
    }


def fix_statistics(log: FlightLog) -> Dict[str, float]:
    """Rate, quality and reach of the marker fixes. No ground truth needed."""
    columns = log.to_arrays()
    if not columns:
        return {}
    have_fix = columns["n_markers"] > 0
    accepted = columns["fix_accepted"] > 0.5
    duration = float(columns["time_s"][-1] - columns["time_s"][0]) or float("nan")
    rmse = _finite(columns["fix_rmse_px"][have_fix])
    ranges = _finite(columns["fix_range_m"][have_fix])
    return {
        "fix_rate_hz": float(have_fix.sum() / duration) if np.isfinite(duration) else float("nan"),
        "fixes": int(have_fix.sum()),
        "fixes_accepted": int(accepted.sum()),
        "acceptance": float(accepted.sum() / max(1, have_fix.sum())),
        "mean_markers_per_fix": float(np.mean(columns["n_markers"][have_fix]))
        if have_fix.any()
        else float("nan"),
        "reprojection_rmse_px_median": float(np.median(rmse)) if rmse.size else float("nan"),
        "reprojection_rmse_px_p95": float(np.percentile(rmse, 95)) if rmse.size else float("nan"),
        "fix_range_m_median": float(np.median(ranges)) if ranges.size else float("nan"),
        "fix_range_m_max": float(np.max(ranges)) if ranges.size else float("nan"),
    }


def waypoint_accuracy(log: FlightLog) -> Dict[str, Any]:
    """Error at each waypoint arrival and how many were reached at all."""
    events = log.waypoint_events or []
    arrived = [e for e in events if str(e.get("kind")) == "arrived"]
    errors = [float(e.get("position_error_m", "nan")) for e in arrived]
    return {
        "waypoints_reached": len(arrived),
        "waypoints_timed_out": sum(1 for e in events if str(e.get("kind")) == "timeout"),
        **summarise_errors(errors, prefix="arrival_error_"),
    }


def fix_residuals(log: FlightLog) -> Dict[str, float]:
    """Distance between each raw marker fix and the fused estimate at that time.

    Available on hardware and the closest thing to an accuracy check that is. It
    does not measure absolute error, because both quantities can be wrong
    together but it does bound the *disagreement* between the two halves of the
    system and a system whose absolute error were large while its two halves
    agreed to a centimetre would need the marker map itself to be wrong.

    Computed from the log rather than during the flight, so it costs nothing in
    the air.
    """
    columns = log.to_arrays()
    if not columns or "fix_accepted" not in columns:
        return {}
    # The estimate on the row that reports a fix is the post-update estimate, so
    # this is a residual after correction rather than an innovation. That is the
    # quantity worth reporting: an innovation says how surprised the filter was,
    # a residual says how much of the surprise it kept.
    mask = columns["fix_accepted"] > 0.5
    if not mask.any():
        return {"residual_samples": 0}
    return {"residual_samples": int(mask.sum()), **summarise_errors(
        _finite(columns["fix_rmse_px"][mask]), prefix="reprojection_px_"
    )}


def static_survey_error(
    estimates: Sequence[np.ndarray], surveyed: Sequence[np.ndarray]
) -> Dict[str, Any]:
    """Estimator error at a set of surveyed points. The hardware-honest metric.

    On the aircraft: mark points on the floor with a tape measure, hover the
    aircraft over each in turn, hold it still and record what the estimator says.
    In simulation: ``scripts/survey_check.py`` does the same thing without the
    hovering, by placing the aircraft at each point.

    The result is a handful of samples rather than a trajectory and it is the
    only figure in this repository that could be quoted about a real Tello without
    a motion-capture system.
    """
    estimates = [np.asarray(e, dtype=float).ravel() for e in estimates]
    surveyed = [np.asarray(s, dtype=float).ravel() for s in surveyed]
    if len(estimates) != len(surveyed):
        raise ValueError("estimates and surveyed points must be the same length")
    if not estimates:
        return {"samples": 0}

    deltas = np.array([e - s for e, s in zip(estimates, surveyed)])
    distance = np.linalg.norm(deltas, axis=1)
    return {
        "samples": len(estimates),
        **summarise_errors(distance, prefix="error_"),
        "bias_m": [float(v) for v in np.mean(deltas, axis=0)],
        "axis_rmse_m": [float(v) for v in np.sqrt(np.mean(deltas**2, axis=0))],
    }


def reprojection_statistics(log: FlightLog) -> Dict[str, float]:
    """Distribution of the pose solve's residual. Needs no truth."""
    columns = log.to_arrays()
    if not columns:
        return {}
    values = _finite(columns["fix_rmse_px"])
    if values.size == 0:
        return {"samples": 0}
    return {
        "samples": int(values.size),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


def evaluate(log: FlightLog) -> Dict[str, Any]:
    """Every metric that applies to this log, in one dictionary.

    Metrics needing ground truth are simply absent when there is none, rather than
    present and NaN, so that a results file from a real flight cannot be read as
    if it contained an ATE that happened to be missing.
    """
    result: Dict[str, Any] = {
        "localisation": localisation_availability(log),
        "fixes": fix_statistics(log),
        "waypoints": waypoint_accuracy(log),
        "reprojection": reprojection_statistics(log),
        "residuals": fix_residuals(log),
        "metadata": log.metadata,
    }
    columns = log.to_arrays()
    if columns and "truth_x" in columns and np.isfinite(columns["truth_x"]).any():
        result["trajectory_error"] = absolute_trajectory_error(log).as_dict()
        result["ground_truth"] = "simulator"
    else:
        result["ground_truth"] = "none"
    return result

