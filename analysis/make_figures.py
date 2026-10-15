#!/usr/bin/env python3
"""Regenerate every figure from the committed results.

Reads CSV and writes PNG. Nothing here recomputes a result: if a figure and a
number in ``docs/RESULTS.md`` disagree, the figure is stale and this script fixes
it and if the underlying flight needs re-flying that is
``scripts/run_mission.py``'s job. Keeping the two separate is what makes it safe
to regenerate the figures on a machine that would take twenty minutes to re-fly
the mission.

Figures produced and what each is for:

``trajectory.png``
    The flown path against the commanded one, in plan view and in height. The
    first thing to look at and the one that makes a controller problem obvious.
``error.png``
    Position error over time against the filter's own reported uncertainty. A
    filter whose error sits outside its own one-sigma band most of the time is
    lying to the failsafe and this is where that shows.
``localisation.png``
    When fixes arrived, how many markers each used and when the aircraft was
    localised. Explains every gap in the error plot.
``detection.png``
    The measured detector curves: range, yaw rate, incidence. The evidence behind
    the marker size, the controller's yaw limit and the planner's incidence
    limit.
``failsafe.png``
    Each failure case as a height trace with the trigger marked. One picture for
    the whole of docs/FAILSAFE.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tello_core.localisation.markers import MarkerMap
from tello_core.logs import FlightLog

# One place for the look of every figure. Deliberately plain: these are
# engineering plots and a reader needs to be able to read the numbers off them.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

FLYING = {"TAKEOFF", "NAVIGATE", "HOLD", "RETURN", "LAND", "EMERGENCY_LAND"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results/latest")
    parser.add_argument("--stem", default="flight")
    parser.add_argument("--out", default="analysis/figures")
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument(
        "--failsafe", default="results/failsafe", help="directory of failsafe-case flights"
    )
    parser.add_argument("--detection", default="results", help="directory of detection sweeps")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    written = []

    trajectory = REPO_ROOT / args.results / f"{args.stem}_trajectory.csv"
    if trajectory.is_file():
        log = FlightLog.read(trajectory)
        marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
        written.append(plot_trajectory(log, marker_map, out / "trajectory.png"))
        written.append(plot_error(log, out / "error.png"))
        written.append(plot_localisation(log, out / "localisation.png"))
    else:
        print(f"no flight log at {trajectory}, skipping the flight figures")

    detection = plot_detection(REPO_ROOT / args.detection, out / "detection.png")
    if detection:
        written.append(detection)

    failsafe = plot_failsafe(REPO_ROOT / args.failsafe, out / "failsafe.png")
    if failsafe:
        written.append(failsafe)

    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0 if written else 1


def _flying_mask(columns) -> np.ndarray:
    return np.array([str(state) in FLYING for state in columns["state"]])


def plot_trajectory(log: FlightLog, marker_map: MarkerMap, path: Path) -> Path:
    columns = log.to_arrays()
    flying = _flying_mask(columns)
    figure, (plan, height) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    # Plan view. The room and the markers first, so the path is read against the
    # thing it is navigating from.
    positions = marker_map.positions()
    plan.scatter(
        positions[:, 0], positions[:, 1], marker="s", s=28, c="#444444",
        label=f"markers ({len(marker_map)})", zorder=3,
    )
    for marker in marker_map:
        normal = marker.normal_world()
        plan.arrow(
            marker.position[0], marker.position[1], normal[0] * 0.22, normal[1] * 0.22,
            head_width=0.06, head_length=0.06, fc="#888888", ec="#888888", zorder=2,
        )

    if np.isfinite(columns["truth_x"]).any():
        plan.plot(
            columns["truth_x"][flying], columns["truth_y"][flying],
            color="#1b6ca8", lw=1.8, label="truth", zorder=5,
        )
    plan.plot(
        columns["est_x"][flying], columns["est_y"][flying],
        color="#d1495b", lw=1.1, ls="--", label="estimate", zorder=6,
    )

    target = np.isfinite(columns["target_x"])
    if target.any():
        plan.scatter(
            columns["target_x"][target], columns["target_y"][target],
            marker="x", s=18, c="#2a9d8f", alpha=0.35, label="commanded", zorder=4,
        )

    plan.set_xlabel("x (m)")
    plan.set_ylabel("y (m)")
    plan.set_title("plan view")
    plan.set_aspect("equal")
    plan.legend(loc="upper right", fontsize=8)

    # Height against time, with the states shaded. The shading is what turns a
    # height trace into a story: the descent at the end is a landing and a
    # reader can see which kind.
    time = columns["time_s"]
    if np.isfinite(columns["truth_z"]).any():
        height.plot(time, columns["truth_z"], color="#1b6ca8", lw=1.6, label="truth")
    height.plot(time, columns["est_z"], color="#d1495b", lw=1.0, ls="--", label="estimate")
    _shade_states(height, columns)
    height.set_xlabel("time (s)")
    height.set_ylabel("height (m)")
    height.set_title("height and flight state")
    height.legend(loc="lower left", fontsize=8)

    figure.suptitle(
        f"{log.metadata.get('mission', 'flight')}: "
        f"{log.metadata.get('waypoints_reached', '?')} of "
        f"{log.metadata.get('waypoints', '?')} waypoints, "
        f"ended {log.metadata.get('final_state', '?')}",
        fontsize=10,
    )
    figure.savefig(path)
    plt.close(figure)
    return path


def _shade_states(axis, columns) -> None:
    """Shade the time axis by flight state, one band per contiguous run."""
    colours = {
        "IDLE": "#f2f2f2",
        "TAKEOFF": "#e8f4ea",
        "NAVIGATE": "#ffffff",
        "HOLD": "#fff6e0",
        "RETURN": "#e8eef8",
        "LAND": "#fde8e8",
        "EMERGENCY_LAND": "#f8d7da",
        "LANDED": "#f2f2f2",
    }
    time = columns["time_s"]
    states = [str(state) for state in columns["state"]]
    start = 0
    seen = set()
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            state = states[start]
            axis.axvspan(
                time[start], time[min(index, len(time) - 1)],
                color=colours.get(state, "#ffffff"), alpha=0.8, lw=0,
                label=state if state not in seen and state != "NAVIGATE" else None,
            )
            seen.add(state)
            start = index


def plot_error(log: FlightLog, path: Path) -> Path:
    columns = log.to_arrays()
    flying = _flying_mask(columns)
    time = columns["time_s"][flying]

    figure, (error_axis, sigma_axis) = plt.subplots(
        2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    if np.isfinite(columns["truth_error_m"]).any():
        error = columns["truth_error_m"][flying]
        sigma = columns["est_sigma_m"][flying]
        error_axis.plot(time, error * 100, color="#d1495b", lw=1.2, label="position error")
        error_axis.fill_between(
            time, 0, sigma * 100, color="#1b6ca8", alpha=0.18,
            label="filter's own one sigma",
        )
        error_axis.fill_between(
            time, 0, 3 * sigma * 100, color="#1b6ca8", alpha=0.08, label="three sigma",
        )
        finite = error[np.isfinite(error)]
        if finite.size:
            error_axis.axhline(
                float(np.sqrt(np.mean(finite**2))) * 100, color="#444444", ls=":", lw=1,
                label="rmse",
            )
        error_axis.set_ylabel("error (cm)")
        error_axis.set_title(
            "position error against the uncertainty the filter reported\n"
            "(error outside the shaded band means the filter is overconfident, "
            "which is the dangerous direction)",
            fontsize=9,
        )
        error_axis.legend(loc="upper left", fontsize=8, ncol=2)
    else:
        error_axis.text(
            0.5, 0.5, "no ground truth in this log", ha="center", va="center",
            transform=error_axis.transAxes,
        )

    sigma_axis.plot(
        columns["time_s"], columns["est_sigma_m"] * 100, color="#1b6ca8", lw=1.0,
        label="reported sigma",
    )
    localised = columns["localised"] > 0.5
    sigma_axis.fill_between(
        columns["time_s"], 0, 1, where=~localised, transform=sigma_axis.get_xaxis_transform(),
        color="#f8d7da", alpha=0.6, lw=0, label="not localised",
    )
    sigma_axis.set_xlabel("time (s)")
    sigma_axis.set_ylabel("sigma (cm)")
    sigma_axis.legend(loc="upper left", fontsize=8)

    figure.savefig(path)
    plt.close(figure)
    return path


def plot_localisation(log: FlightLog, path: Path) -> Path:
    columns = log.to_arrays()
    figure, (markers_axis, range_axis) = plt.subplots(
        2, 1, figsize=(9, 5.0), sharex=True
    )

    have_fix = columns["n_markers"] > 0
    accepted = columns["fix_accepted"] > 0.5
    markers_axis.scatter(
        columns["time_s"][have_fix & accepted], columns["n_markers"][have_fix & accepted],
        s=12, c="#2a9d8f", label="accepted",
    )
    markers_axis.scatter(
        columns["time_s"][have_fix & ~accepted], columns["n_markers"][have_fix & ~accepted],
        s=16, c="#d1495b", marker="x", label="gated out",
    )
    markers_axis.set_ylabel("markers in fix")
    markers_axis.set_ylim(bottom=0)
    markers_axis.set_title(
        "marker fixes: how many markers each used and whether the filter took it",
        fontsize=9,
    )
    markers_axis.legend(loc="upper right", fontsize=8)

    range_axis.plot(
        columns["time_s"][have_fix], columns["fix_range_m"][have_fix],
        color="#1b6ca8", lw=0.8, marker=".", ms=3, label="range to markers",
    )
    twin = range_axis.twinx()
    twin.plot(
        columns["time_s"][have_fix], columns["fix_rmse_px"][have_fix],
        color="#e9c46a", lw=0.8, label="reprojection residual",
    )
    twin.set_ylabel("residual (px)")
    twin.grid(False)
    range_axis.set_xlabel("time (s)")
    range_axis.set_ylabel("range (m)")
    handles = range_axis.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = range_axis.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    range_axis.legend(handles, labels, loc="upper right", fontsize=8)

    figure.savefig(path)
    plt.close(figure)
    return path


def _read_csv(path: Path):
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows, key) -> np.ndarray:
    return np.array([float(row[key]) if row.get(key) not in (None, "") else np.nan
                     for row in rows])


def plot_detection(directory: Path, path: Path):
    sweeps = {
        "range": _read_csv(directory / "detection_range.csv"),
        "yaw": _read_csv(directory / "detection_yaw_rate.csv"),
        "incidence": _read_csv(directory / "detection_incidence.csv"),
    }
    if not any(sweeps.values()):
        print(f"no detection sweeps in {directory}, skipping detection.png")
        return None

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    if sweeps["range"]:
        rows = sweeps["range"]
        axis = axes[0]
        axis.plot(_column(rows, "range_m"), _column(rows, "decode_rate") * 100,
                  color="#1b6ca8", marker="o", ms=3, label="decoded")
        axis.plot(_column(rows, "range_m"), _column(rows, "fix_rate") * 100,
                  color="#2a9d8f", marker=".", ms=3, ls="--", label="gave a fix")
        axis.set_xlabel("range (m)")
        axis.set_ylabel("per cent of frames")
        axis.set_title("detection against range", fontsize=9)
        axis.set_ylim(-5, 105)
        axis.legend(fontsize=8)

    if sweeps["yaw"]:
        rows = sweeps["yaw"]
        axis = axes[1]
        axis.plot(_column(rows, "yaw_rate_dps"), _column(rows, "decode_rate") * 100,
                  color="#d1495b", marker="o", ms=3)
        axis.axvline(30.0, color="#444444", ls=":", lw=1)
        axis.text(31, 50, "controller limit", fontsize=7, rotation=90, va="center")
        axis.set_xlabel("yaw rate (deg/s)")
        axis.set_title("detection against turn rate\n(motion blur)", fontsize=9)
        axis.set_ylim(-5, 105)

    if sweeps["incidence"]:
        rows = sweeps["incidence"]
        axis = axes[2]
        axis.plot(_column(rows, "incidence_deg"), _column(rows, "decode_rate") * 100,
                  color="#e76f51", marker="o", ms=3)
        axis.axvline(50.0, color="#444444", ls=":", lw=1)
        axis.text(51, 50, "planner limit", fontsize=7, rotation=90, va="center")
        axis.set_xlabel("incidence (deg)")
        axis.set_title("detection against viewing angle", fontsize=9)
        axis.set_ylim(-5, 105)

    figure.suptitle(
        "measured detector performance, 0.30 m markers through the simulated camera",
        fontsize=10,
    )
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_failsafe(directory: Path, path: Path):
    paths = sorted(directory.glob("*_trajectory.csv"))
    if not paths:
        print(f"no failsafe flights in {directory}, skipping failsafe.png")
        return None

    rows = int(np.ceil(len(paths) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(11, 2.4 * rows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for axis, flight in zip(axes, paths):
        log = FlightLog.read(flight)
        columns = log.to_arrays()
        name = flight.name.replace("_trajectory.csv", "")
        truth = columns.get("truth_z")
        if truth is not None and np.isfinite(truth).any():
            axis.plot(columns["time_s"], truth, color="#1b6ca8", lw=1.4, label="height")
        else:
            axis.plot(columns["time_s"], columns["est_z"], color="#1b6ca8", lw=1.4,
                      label="height (estimated)")
        _shade_states(axis, columns)

        for event in log.failsafe_events or []:
            if event.get("cause") in ("NONE", "", None):
                continue
            axis.axvline(float(event["time_s"]), color="#d1495b", ls="--", lw=1.2)
            axis.text(
                float(event["time_s"]), axis.get_ylim()[1] * 0.92, f" {event['cause']}",
                fontsize=7, color="#d1495b", va="top",
            )
        axis.set_title(f"{name}  ->  {log.metadata.get('final_state', '?')}", fontsize=9)
        axis.set_ylabel("height (m)")
        axis.set_xlabel("time (s)")

    for axis in axes[len(paths):]:
        axis.axis("off")

    figure.suptitle("every failure case, with the trigger marked", fontsize=10)
    figure.savefig(path)
    plt.close(figure)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
