#!/usr/bin/env python3
"""Assert that a flight log shows what it should. Used by CI, runnable locally.

The assertions live here rather than inline in the workflow for one reason: when
CI says a flight was wrong, the first thing anyone wants is to run the same check
on their own machine and see the same message.

What it checks and why each is worth a line of CI time:

* **It landed.** An autonomous flight that ends any other way ended because
  something stopped running. This is the check that would catch the mission
  runner hanging.
* **The flight was not ended by a failsafe.** With no ``--expect-cause``, a
  flight whose ``final_cause`` is anything other than ``NONE`` is a failure. Note
  what this does *not* say: a trigger the aircraft recovered from is allowed. A
  flight that loses sight of the markers while turning, holds, reacquires and
  carries on has demonstrated the failsafe working rather than a fault and
  failing CI for it would mean either calling correct behaviour a regression or
  loosening the localisation timeout until the failsafe protected nothing.
* **It did not trip repeatedly.** ``--max-triggers`` bounds how many times it may
  recover, so a flight that starts spending half its time in HOLD still fails
  even though every hold ends. That is the regression signal the blanket
  no-triggers rule was really providing.
* **The expected cause did trip**, when one is named. The link-loss case has to
  actually lose the link.
* **The controller fell silent**, when ``--expect-silent`` is given. This is the
  one failsafe behaviour that is both counter-intuitive and invisible from the
  outside: on link loss the controller must stop transmitting, because on a
  degraded link an rc command that gets through resets the aircraft's own
  fifteen-second dead-man timer and keeps an uncontrolled aircraft airborne.
* **The estimate stayed sane.** A loose bound, not an accuracy claim: the exact
  figures belong in docs/RESULTS.md and a tight threshold here would fail on a
  different OpenCV version for reasons that have nothing to do with correctness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.eval.metrics import evaluate
from tello_core.logs import FlightLog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results/ci")
    parser.add_argument("--stem", default="smoke")
    parser.add_argument(
        "--expect-cause",
        default=None,
        help="a failsafe cause that must appear. Without it, any trigger fails.",
    )
    parser.add_argument(
        "--max-triggers",
        type=int,
        default=2,
        help="how many failsafe triggers the flight may recover from before it "
             "counts as a failure. Only applies without --expect-cause.",
    )
    parser.add_argument(
        "--expect-silent",
        action="store_true",
        help="require that the controller stopped transmitting at some point",
    )
    parser.add_argument(
        "--expect-waypoints",
        type=int,
        default=None,
        help="number of waypoints that must have been reached",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=0.60,
        help="loose upper bound on position RMSE up to the first failsafe trigger, "
             "metres. Not an accuracy claim.",
    )
    parser.add_argument(
        "--min-availability",
        type=float,
        default=0.50,
        help="fraction of the flight that must have been localised",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path = REPO_ROOT / args.results / f"{args.stem}_trajectory.csv"
    if not path.is_file():
        print(f"no flight log at {path}", file=sys.stderr)
        return 2

    log = FlightLog.read(path)
    metrics = evaluate(log)
    problems = []

    final = log.metadata.get("final_state")
    if log.metadata.get("aborted"):
        problems.append(f"the flight was abandoned before take-off: {log.metadata['aborted']}")
    elif final != "LANDED":
        problems.append(f"ended in {final}, not LANDED")

    causes = {
        event.get("cause")
        for event in (log.failsafe_events or [])
        if event.get("cause") not in ("NONE", "", None)
    }
    triggers = [
        event
        for event in (log.failsafe_events or [])
        if event.get("cause") not in ("NONE", "", None)
    ]
    final_cause = log.metadata.get("final_cause")
    if args.expect_cause:
        if args.expect_cause not in causes:
            problems.append(
                f"expected a {args.expect_cause} trigger, got {sorted(causes) or 'none'}"
            )
    else:
        if final_cause not in ("NONE", "", None):
            problems.append(
                f"the flight ended because of {final_cause}. A trigger it recovers "
                "from is allowed; being landed by a failsafe is not."
            )
        if len(triggers) > args.max_triggers:
            problems.append(
                f"tripped {len(triggers)} times ({sorted(causes)}), above the "
                f"limit of {args.max_triggers}. Each one was recovered from, so "
                "nothing here is unsafe. A flight that keeps stopping is still a "
                "flight whose perception or planning has got worse."
            )

    columns = log.to_arrays()
    if args.expect_silent:
        silent = int(np.sum(columns["rc_sent"] < 0.5)) if "rc_sent" in columns else 0
        if silent == 0:
            problems.append(
                "the controller never stopped transmitting. On a degraded link that "
                "keeps resetting the aircraft's own dead-man timer, which is the one "
                "thing the link-loss response must not do."
            )

    if args.expect_waypoints is not None:
        reached = int(log.metadata.get("waypoints_reached", 0))
        if reached < args.expect_waypoints:
            problems.append(
                f"reached {reached} waypoints, expected at least {args.expect_waypoints}"
            )

    availability = metrics.get("localisation", {}).get("availability", float("nan"))
    if np.isfinite(availability) and availability < args.min_availability:
        problems.append(
            f"localised for only {availability * 100:.0f} per cent of the flight, "
            f"below the {args.min_availability * 100:.0f} per cent floor"
        )

    rmse = _rmse_before_the_trigger(log, columns)
    if np.isfinite(rmse) and rmse > args.max_rmse:
        problems.append(
            f"position RMSE {rmse * 100:.0f} cm, above the "
            f"{args.max_rmse * 100:.0f} cm bound"
        )

    _report(log, metrics, problems)
    return 1 if problems else 0


def _rmse_before_the_trigger(log: FlightLog, columns) -> float:
    """Position RMSE up to the first failsafe trigger, or over the whole flight.

    Holding the whole flight to a nominal accuracy bound would be wrong for a
    failure case and wrong in a way that hides the thing worth checking. Once the
    link has dropped the estimator has no telemetry and is dead reckoning by
    design, so the error after that point is a measure of how long the descent
    took, not of whether the navigation works. The link-loss flight scores about
    125 cm over its whole duration and about 10 cm up to the moment the link went.

    Before the trigger is the part that has to be right.
    """
    if not columns or "truth_error_m" not in columns:
        return float("nan")

    triggers = [
        float(event["time_s"])
        for event in (log.failsafe_events or [])
        if event.get("cause") not in ("NONE", "", None)
    ]
    time = columns["time_s"]
    error = columns["truth_error_m"]
    states = [str(state) for state in columns["state"]]
    flying = np.array(
        [s in {"TAKEOFF", "NAVIGATE", "HOLD", "RETURN", "LAND", "EMERGENCY_LAND"}
         for s in states]
    )
    mask = flying & np.isfinite(error)
    if triggers:
        mask &= time <= min(triggers)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(error[mask] ** 2)))


def _report(log: FlightLog, metrics, problems) -> None:
    metadata = log.metadata
    duration = log.rows[-1].time_s if log.rows else float("nan")
    print(f"flight        {metadata.get('mission', '?')} ({metadata.get('fault_schedule', '?')})")
    print(f"duration      {duration:.1f} s")
    print(f"final state   {metadata.get('final_state')} ({metadata.get('final_cause')})")
    print(
        f"waypoints     {metadata.get('waypoints_reached')} of {metadata.get('waypoints')}"
    )
    localisation = metrics.get("localisation", {})
    print(
        f"localised     {localisation.get('availability', float('nan')) * 100:.1f} per cent, "
        f"longest gap {localisation.get('longest_gap_s', float('nan')):.1f} s"
    )
    error = metrics.get("trajectory_error")
    if error:
        print(f"position rmse {error['position_rmse_m'] * 100:.1f} cm over the whole flight")
        columns = log.to_arrays()
        before = _rmse_before_the_trigger(log, columns)
        if np.isfinite(before):
            print(f"              {before * 100:.1f} cm up to the first failsafe trigger")
    for event in log.failsafe_events or []:
        print(
            f"  {event['time_s']:>7}s  {event['from_state']} -> {event['to_state']}"
            f"  {event['cause']}  {event['detail']}"
        )
    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
    else:
        print("ok")


if __name__ == "__main__":
    raise SystemExit(main())
