#!/usr/bin/env python3
"""Turn the failsafe flight logs into one table and check they did what they should.

Reads every flight in ``results/failsafe/`` and reports, for each, what tripped,
how long it took to trip, what the aircraft did about it and where it ended up.
Writes ``results/failsafe_log.csv``, which is the table ``docs/FAILSAFE.md``
quotes.

It also asserts the expected outcome for each named case and exits non-zero if
one is wrong, so the failsafe evidence cannot quietly stop being evidence. The
expectations are deliberately about behaviour and not about numbers: "the link
dropped and the aircraft landed" is the claim worth defending and "it landed
4.3 seconds later" is a measurement that may legitimately change.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.logs import FlightLog

# What each case must end up doing.
#
# Three kinds of expectation, because "did the failsafe behave" is three
# different questions:
#
# `requires`   a cause that must appear. The link-loss case has to lose the link.
# `allows`     causes that may appear without being a failure. A nominal flight
#              is allowed a brief HOLD: the aircraft loses sight of the markers
#              for a couple of seconds while turning, holds, reacquires and
#              carries on. That is the failsafe working, not a fault and
#              forbidding it would mean either calling correct behaviour a
#              regression or loosening the localisation timeout until the
#              failsafe stopped protecting anything.
# `ends_clean` the flight must not end *because of* a failsafe. This is the
#              question a transient hold does not answer and the one that
#              matters: a mission that completed and then landed normally is a
#              success however many times it paused on the way.
EXPECTED = {
    "nominal": {
        "requires": set(),
        "allows": {"LOCALISATION_LOST"},
        "final": {"LANDED"},
        "ends_clean": True,
        "must_complete": True,
    },
    "link_loss": {
        "requires": {"LINK_LOSS"},
        "allows": {"LOCALISATION_LOST"},
        "final": {"LANDED", "LAND"},
        "must_complete": False,
    },
    "video_loss": {
        "requires": {"LOCALISATION_LOST"},
        "allows": set(),
        "final": {"LANDED", "LAND"},
        "must_complete": False,
    },
    "low_battery": {
        "requires": {"LOW_BATTERY"},
        "allows": {"LOCALISATION_LOST"},
        "final": {"LANDED", "LAND", "RETURN"},
        "must_complete": False,
    },
    "critical_battery": {
        "requires": {"CRITICAL_BATTERY"},
        "allows": {"LOW_BATTERY", "LOCALISATION_LOST"},
        "final": {"LANDED", "EMERGENCY_LAND", "LAND"},
        "must_complete": False,
    },
    # Already below the return threshold on the ground, so the flight is refused.
    # The aircraft ends where it started, which is the point.
    "refused_battery": {
        "requires": {"LOW_BATTERY"},
        "allows": set(),
        "final": {"LANDED"},
        "must_complete": False,
        "must_refuse": True,
    },
    # The case that stops the thresholds being set too tight. A failsafe that
    # fires on ordinary wireless attrition is a failsafe that gets turned off.
    "lossy_link": {
        "requires": set(),
        "allows": {"LOCALISATION_LOST"},
        "final": {"LANDED"},
        "ends_clean": True,
        "must_complete": True,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results/failsafe")
    parser.add_argument("--out", default="results/failsafe_log.csv")
    parser.add_argument(
        "--no-check", action="store_true", help="report without asserting the outcomes"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    directory = REPO_ROOT / args.results
    paths = sorted(directory.glob("*_trajectory.csv"))
    if not paths:
        print(f"no flights found in {directory}", file=sys.stderr)
        return 2

    rows = []
    failures = []
    for path in paths:
        name = path.name.replace("_trajectory.csv", "")
        log = FlightLog.read(path)
        row = _summarise(name, log)
        rows.append(row)
        if not args.no_check:
            failures.extend(_check(name, row, log))

    _write(REPO_ROOT / args.out, rows)
    _print(rows)

    if failures:
        print("\nfailures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


def _summarise(name: str, log: FlightLog) -> dict:
    events = log.failsafe_events or []
    triggers = [e for e in events if e.get("cause") not in ("NONE", "", None)]
    first = triggers[0] if triggers else None
    landed = next((e for e in events if e.get("to_state") == "LANDED"), None)

    columns = log.to_arrays()
    duration = float(columns["time_s"][-1]) if columns else float("nan")
    silent = 0
    if columns and "rc_sent" in columns:
        silent = int(np.sum(columns["rc_sent"] < 0.5))

    detect_time = float(first["time_s"]) if first else float("nan")
    land_time = float(landed["time_s"]) if landed else float("nan")

    return {
        "case": name,
        "cause": first["cause"] if first else "NONE",
        "detail": first["detail"] if first else "",
        "detected_s": round(detect_time, 2) if np.isfinite(detect_time) else "",
        "landed_s": round(land_time, 2) if np.isfinite(land_time) else "",
        "response_s": (
            round(land_time - detect_time, 2)
            if np.isfinite(detect_time) and np.isfinite(land_time)
            else ""
        ),
        "final_state": log.metadata.get("final_state", ""),
        "final_cause": log.metadata.get("final_cause", ""),
        "aborted": log.metadata.get("aborted", ""),
        "waypoints_reached": log.metadata.get("waypoints_reached", ""),
        "waypoints": log.metadata.get("waypoints", ""),
        "flight_s": round(duration, 2) if np.isfinite(duration) else "",
        "silent_cycles": silent,
        "transitions": len(events),
    }


def _check(name: str, row: dict, log: FlightLog) -> list:
    expected = EXPECTED.get(name)
    if expected is None:
        return []
    problems = []
    causes = {e.get("cause") for e in (log.failsafe_events or [])} - {"NONE", "", None}

    missing = expected["requires"] - causes
    if missing:
        problems.append(
            f"{name}: expected {sorted(missing)}, got {sorted(causes) or 'none'}"
        )

    unexpected = causes - expected["requires"] - expected["allows"]
    if unexpected:
        problems.append(f"{name}: unexpected trigger {sorted(unexpected)}")

    if row["final_state"] not in expected["final"]:
        problems.append(
            f"{name}: ended in {row['final_state']}, expected one of {sorted(expected['final'])}"
        )

    if expected.get("ends_clean") and log.metadata.get("final_cause") not in ("NONE", None):
        problems.append(
            f"{name}: the flight ended because of {log.metadata.get('final_cause')}. A "
            "transient hold is allowed; being landed by a failsafe is not."
        )

    if expected["must_complete"] and row["waypoints_reached"] != row["waypoints"]:
        problems.append(
            f"{name}: reached {row['waypoints_reached']} of {row['waypoints']} waypoints"
        )

    if expected.get("must_refuse") and not log.metadata.get("aborted"):
        problems.append(
            f"{name}: the flight should have been refused before take-off but the "
            "log is not marked as aborted"
        )

    if name == "link_loss" and row["silent_cycles"] == 0:
        problems.append(
            "link_loss: the controller never fell silent. On a degraded link that "
            "keeps resetting the aircraft's own dead-man timer, which is the one "
            "thing that must not happen."
        )
    return problems


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def _print(rows) -> None:
    print()
    header = (
        f"{'case':<18} {'first trigger':<20} {'detected':>9} {'landed':>8} "
        f"{'ended':<16} {'because':<20} {'wp'}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<18} {row['cause']:<20} {str(row['detected_s']):>9} "
            f"{str(row['landed_s']):>8} {row['final_state']:<16} "
            f"{row['final_cause']:<20} {row['waypoints_reached']}/{row['waypoints']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
