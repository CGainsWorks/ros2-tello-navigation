#!/usr/bin/env python3
"""How much of the tuning survives being wrong about the aircraft.

The controller gains in ``src/tello_bringup/config/control.yaml`` were tuned
against the simulated plant in ``src/tello_core/sim/dynamics.py``, whose velocity
time constants are estimates rather than measurements. That is a real weakness and
the honest way to report it is not a disclaimer, it is a number: fly the same
mission with the same gains against a range of plants and see what happens.

This script does that. It varies the horizontal velocity time constant, the
vertical one and the strength of the air disturbance, one at a time around the
nominal values and reports tracking error and whether the mission completed.

What a good result looks like: the mission completes across the whole range and
the error grows smoothly. What a bad result looks like: completion fails, or the
error jumps, at some value inside the range, which would mean the gains are tuned
to a plant nobody has measured and would not survive contact with an aircraft.

Writes ``results/tuning_sweep.csv``. ``docs/PID_TUNING.md`` quotes it.

It flies ``missions/hop.yaml`` rather than the reference mission, because a
sensitivity sweep needs many flights and the short mission exercises the same
loops in a fifth of the time.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.control.mission import Mission, MissionRunner, MissionSettings
from tello_core.control.pid import ControlGains
from tello_core.eval.metrics import evaluate
from tello_core.localisation.markers import MarkerMap
from tello_core.sim.dynamics import DynamicsSettings
from tello_core.sim.link import SimLink, SimSettings
from tello_core.sim.world import Room

# Each entry is a field of DynamicsSettings and the values to try. The nominal
# value is included so that the sweep contains its own baseline rather than
# relying on a number from another run.
SWEEPS = {
    "horizontal_tau": [0.20, 0.28, 0.35, 0.45, 0.60],
    "vertical_tau": [0.25, 0.35, 0.45, 0.60, 0.80],
    "drift_sigma": [0.0, 0.025, 0.05, 0.10, 0.20],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mission", default="missions/hop.yaml")
    parser.add_argument("--map", dest="marker_map", default="maps/marker_map.yaml")
    parser.add_argument("--config", default="src/tello_bringup/config/aircraft.yaml")
    parser.add_argument("--gains", default="src/tello_bringup/config/control.yaml")
    parser.add_argument("--out", default="results/tuning_sweep.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-flight-time", type=float, default=60.0)
    parser.add_argument("--only", nargs="*", default=None, help="limit to these parameters")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    marker_map = MarkerMap.from_yaml(REPO_ROOT / args.marker_map)
    config = AircraftConfig.from_yaml(REPO_ROOT / args.config)
    gains = ControlGains.from_yaml(REPO_ROOT / args.gains)
    camera = CameraModel.tello_nominal()
    mission = Mission.from_yaml(REPO_ROOT / args.mission)
    room = Room.from_marker_map(marker_map)

    nominal = DynamicsSettings()
    sweeps = {k: v for k, v in SWEEPS.items() if args.only is None or k in args.only}

    rows = []
    for parameter, values in sweeps.items():
        print(f"\n{parameter}  (nominal {getattr(nominal, parameter)})")
        for value in values:
            settings = replace(nominal, **{parameter: value})
            row = _fly(
                room, config, camera, marker_map, mission, gains, settings, args
            )
            row.update({"parameter": parameter, "value": value,
                        "nominal": getattr(nominal, parameter)})
            rows.append(row)
            print(
                f"  {value:<6} waypoints {row['waypoints_reached']}/{row['waypoints']}  "
                f"rmse {row['position_rmse_m'] * 100:5.1f} cm  "
                f"arrival {row['arrival_error_rmse_m'] * 100:5.1f} cm  "
                f"localised {row['availability'] * 100:5.1f} per cent  "
                f"{row['final_state']}"
            )

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")

    _verdict(rows)
    return 0


def _fly(room, config, camera, marker_map, mission, gains, dynamics, args) -> dict:
    link = SimLink(
        room,
        config,
        camera,
        SimSettings(seed=args.seed),
        dynamics_settings=dynamics,
        start_position=np.array(
            [mission.waypoints[0].position[0], mission.waypoints[0].position[1], 0.03]
        ),
        start_yaw=0.0,
    )
    runner = MissionRunner(
        link,
        mission,
        marker_map,
        config,
        camera,
        MissionSettings(
            control_rate=20.0, max_flight_time=args.max_flight_time, gains=gains
        ),
    )
    log = runner.run()
    metrics = evaluate(log)
    error = metrics.get("trajectory_error", {})
    return {
        "waypoints": log.metadata.get("waypoints", 0),
        "waypoints_reached": log.metadata.get("waypoints_reached", 0),
        "final_state": log.metadata.get("final_state", ""),
        "final_cause": log.metadata.get("final_cause", ""),
        "position_rmse_m": round(float(error.get("position_rmse_m", float("nan"))), 4),
        "position_p95_m": round(float(error.get("position_p95_m", float("nan"))), 4),
        "arrival_error_rmse_m": round(
            float(metrics.get("waypoints", {}).get("arrival_error_rmse", float("nan"))), 4
        ),
        "availability": round(
            float(metrics.get("localisation", {}).get("availability", float("nan"))), 4
        ),
        "consistency": round(float(error.get("consistency_ratio", float("nan"))), 3),
    }


def _verdict(rows) -> None:
    """Say plainly whether the gains survived the sweep."""
    print("\nverdict")
    failed = [r for r in rows if r["waypoints_reached"] < r["waypoints"]]
    if failed:
        print(f"  {len(failed)} of {len(rows)} plants did not complete the mission:")
        for row in failed:
            print(
                f"    {row['parameter']} = {row['value']}  "
                f"{row['waypoints_reached']}/{row['waypoints']}  "
                f"{row['final_state']} ({row['final_cause']})"
            )
        print(
            "  The gains are tuned to a narrower range of plants than the plant is\n"
            "  known to. They should not be trusted on hardware without retuning."
        )
    else:
        errors = [r["position_rmse_m"] for r in rows if np.isfinite(r["position_rmse_m"])]
        if errors:
            print(
                f"  every plant completed the mission. Position RMSE ranged from "
                f"{min(errors) * 100:.1f} to {max(errors) * 100:.1f} cm "
                f"(a factor of {max(errors) / max(min(errors), 1e-9):.1f})."
            )
        print(
            "  The gains tolerate the range of plants swept here. That is not the\n"
            "  same as tolerating a real Tello, because the sweep is around a plant\n"
            "  model that is itself unverified; see docs/HARDWARE.md."
        )


if __name__ == "__main__":
    raise SystemExit(main())
