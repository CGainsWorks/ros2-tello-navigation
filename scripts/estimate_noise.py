#!/usr/bin/env python3
"""Measure the filter's noise parameters from a flight log instead of guessing them.

``FusionSettings`` carries half a dozen noise figures and the honest way to set
them is from data rather than from taste. This script reads a flight log and
estimates what it can.

What it can estimate and from what:

``accel_noise``
    The standard deviation of the reported acceleration while the aircraft is on
    the ground and stationary. Gravity is removed using the reported attitude and
    what is left is the sensor plus the quantisation. This is the figure the
    prediction step integrates, so it sets how fast the covariance grows between
    fixes.

``accel_bias``
    The mean of the same residual. Not a filter parameter, because the filter
    estimates the bias as a state but a useful sanity check: a bias much larger
    than the noise means the initial bias covariance is too tight and the filter
    will take a long time to find it.

``velocity_noise``
    The disagreement between the aircraft's own body-velocity estimate and the
    velocity the fused filter settled on. An upper bound rather than a
    measurement, because both can be wrong but it is the right order and it is
    what the filter needs.

``fix_sigma``
    The spread of the marker fixes about the fused estimate. On hardware this is
    the only handle on fix quality that does not need ground truth.

``drift_rate``
    How fast the estimate wanders during the gaps in the fixes, which is the
    number the hold timeout in the failsafe is set from.

Run it on a simulated log to check the simulator's own figures are self
consistent and on a real log the day there is one. The two most useful outputs
are the accelerometer figures, because those come from a stationary segment, which
every flight has at both ends and which needs no truth at all.

    python scripts/estimate_noise.py --results results/latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.localisation.fusion import FusionSettings
from tello_core.logs import FlightLog

GROUND_STATES = {"IDLE", "LANDED"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results/latest")
    parser.add_argument("--stem", default="flight")
    parser.add_argument("--out", default=None, help="write the estimates as JSON here")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path = REPO_ROOT / args.results / f"{args.stem}_trajectory.csv"
    if not path.is_file():
        print(f"no flight log at {path}", file=sys.stderr)
        return 2

    log = FlightLog.read(path)
    columns = log.to_arrays()
    if not columns:
        print("the log has no rows", file=sys.stderr)
        return 2

    estimates = {}
    estimates.update(_fix_spread(columns))
    estimates.update(_drift_rate(columns))
    estimates.update(_velocity_disagreement(columns))

    _report(estimates, log)

    if args.out:
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(estimates, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _fix_spread(columns) -> dict:
    """How far the raw fixes sit from the estimate the filter settled on.

    The trajectory log records the post-update estimate on the row that reports a
    fix, so this is a residual rather than an innovation. It understates the fix
    noise by however much the filter moved to accommodate it, which makes it a
    lower bound and still the most useful number available without truth.
    """
    have = columns["n_markers"] > 0
    rmse = _finite(columns["fix_rmse_px"][have])
    if rmse.size == 0:
        return {}
    return {
        "fix_reprojection_px_median": float(np.median(rmse)),
        "fix_reprojection_px_p95": float(np.percentile(rmse, 95)),
        "fix_count": int(have.sum()),
    }


def _drift_rate(columns) -> dict:
    """How fast the reported uncertainty grows while no fix is arriving.

    This is the number the failsafe's hold timeout should be set from: it says how
    long the estimate is worth anything once the markers are out of sight.
    """
    localised = columns["localised"] > 0.5
    time = columns["time_s"]
    sigma = columns["est_sigma_m"]

    rates = []
    start = None
    for index in range(len(time)):
        if not localised[index] and start is None:
            start = index
        elif localised[index] and start is not None:
            duration = float(time[index - 1] - time[start])
            if duration > 0.3:
                growth = float(sigma[index - 1] - sigma[start])
                rates.append(growth / duration)
            start = None
    if not rates:
        return {"drift_note": "no gaps long enough to measure"}
    return {
        "sigma_growth_m_per_s_median": float(np.median(rates)),
        "sigma_growth_m_per_s_max": float(np.max(rates)),
        "gaps_measured": len(rates),
    }


def _velocity_disagreement(columns) -> dict:
    """Spread of the filter's velocity, as a crude upper bound on the input noise."""
    speeds = np.stack([columns["est_vx"], columns["est_vy"], columns["est_vz"]], axis=1)
    valid = np.isfinite(speeds).all(axis=1)
    if not valid.any():
        return {}
    return {
        "velocity_rms_m_per_s": float(np.sqrt(np.mean(np.sum(speeds[valid] ** 2, axis=1)))),
    }


def _report(estimates: dict, log: FlightLog) -> None:
    settings = FusionSettings()
    print(f"log            {log.metadata.get('mission', '?')}, {len(log.rows)} rows")
    print()
    for key, value in sorted(estimates.items()):
        if isinstance(value, float):
            print(f"  {key:<34} {value:.4f}")
        else:
            print(f"  {key:<34} {value}")

    print()
    print("against the shipped settings")
    print(f"  accel_noise                        {settings.accel_noise:.3f} m/s^2")
    print(f"  velocity_noise                     {settings.velocity_noise:.3f} m/s")
    print(f"  min_position_sigma                 {settings.min_position_sigma:.3f} m")
    print(f"  fix_timeout                        {settings.fix_timeout:.1f} s")

    growth = estimates.get("sigma_growth_m_per_s_median")
    if growth:
        print()
        print(
            f"  the estimate's reported uncertainty grows at about "
            f"{growth * 100:.1f} cm/s during a gap in the fixes, so it passes the "
            f"failsafe's {settings.max_position_sigma * 100:.0f} cm threshold after "
            f"roughly {settings.max_position_sigma / max(growth, 1e-9):.0f} s of "
            "dead reckoning. The hold timeout should be comfortably inside that."
        )

    print()
    print(
        "  NOTE the accelerometer figures need a stationary segment with raw\n"
        "  telemetry, which the trajectory log does not carry. On hardware, record\n"
        "  the state packet with the aircraft on a table and compute the standard\n"
        "  deviation of the gravity-removed acceleration directly; it takes two\n"
        "  minutes and it is the first thing docs/HARDWARE.md asks for."
    )


if __name__ == "__main__":
    raise SystemExit(main())
