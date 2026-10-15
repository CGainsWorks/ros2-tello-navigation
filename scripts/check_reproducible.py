#!/usr/bin/env python3
"""Check that the committed results can still be produced from the committed code.

What this compares and what it deliberately does not
-----------------------------------------------------
It re-flies the reference mission with the committed seed and compares the
headline metrics with ``results/reference_metrics.json`` inside a stated
tolerance. It does **not** diff the trajectory CSV byte for byte and that is a
decision rather than an oversight.

Byte-identical reproduction across machines is not achievable here and a check
that pretends otherwise is worse than no check at all. The reasons are concrete:
OpenCV's QR decoder and its sub-pixel corner refinement are not bit-stable across
versions, ``solvePnP`` uses an iterative solve whose last digit depends on the
BLAS and matplotlib does not render byte-identical PNGs across versions. A check
built on any of those fails on every run for reasons unrelated to correctness and
a check that always fails is a check nobody reads.

So the contract is in two parts. The number of waypoints reached and the state
the flight ended in must match **exactly**: those are the claims worth defending
and neither can move for an innocent reason. Every statistic is compared within a
wide per-metric tolerance, which catches a gross change and deliberately does not
try to catch a small one, because a small one is what a different OpenCV build
produces by itself.

What a failure means
--------------------
Either the committed results are stale and should be regenerated with
``make demo``, or something has stopped being reproducible. Both are worth
knowing about and the output says which metric moved and by how much so that it
is possible to tell the two apart.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

# Metric, how to reach it in the metrics dictionary and how far it may move.
#
# The tolerances are absolute and chosen per quantity rather than as one blanket
# percentage, because these numbers have very different sensitivities. The number
# of waypoints reached is an integer and must match exactly: a mission that
# stopped completing is a regression regardless of why. Position error may move a
# few centimetres on a different OpenCV version, because a sub-pixel corner
# refinement that lands half a pixel differently moves a fix by millimetres and
# the controller then flies a slightly different path. Availability is a fraction
# of a few hundred samples, so one extra dropped frame moves it by a fifth of a
# per cent.
CHECKS = [
    # Exact. These two are the real regression guard: a mission that stopped
    # completing, or a flight that stopped ending on the ground, is a regression
    # regardless of why and neither can move for a reason as innocent as a
    # different OpenCV build.
    ("waypoints reached", ("metadata", "waypoints_reached"), 0.0, "{:.0f}"),
    ("final state", ("metadata", "final_state"), None, "{}"),
    # Tolerances and they are deliberately wide. Half a pixel of difference in
    # where a sub-pixel corner refinement lands moves a fix by millimetres, the
    # controller then flies a slightly different path and by the end of a
    # thirty-second flight the trajectory statistics have moved by centimetres.
    # That is not a regression and a check that called it one would be a check
    # that failed every time the runner image was updated. These catch a gross
    # change, which is what they are for.
    ("position rmse (m)", ("trajectory_error", "position_rmse_m"), 0.15, "{:.3f}"),
    ("position p95 (m)", ("trajectory_error", "position_p95_m"), 0.25, "{:.3f}"),
    ("yaw rmse (deg)", ("trajectory_error", "yaw_rmse_deg"), 2.5, "{:.2f}"),
    ("localisation availability", ("localisation", "availability"), 0.20, "{:.3f}"),
    ("fix acceptance", ("fixes", "acceptance"), 0.20, "{:.3f}"),
    ("markers per fix", ("fixes", "mean_markers_per_fix"), 0.8, "{:.2f}"),
    ("reprojection median (px)", ("fixes", "reprojection_rmse_px_median"), 0.6, "{:.2f}"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference", default="results/reference_metrics.json",
        help="the committed metrics to compare against",
    )
    parser.add_argument("--mission", default="missions/perimeter.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out", default="results/reproduce", help="where to put the re-flown log"
    )
    parser.add_argument(
        "--use-existing", action="store_true",
        help="compare against results/latest instead of re-flying",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="overwrite the committed reference with this run. Use when a change "
             "is intended to move the numbers.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    reference_path = REPO_ROOT / args.reference

    if args.use_existing:
        produced_path = REPO_ROOT / "results" / "latest" / "flight_metrics.json"
        if not produced_path.is_file():
            print(f"no metrics at {produced_path}; run `make demo` first", file=sys.stderr)
            return 2
    else:
        produced_path = _refly(args)
        if produced_path is None:
            return 2

    with open(produced_path, encoding="utf-8") as handle:
        produced = json.load(handle)

    if args.update or not reference_path.is_file():
        if not reference_path.is_file():
            print(f"no reference at {reference_path}, writing this run as the reference")
        _write_reference(reference_path, produced)
        return 0

    with open(reference_path, encoding="utf-8") as handle:
        reference = json.load(handle)

    return _compare(reference, produced, reference_path)


def _refly(args) -> Path:
    print(f"re-flying {args.mission} with seed {args.seed}")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_mission.py"),
        "--mission", args.mission,
        "--seed", str(args.seed),
        "--out", args.out,
        "--stem", "flight",
        "--quiet",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print("the flight itself failed", file=sys.stderr)
        return None
    return REPO_ROOT / args.out / "flight_metrics.json"


def _get(data, keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _write_reference(path: Path, produced) -> None:
    """Keep only the compared metrics, plus the provenance needed to read them."""
    payload = {
        "note": (
            "Headline metrics of the reference flight. Compared by "
            "scripts/check_reproducible.py within per-metric tolerances; see that "
            "script for why this is not a byte-for-byte comparison."
        ),
        "metadata": {
            key: produced.get("metadata", {}).get(key)
            for key in (
                "mission", "seed", "fault_schedule", "waypoints", "waypoints_reached",
                "final_state", "final_cause", "survey_method", "camera_source",
                "detector_backend", "control_rate_hz", "gains_file", "marker_count",
            )
        },
        "localisation": produced.get("localisation", {}),
        "fixes": produced.get("fixes", {}),
        "waypoints": produced.get("waypoints", {}),
        "trajectory_error": produced.get("trajectory_error", {}),
        "ground_truth": produced.get("ground_truth"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def _compare(reference, produced, reference_path: Path) -> int:
    print()
    header = f"{'metric':<28} {'committed':>12} {'produced':>12} {'moved':>10}  tolerance"
    print(header)
    print("-" * len(header))

    failures = []
    for name, keys, tolerance, fmt in CHECKS:
        want = _get(reference, keys)
        got = _get(produced, keys)
        if want is None or got is None:
            print(f"{name:<28} {'absent':>12} {'absent':>12}")
            continue

        if tolerance is None:
            same = want == got
            print(
                f"{name:<28} {str(want):>12} {str(got):>12} "
                f"{'same' if same else 'CHANGED':>10}  exact"
            )
            if not same:
                failures.append(f"{name}: committed {want}, produced {got}")
            continue

        moved = abs(float(got) - float(want))
        within = moved <= tolerance + 1e-12
        print(
            f"{name:<28} {fmt.format(float(want)):>12} {fmt.format(float(got)):>12} "
            f"{moved:>10.4f}  {tolerance:g}{'' if within else '  <-- OUTSIDE'}"
        )
        if not within:
            failures.append(
                f"{name}: committed {fmt.format(float(want))}, produced "
                f"{fmt.format(float(got))}, moved {moved:.4f} against a tolerance of {tolerance:g}"
            )

    print()
    if failures:
        print("the committed results are not reproducible:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            "\nEither the committed results are stale, in which case run\n"
            "  make demo && python scripts/check_reproducible.py --use-existing --update\n"
            "and commit the change, or something has stopped being reproducible.",
            file=sys.stderr,
        )
        return 1

    print(f"every metric is within tolerance of {reference_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
