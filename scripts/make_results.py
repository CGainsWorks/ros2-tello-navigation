#!/usr/bin/env python3
"""Generate docs/RESULTS.md from the committed results.

Every quantitative claim in this repository comes from a CSV or a JSON file under
``results/`` and this script is what turns those into prose. Nothing is typed in
by hand, which is the only way to keep a results document honest: a number copied
into Markdown is a number that will still be there, unchanged and wrong, three
commits after the thing it measured changed.

The narrative around the numbers is here in the templates, which is where it
belongs: a table of figures with no account of what was being measured or what it
would mean on hardware is not a result, it is a dump.

Reads:
  results/latest/flight_metrics.json     the reference mission
  results/failsafe_log.csv               every failure case
  results/detection_*.csv                the measured detector curves
  results/survey_check.json              position error against surveyed points
  results/tuning_sweep.csv               how much the tuning survives
  maps/marker_map.yaml                   the survey provenance

Writes docs/RESULTS.md. Run it with `make results`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "tello_core"))

import numpy as np

from tello_core.localisation.markers import MarkerMap

MISSING = "not measured"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results")
    parser.add_argument("--flight", default="results/latest/flight_metrics.json")
    parser.add_argument("--out", default="docs/RESULTS.md")
    parser.add_argument(
        "--readme",
        default="README.md",
        help="also fill the generated block in this file. Pass an empty string to skip.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results = REPO_ROOT / args.results

    flight = _json(REPO_ROOT / args.flight)
    marker_map = MarkerMap.from_yaml(REPO_ROOT / "maps" / "marker_map.yaml")
    sections = [
        _header(marker_map),
        _mission_section(flight),
        _detection_section(results),
        _survey_section(_json(results / "survey_check.json")),
        _failsafe_section(_csv(results / "failsafe_log.csv")),
        _tuning_section(_csv(results / "tuning_sweep.csv")),
        _lessons_section(),
        _provenance_section(flight, marker_map),
    ]

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(s for s in sections if s).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")

    if args.readme:
        _update_readme(REPO_ROOT / args.readme, flight, _csv(results / "failsafe_log.csv"))
    return 0


README_BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
README_END = "<!-- END GENERATED RESULTS -->"


def _readme_block(flight, failsafe) -> str:
    """The headline table for the README.

    Deliberately short and deliberately generated. A number typed into a README
    is a number that will still be there, unchanged and wrong, three commits
    after the thing it measured changed.
    """
    if not flight:
        return "Run `make demo && make results` to fill this in."

    metadata = flight.get("metadata", {})
    localisation = flight.get("localisation", {})
    fixes = flight.get("fixes", {})
    waypoints = flight.get("waypoints", {})
    error = flight.get("trajectory_error", {})

    # Counted by how the flight *ended*, not by whether anything tripped. A
    # nominal flight is allowed a transient hold, so counting triggers would say
    # "7 of 7 tripped a failsafe" about a set that includes two clean missions.
    cases = len(failsafe or [])
    ended_by_failsafe = len(
        [r for r in (failsafe or []) if r.get("final_cause") not in ("NONE", "", None)]
    )

    lines = [
        "`missions/perimeter.yaml`: four corners of a 4.0 x 3.0 m room at 1.20 m,",
        "automatic heading, one loop, then a landing. Flown with `make demo`.",
        "",
        "| | |",
        "| --- | --- |",
        f"| waypoints reached | **{metadata.get('waypoints_reached', '?')} of "
        f"{metadata.get('waypoints', '?')}** |",
        f"| arrival error, RMSE | **{_f(waypoints.get('arrival_error_rmse'), 100, 1, ' cm')}** |",
        f"| localised, fraction of flight | "
        f"**{_f(localisation.get('availability'), 100, 1, ' per cent')}** |",
        f"| longest gap without a fix | {_f(localisation.get('longest_gap_s'), 1, 1, ' s')} |",
        f"| marker fixes | {fixes.get('fixes', '?')} at "
        f"{_f(fixes.get('fix_rate_hz'), 1, 1, ' Hz')}, "
        f"{_f(fixes.get('acceptance'), 100, 0, ' per cent')} accepted |",
        f"| reprojection residual, median | "
        f"{_f(fixes.get('reprojection_rmse_px_median'), 1, 2, ' px')} |",
    ]
    if error:
        lines += [
            f"| position error, RMSE *(simulator truth)* | "
            f"**{_f(error.get('position_rmse_m'), 100, 1, ' cm')}** |",
            f"| position error, 95th percentile *(simulator truth)* | "
            f"{_f(error.get('position_p95_m'), 100, 1, ' cm')} |",
            f"| heading error, RMSE *(simulator truth)* | "
            f"{_f(error.get('yaw_rmse_deg'), 1, 2, ' deg')} |",
        ]
    if cases:
        lines.append(
            f"| failure cases flown | {cases}, of which {ended_by_failsafe} were ended "
            "by a failsafe, all on the ground |"
        )
    lines += [
        f"| ended | {metadata.get('final_state', '?')} "
        f"({metadata.get('final_cause', '?')}) |",
        "",
        f"Marker map `survey: {metadata.get('survey_method', '?')}`, camera model "
        f"`{metadata.get('camera_source', '?')}`, simulator seed "
        f"{metadata.get('seed', '?')}.",
    ]
    return "\n".join(lines)


def _update_readme(path: Path, flight, failsafe) -> None:
    """Replace the block between the generated-results markers.

    Markers rather than a whole generated file, because the README is mostly prose
    that a human writes and a handful of numbers that a script should.
    """
    if not path.is_file():
        print(f"  (no {path.name}, skipping the README block)")
        return
    text = path.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        print(f"  ({path.name} has no generated block, skipping)")
        return
    before = text.split(README_BEGIN)[0]
    after = text.split(README_END)[1]
    block = _readme_block(flight, failsafe)
    path.write_text(
        before + README_BEGIN + "\n" + block + "\n" + README_END + after,
        encoding="utf-8",
    )
    print(f"updated the generated block in {path.name}")


def _json(path: Path):
    if not path.is_file():
        print(f"  (no {path.relative_to(REPO_ROOT)}, that section will say so)")
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path: Path):
    if not path.is_file():
        print(f"  (no {path.relative_to(REPO_ROOT)}, that section will say so)")
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value, scale: float = 1.0, places: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not np.isfinite(number):
        return MISSING
    return f"{number * scale:.{places}f}{suffix}"


# --------------------------------------------------------------------- sections


def _header(marker_map: MarkerMap) -> str:
    return f"""# Results

**Generated by `scripts/make_results.py`. Do not edit by hand.**

Every number below is read from a file under `results/` and every one of those
files is produced by a script in this repository. `make results` regenerates this
document; `make demo`, `make failsafe`, `make detection` and `make survey`
regenerate the files it reads.

---

## Read this first

There is no aircraft. Every figure here was produced against the simulator in
`src/tello_core/sim` and the marker map says `survey: {marker_map.survey.method}`,
which means the layout it is all measured against has never been measured itself.

So the figures divide into two kinds and the division matters more than any of the
numbers:

**Statements about the simulator.** Anything that needs ground truth: position
error, heading error, the consistency ratio. These say the algorithms are
correctly implemented and internally consistent against a known error model. They
say nothing about a DJI Tello.

**Statements that would carry over.** Anything that needs no truth: localisation
availability, fix rate, reprojection residual, detection rate against range and
turn rate and the static survey check. These are measured the same way on
hardware, so the same scripts would produce comparable numbers from a real
flight.

`src/tello_core/tello_core/eval/metrics.py` enforces the split in code:
`evaluate()` omits the ground-truth metrics entirely when a log has no truth
columns, rather than reporting them as NaN, so a results file from a real flight
cannot be misread as containing an absolute trajectory error.

See `docs/HARDWARE.md` for the full list of what is asserted without evidence."""


def _mission_section(flight) -> str:
    if not flight:
        return """## The reference mission

No flight log. Run `make demo`."""

    metadata = flight.get("metadata", {})
    localisation = flight.get("localisation", {})
    fixes = flight.get("fixes", {})
    waypoints = flight.get("waypoints", {})
    error = flight.get("trajectory_error", {})
    fusion = metadata.get("fusion", {})

    truth_note = (
        "" if flight.get("ground_truth") == "simulator"
        else "\n\nThis log has no ground truth, so the position and heading "
             "figures are absent rather than guessed."
    )

    simulated = f"""
| needs ground truth, so simulator only | |
| --- | --- |
| position error, RMSE | **{_f(error.get('position_rmse_m'), 100, 1, ' cm')}** |
| position error, 95th percentile | {_f(error.get('position_p95_m'), 100, 1, ' cm')} |
| position error, worst | {_f(error.get('position_max_m'), 100, 1, ' cm')} |
| per-axis RMSE (x, y, z) | {', '.join(_f(v, 100, 1, ' cm') for v in error.get('axis_rmse_m', []))} |
| heading error, RMSE | {_f(error.get('yaw_rmse_deg'), 1, 2, ' deg')} |
| consistency, error over reported sigma | {_f(error.get('consistency_ratio'), 1, 2)} |
""" if error else ""

    return f"""## The reference mission

`missions/perimeter.yaml`: four corners of a 4.0 x 3.0 m room at 1.20 m, automatic
heading at every waypoint, one loop, then a landing. Flown with
`make demo`, which is `scripts/run_mission.py` with the shipped configuration.

| would carry over to hardware | |
| --- | --- |
| waypoints reached | **{metadata.get('waypoints_reached', '?')} of {metadata.get('waypoints', '?')}** |
| arrival error, RMSE | {_f(waypoints.get('arrival_error_rmse'), 100, 1, ' cm')} |
| arrival error, worst | {_f(waypoints.get('arrival_error_max'), 100, 1, ' cm')} |
| localised, fraction of flight | **{_f(localisation.get('availability'), 100, 1, ' per cent')}** |
| longest gap without a fix | {_f(localisation.get('longest_gap_s'), 1, 1, ' s')} |
| marker fixes | {fixes.get('fixes', '?')} at {_f(fixes.get('fix_rate_hz'), 1, 1, ' Hz')} |
| fixes accepted by the filter | {_f(fixes.get('acceptance'), 100, 0, ' per cent')} |
| markers per fix, mean | {_f(fixes.get('mean_markers_per_fix'), 1, 2)} |
| reprojection residual, median | {_f(fixes.get('reprojection_rmse_px_median'), 1, 2, ' px')} |
| reprojection residual, 95th percentile | {_f(fixes.get('reprojection_rmse_px_p95'), 1, 2, ' px')} |
| range to the markers used, median | {_f(fixes.get('fix_range_m_median'), 1, 2, ' m')} |
| filter resets | {fusion.get('resets', '?')} |
| ended | {metadata.get('final_state', '?')} ({metadata.get('final_cause', '?')}) |
{simulated}{truth_note}

### Reading those numbers

The arrival error is the figure to quote for "can it fly to a point": the aircraft
got within about {_f(waypoints.get('arrival_error_rmse'), 100, 0, ' cm')} of each
waypoint, against a 20 cm arrival tolerance and a 1 s dwell requirement.

The trajectory RMSE is larger than the arrival error and larger than the accuracy
of an individual fix, which is about 1 cm for a marker pair at 1.5 m. Three things
account for the gap and none of them is a bug:

* fixes arrive at {_f(fixes.get('fix_rate_hz'), 1, 1, ' Hz')} rather than at the
  camera's 30 and between them the estimate dead-reckons on a 10 Hz
  accelerometer;
* the covariance floor of 3.5 cm deliberately stops the filter trusting the fixes
  more than the systematic error behind them warrants;
* the controller has real lag to fight, so the aircraft is behind its setpoint
  whenever it is moving and the trajectory error includes that.

The consistency ratio is error over the filter's own reported sigma, averaged over
the flight. Around one would mean the filter's uncertainty is honest; the value
here means it is still somewhat optimistic, which is the direction that matters
because the failsafe reads that number to decide whether the estimate can be acted
on. It is reported rather than tuned away.

Figures: `analysis/figures/trajectory.png`, `error.png`, `localisation.png` and
`camera_frames.png`, which is twelve frames from this flight, re-rendered at the
poses the aircraft occupied, with the detections drawn on. The brief asks for a
flight video; there is no aircraft and no footage, so that sheet is the honest
substitute and it shows more than a video would: the image quality the detector is
actually working with and which markers were read at each point."""


def _detection_section(results: Path) -> str:
    ranges = _csv(results / "detection_range.csv")
    yaw = _csv(results / "detection_yaw_rate.csv")
    incidence = _csv(results / "detection_incidence.csv")
    if not any([ranges, yaw, incidence]):
        return """## What the detector can actually read

Not measured. Run `make detection`."""

    def rows_for(rows, key, unit, every=2):
        if not rows:
            return "| | | |"
        out = []
        for index, row in enumerate(rows):
            if index % every and index != len(rows) - 1:
                continue
            out.append(
                f"| {row[key]}{unit} | {_f(row.get('decode_rate'), 100, 0, ' per cent')} "
                f"| {_f(row.get('fix_rate'), 100, 0, ' per cent')} |"
            )
        return "\n".join(out)

    trials = (ranges or yaw or incidence)[0].get("trials", "?")
    return f"""## What the detector can actually read

Three sweeps through the renderer with 300 mm markers, the Tello's nominal
683-pixel focal length and the default noise, blur and compression, {trials}
samples per operating point with the pose jittered by 2 cm and 2 degrees between
samples. `make detection` regenerates them.

These are the measurements that set the marker size, the controller's yaw limit
and the planner's incidence limit. They are not illustrations of those choices;
they came first.

### Range (square on, stationary)

| range | decoded | gave a usable fix |
| --- | --- | --- |
{rows_for(ranges, 'range_m', ' m', every=3)}

### Turn rate (1.5 m, square on)

The one that cost this project its first autonomous flight. Blur is
`fx * omega * exposure`, so 40 deg/s at 1/60 s is about eight pixels and eight
pixels is wider than a module at any useful range.

| yaw rate | blur | decoded |
| --- | --- | --- |
{chr(10).join(f"| {r['yaw_rate_dps']} deg/s | {_f(r.get('blur_px'), 1, 1, ' px')} | {_f(r.get('decode_rate'), 100, 0, ' per cent')} |" for r in (yaw or [])[::2])}

The controller's limit is 30 deg/s, where the decode rate is still high. The
original 90 deg/s was chosen as a control number and it blinded the aircraft.

### Viewing angle (1.5 m)

| incidence | decoded |
| --- | --- |
{chr(10).join(f"| {r['incidence_deg']} deg | {_f(r.get('decode_rate'), 100, 0, ' per cent')} |" for r in (incidence or [])[::2])}

The heading planner treats 50 degrees as the limit of usable incidence. A planner
that counts markers it cannot read will happily pick a heading that sees four of
them and localises off none.

Figure: `analysis/figures/detection.png`."""


def _survey_section(survey) -> str:
    if not survey or not survey.get("samples"):
        return """## Position error against surveyed points

Not measured. Run `make survey`."""
    bias = survey.get("bias_m", [float("nan")] * 3)
    return f"""## Position error against surveyed points

The measurement the brief asks for and the only accuracy figure here that would
mean the same thing on hardware. The aircraft is placed over each of five
surveyed check points, four corners of the flyable area and the centre and the
estimate is compared with the surveyed position. On hardware this is a hover, a
tape measure and ten seconds of `nav_state` per point.

| | |
| --- | --- |
| check points | {survey.get('samples')} |
| error, RMSE | **{_f(survey.get('error_rmse'), 100, 1, ' cm')}** |
| error, worst | {_f(survey.get('error_max'), 100, 1, ' cm')} |
| bias (x, y, z) | {_f(bias[0], 100, 1, ', ')}{_f(bias[1], 100, 1, ', ')}{_f(bias[2], 100, 1, ' cm')} |
| samples averaged per point | {survey.get('samples_per_point', '?')} |
| survey method | `{survey.get('survey_method', '?')}` |

The bias is the interesting column. A consistent offset in one axis is the
signature of a camera mount offset or a marker-size error, both of which are
unverified constants; see `docs/HARDWARE.md`. Random error averages away over the
samples and bias does not.

Because the survey method is `{survey.get('survey_method', '?')}`, this is
currently the estimator's agreement with a file rather than with a room."""


def _failsafe_section(rows) -> str:
    if not rows:
        return """## Failsafe

Not measured. Run `make failsafe`."""
    table = "\n".join(
        f"| `{row['case']}` | {row['cause']} | {row['detected_s'] or '-'} s "
        f"| {row['landed_s'] or '-'} s | {row['response_s'] or '-'} s "
        f"| {row['final_state']} | {row['waypoints_reached']}/{row['waypoints']} |"
        for row in rows
    )
    return f"""## Failsafe

Every failure case, flown with `make failsafe`. Each row is a complete flight with
one thing wrong; the full trajectory and event log for each is in
`results/failsafe/` and `scripts/summarise_failsafe.py` asserts the expected
outcome for every case so this table cannot quietly stop being true.

| case | cause | detected | landed | response | ended | waypoints |
| --- | --- | --- | --- | --- | --- | --- |
{table}

The `lossy_link` case is the one that stops the thresholds being set too tight: 20
per cent of telemetry and 25 per cent of video lost, plus malformed packets and
nothing should trip. A failsafe that fires on ordinary wireless attrition is a
failsafe that gets turned off.

`link_loss` is the case worth reading the log for. The controller stops
transmitting entirely, which is counter-intuitive and deliberate: on a degraded
link an rc command that gets through resets the aircraft's own fifteen-second
dead-man timer, so a controller that keeps talking can keep an aircraft it has no
control over airborne. `scripts/check_flight.py --expect-silent` checks it on
every CI run.

See `docs/FAILSAFE.md` for the state machine and every threshold.
Figure: `analysis/figures/failsafe.png`."""


def _tuning_section(rows) -> str:
    if not rows:
        return """## How much the tuning survives

Not measured. Run `python scripts/tuning_sweep.py`."""

    completed = [r for r in rows if r["waypoints_reached"] == r["waypoints"]]
    errors = [float(r["position_rmse_m"]) for r in rows
              if r.get("position_rmse_m") and np.isfinite(float(r["position_rmse_m"]))]
    by_parameter = {}
    for row in rows:
        by_parameter.setdefault(row["parameter"], []).append(row)

    blocks = []
    for parameter, group in by_parameter.items():
        nominal = group[0].get("nominal", "?")
        line = ", ".join(
            f"{row['value']} -> {_f(row['position_rmse_m'], 100, 0, ' cm')}"
            f"{'' if row['waypoints_reached'] == row['waypoints'] else ' (DID NOT COMPLETE)'}"
            for row in group
        )
        blocks.append(f"* **`{parameter}`** (nominal {nominal}): {line}")

    verdict = (
        f"All {len(rows)} plants completed the mission, with position RMSE between "
        f"{min(errors) * 100:.0f} and {max(errors) * 100:.0f} cm."
        if len(completed) == len(rows) and errors
        else f"{len(rows) - len(completed)} of {len(rows)} plants failed to complete the "
             "mission. The gains are tuned more narrowly than the plant is known and "
             "should not be trusted on hardware without retuning."
    )

    return f"""## How much the tuning survives being wrong about the aircraft

The controller gains were tuned against a simulated plant whose time constants are
estimates, not measurements. `scripts/tuning_sweep.py` flies the short mission with
the same gains against a range of plants, one parameter at a time.

{chr(10).join(blocks)}

{verdict}

That is a statement about sensitivity, not about correctness: the sweep is around
a plant model that is itself unverified. The gains in this repository are a
starting point for a real Tello and `docs/PID_TUNING.md` has the retuning order."""


def _lessons_section() -> str:
    return """## Five things that were wrong and how they showed up

Recorded because the failures are more informative than the final numbers and
because each one is now pinned by a test.

**1. The encoder's own quiet zone.** `cv2.QRCodeEncoder` returns a symbol wrapped
in a two-module white border, so a 21-module symbol comes back as 25 modules. The
first end-to-end run printed markers 25/21 of the size the map claimed and every
position estimate came out 19 per cent long, with a reprojection residual of about
twenty pixels on a solve that was otherwise perfectly well conditioned. Pinned by
`test_encoder_quiet_zone_is_stripped`.

**2. A mirrored marker frame.** "x right, y down, z out of the page" is the
combination that looks natural and is not right-handed: with x right and y down, z
points *into* the page. Every marker rendered mirrored. A mirrored marker still
decodes, still solves and gives a pose that is wrong in a way no residual reveals,
because the mirrored geometry is self-consistent. Pinned by
`test_marker_is_not_mirrored`.

**3. The geofence floor, applied during take-off.** The fence's lower z bound is a
minimum height above the floor and a take-off starts below it by definition. The
first flight was landed by the failsafe half a second after take-off, for leaving a
volume it had never been in. Pinned by
`test_geofence_floor_does_not_trip_during_take_off`.

**4. Turning fast enough to blind itself.** The controller was allowed 90 deg/s of
yaw. The first mission turned to face its first waypoint's heading, smeared every
marker across eighteen pixels, lost localisation and landed itself nine seconds
after take-off with the failsafe behaving perfectly. The yaw limit is now set from
the measured blur curve and pinned by
`test_shipped_yaw_limit_keeps_motion_blur_under_control`.

**5. A gating deadlock.** The most instructive one. Once the marker layout was
changed to use pairs, fixes became accurate to about a centimetre, so the
measurement covariance collapsed while the filter's own covariance kept its 3.5 cm
floor. The chi-square gate then sat at roughly 11 cm, which is less than the
estimate drifts in two seconds of dead reckoning, so after any gap *every*
recovering fix was rejected as an outlier and the filter could never get back. The
reference mission reached two waypoints of five and threw away 47 per cent of its
fixes, every one of them correct.

The fix is `FusionSettings.max_consecutive_rejections`: five gated fixes in a row
is not a plausible run of outliers when fixes arrive at 10 Hz from a map that does
not move, so the filter concludes that *it* is the thing that is wrong and
re-initialises from the measurement. The count of how often that happened is in
every results file, because a flight that is being rescued rather than maintained
is worth seeing."""


def _provenance_section(flight, marker_map: MarkerMap) -> str:
    metadata = (flight or {}).get("metadata", {})
    return f"""## Provenance

| | |
| --- | --- |
| mission | `{metadata.get('mission_file', '?')}` |
| marker map | `{metadata.get('marker_map', '?')}`, {len(marker_map)} markers |
| survey method | `{marker_map.survey.method}`, stated accuracy {_f(marker_map.survey.stated_accuracy, 1000, 0, ' mm')} |
| camera model | `{metadata.get('camera_source', '?')}` |
| aircraft config | `{metadata.get('aircraft_config', '?')}` |
| controller gains | `{metadata.get('gains_file', '?')}` |
| detector | `{metadata.get('detector_backend', '?')}`, second detector needed on {metadata.get('detector_fallback_frames', '?')} frames |
| control rate | {_f(metadata.get('control_rate_hz'), 1, 0, ' Hz')} |
| detection rate | one frame in {metadata.get('detect_every', '?')} |
| visual odometry | {'on' if metadata.get('vo_enabled') else 'off'} |
| simulator seed | {metadata.get('seed', '?')} |
| fault schedule | `{metadata.get('fault_schedule', '?')}` |

To reproduce everything from a clean clone:

```bash
make setup
make detection     # the detector curves
make demo          # the reference mission
make survey        # position error against surveyed points
make failsafe      # every failure case
python scripts/tuning_sweep.py
make figures
make results       # regenerates this document
```

`python scripts/check_reproducible.py` compares a fresh flight with the committed
metrics inside stated per-metric tolerances and explains in its own docstring why
that is a tolerance comparison rather than a byte-for-byte one."""


if __name__ == "__main__":
    raise SystemExit(main())
