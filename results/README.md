# Committed results

Every number in `docs/RESULTS.md` and in the README's results block comes from a
file in this directory. Every one of these files is produced by a script in the
repository. Nothing here is edited by hand.

| file | produced by | what it is |
| --- | --- | --- |
| `latest/flight_*` | `make demo` | the reference mission: trajectory, failsafe and waypoint event logs, metadata, metrics |
| `reference_metrics.json` | `python scripts/check_reproducible.py --update` | the committed metrics a fresh flight is compared against |
| `failsafe/*` | `make failsafe` | one complete flight per failure case |
| `failsafe_log.csv` | `scripts/summarise_failsafe.py` | the failsafe table, one row per case |
| `detection_range.csv` | `make detection` | decode and fix rate against distance |
| `detection_yaw_rate.csv` | `make detection` | decode rate against turn rate. Sets the controller's yaw limit. |
| `detection_incidence.csv` | `make detection` | decode rate against viewing angle. Sets the planner's incidence limit. |
| `survey_check.csv`, `.json` | `make survey` | position error against surveyed check points |
| `tuning_sweep.csv` | `python scripts/tuning_sweep.py` | how much the controller tuning survives a wrong plant |

## Why these are committed

So that a reader can check a claim without running anything and so that a change
that moves a number shows up as a diff. `scripts/check_reproducible.py` compares a
fresh flight against `reference_metrics.json` inside per-metric tolerances. Its
docstring explains why that is a tolerance comparison rather than a
byte-for-byte one: OpenCV's decoder and sub-pixel refinement are not bit-stable
across versions. A check that fails on every run is a check nobody reads.

They are CSV and JSON, a few tens of kilobytes in total. The figures are *not*
committed as the source of truth: `make figures` regenerates them from these
files.

## Not committed

`results/latest/` is overwritten by every `make demo`, which is deliberate: it is
the current flight. `results/local/` is in `.gitignore` for scratch runs.
`results/ci/` is written by CI and uploaded as an artefact rather than committed.
