"""Flight logging: one wide CSV per flight, plus the event logs beside it.

A single flat table with one row per control cycle, rather than one file per
topic. The alternative, a file per signal, is tidier to write and much worse to
use: every question worth asking of a flight log is a question about two signals
at the same instant and joining separately-sampled files on a timestamp is how
plots end up quietly misaligned.

Ground truth is written into the same table when it exists, in columns prefixed
``truth_``. The flight software never sees those columns and cannot: they are
added by the recorder from the simulator's own state and on a real flight they
are simply absent. That is the only difference between a simulated log and a real
one, which is what lets the same evaluation code read both.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# The repository root, when this package is being used from a source tree. When
# it is installed, ``parents[3]`` is site-packages and nothing is relative to it,
# which ``provenance_path`` handles by leaving the path alone.
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def provenance_path(path: Optional[str]) -> Optional[str]:
    """How a source file is recorded in a results file.

    Repository-relative and POSIX-separated, because these strings are committed.
    An absolute Windows path in a committed JSON file is a provenance record that
    is wrong everywhere except the one machine that produced it: it makes the
    file look machine-specific in review, it breaks any check that compares the
    metadata across platforms and it quietly leaks a home directory.

    A path outside the repository is left exactly as it is. A map or a mission
    loaded from somewhere else really is somewhere else and shortening it would
    be a lie rather than a tidy-up.
    """
    if not path:
        return path
    resolved = Path(path).resolve()
    for root in (Path.cwd().resolve(), _SOURCE_ROOT):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(resolved)


@dataclass
class FlightRow:
    """One control cycle."""

    time_s: float
    state: str
    cause: str

    est_x: float
    est_y: float
    est_z: float
    est_yaw_deg: float
    est_vx: float
    est_vy: float
    est_vz: float
    est_sigma_m: float
    localised: int
    time_since_fix_s: float

    target_x: float = float("nan")
    target_y: float = float("nan")
    target_z: float = float("nan")
    target_yaw_deg: float = float("nan")
    waypoint_index: int = -1
    waypoint_name: str = ""

    rc_lr: int = 0
    rc_fb: int = 0
    rc_ud: int = 0
    rc_yaw: int = 0
    rc_sent: int = 1

    n_markers: int = 0
    marker_ids: str = ""
    fix_rmse_px: float = float("nan")
    fix_range_m: float = float("nan")
    fix_accepted: int = 0

    vo_valid: int = 0
    vo_tracks: int = 0
    vo_inlier_ratio: float = float("nan")

    battery: float = float("nan")
    temperature_c: float = float("nan")
    height_m: float = float("nan")
    link_state_age_s: float = float("nan")
    link_frame_age_s: float = float("nan")

    truth_x: float = float("nan")
    truth_y: float = float("nan")
    truth_z: float = float("nan")
    truth_yaw_deg: float = float("nan")
    truth_error_m: float = float("nan")


@dataclass
class FlightLog:
    """Rows, events and the metadata that makes them interpretable."""

    rows: List[FlightRow] = field(default_factory=list)
    failsafe_events: List[Dict[str, Any]] = field(default_factory=list)
    waypoint_events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def append(self, row: FlightRow) -> None:
        self.rows.append(row)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Columns as numpy arrays, for the metrics and the plots."""
        if not self.rows:
            return {}
        keys = list(asdict(self.rows[0]).keys())
        columns: Dict[str, np.ndarray] = {}
        for key in keys:
            values = [getattr(row, key) for row in self.rows]
            if isinstance(values[0], str):
                columns[key] = np.array(values, dtype=object)
            else:
                columns[key] = np.array(values, dtype=float)
        return columns

    # ------------------------------------------------------------------- output

    def write(self, directory, stem: str = "flight") -> Dict[str, Path]:
        """Write the trajectory, the two event logs and the metadata.

        Returns the paths written, so a caller can report them rather than
        guessing at them. Files are always written, even when a list is empty: an
        empty ``failsafe_log.csv`` with a header row says "nothing tripped" and a
        missing one says "this run did not get that far" and those are different.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        trajectory = directory / f"{stem}_trajectory.csv"
        fieldnames = list(asdict(self.rows[0]).keys()) if self.rows else _row_fieldnames()
        with open(trajectory, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(_round_row(asdict(row)))
        written["trajectory"] = trajectory

        failsafe = directory / f"{stem}_failsafe.csv"
        _write_dicts(
            failsafe,
            self.failsafe_events,
            ["time_s", "from_state", "to_state", "cause", "detail"],
        )
        written["failsafe"] = failsafe

        waypoints = directory / f"{stem}_waypoints.csv"
        _write_dicts(
            waypoints,
            self.waypoint_events,
            ["time_s", "index", "name", "kind", "position_error_m"],
        )
        written["waypoints"] = waypoints

        meta = directory / f"{stem}_meta.json"
        with open(meta, "w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True, default=_jsonable)
            handle.write("\n")
        written["metadata"] = meta
        return written

    @staticmethod
    def read(path) -> "FlightLog":
        """Read a trajectory CSV back. Events and metadata are loaded if they sit
        beside it under the same stem."""
        path = Path(path)
        rows: List[FlightRow] = []
        fields = {f for f in _row_fieldnames()}
        with open(path, "r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                kwargs: Dict[str, Any] = {}
                for key, value in raw.items():
                    if key not in fields:
                        continue
                    kwargs[key] = _coerce(key, value)
                rows.append(FlightRow(**kwargs))

        log = FlightLog(rows=rows)
        stem = path.name.replace("_trajectory.csv", "")
        directory = path.parent
        failsafe = directory / f"{stem}_failsafe.csv"
        if failsafe.is_file():
            with open(failsafe, "r", newline="", encoding="utf-8") as handle:
                log.failsafe_events = list(csv.DictReader(handle))
        waypoints = directory / f"{stem}_waypoints.csv"
        if waypoints.is_file():
            with open(waypoints, "r", newline="", encoding="utf-8") as handle:
                log.waypoint_events = list(csv.DictReader(handle))
        meta = directory / f"{stem}_meta.json"
        if meta.is_file():
            with open(meta, "r", encoding="utf-8") as handle:
                log.metadata = json.load(handle)
        return log


def _row_fieldnames() -> Sequence[str]:
    return list(FlightRow.__dataclass_fields__.keys())


_STRING_FIELDS = {"state", "cause", "waypoint_name", "marker_ids"}
_INT_FIELDS = {
    "localised",
    "waypoint_index",
    "rc_lr",
    "rc_fb",
    "rc_ud",
    "rc_yaw",
    "rc_sent",
    "n_markers",
    "fix_accepted",
    "vo_valid",
    "vo_tracks",
}


def _coerce(key: str, value: Optional[str]):
    if key in _STRING_FIELDS:
        return value or ""
    if value is None or value == "":
        return -1 if key in _INT_FIELDS else float("nan")
    if key in _INT_FIELDS:
        return int(float(value))
    return float(value)


def _round_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Round floats on the way out.

    Full double precision in a CSV is noise: it triples the file size and it makes
    the committed results churn on every platform whose libm rounds the last bit
    differently, which turns the reproducibility check in CI into a coin toss.
    Four decimals is a tenth of a millimetre.
    """
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = f"{value:.4f}" if np.isfinite(value) else ""
        else:
            out[key] = value
    return out


def _write_dicts(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.ravel()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
