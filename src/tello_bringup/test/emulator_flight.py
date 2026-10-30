"""Shared harness for the launch tests that fly the ROS 2 graph against the emulator.

Each test file launches ``emulator.launch.py`` (the Tello emulator plus
``hardware.launch.py`` pointed at 127.0.0.1), watches the graph's own topics from
a small observer node and, once the aircraft is down, compares what the graph
believed with the emulator's ground truth.

The ground truth is the emulator's CSV and nothing in the graph can read it, so
position errors measured here are honest about the estimator. They are also
simulated: the aircraft, the camera, the room and the radio link are all the
emulator's. Every summary this writes says so.

Output goes to ``$TELLO_RESULTS/emulator/<scenario>/`` (a temporary directory
when the variable is unset): ``truth.csv``, ``events.jsonl`` and
``summary.json``. Paths written into the summary are relative to that directory.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import launch_testing.actions
import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from tello_msgs.msg import FailsafeEvent, NavState, WaypointEvent

FLIGHT_TIMEOUT = 150.0
"""Seconds of wall clock allowed for a scenario to reach LANDED."""


def output_dir(scenario: str) -> Path:
    base = os.environ.get("TELLO_RESULTS") or tempfile.mkdtemp(prefix="tello-ros2-")
    path = Path(base) / "emulator" / scenario
    path.mkdir(parents=True, exist_ok=True)
    for stale in ("truth.csv", "events.jsonl", "summary.json"):
        (path / stale).unlink(missing_ok=True)
    return path


def describe(scenario: str, **arguments: str):
    """The launch description for one scenario and the context the tests get."""
    output = output_dir(scenario)
    launch_arguments = {
        "auto_arm": "true",
        "truth_log": str(output / "truth.csv"),
        "event_log": str(output / "events.jsonl"),
        **{key: str(value) for key, value in arguments.items()},
    }
    path = Path(get_package_share_directory("tello_bringup")) / "launch" / "emulator.launch.py"
    description = LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(path)),
                launch_arguments=launch_arguments.items(),
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )
    return description, {"output": output, "scenario": scenario}


class Observer(Node):
    """Listens to the graph the way an operator's laptop would."""

    def __init__(self) -> None:
        super().__init__("tello_test_observer")
        self.failsafe: List[dict] = []
        self.waypoints: List[dict] = []
        self.nav: List[dict] = []
        reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.create_subscription(FailsafeEvent, "failsafe_event", self._on_failsafe, reliable)
        self.create_subscription(WaypointEvent, "waypoint_event", self._on_waypoint, reliable)
        self.create_subscription(NavState, "nav_state", self._on_nav, reliable)

    def _on_failsafe(self, message: FailsafeEvent) -> None:
        self.failsafe.append(
            {
                "wall_time": time.time(),
                "from": message.from_state,
                "to": message.to_state,
                "cause": message.cause,
                "detail": message.detail,
            }
        )

    def _on_waypoint(self, message: WaypointEvent) -> None:
        self.waypoints.append(
            {
                "wall_time": _stamp(message.header.stamp),
                "index": int(message.index),
                "name": message.name,
                "kind": message.kind,
                "estimated_error": float(message.position_error),
            }
        )

    def _on_nav(self, message: NavState) -> None:
        self.nav.append(
            {
                "wall_time": _stamp(message.header.stamp),
                "position": [message.position.x, message.position.y, message.position.z],
                "localised": bool(message.localised),
            }
        )

    @property
    def landed(self) -> bool:
        return any(event["to"] == "LANDED" for event in self.failsafe)

    def causes(self) -> List[str]:
        return [event["cause"] for event in self.failsafe]

    def wait_for(self, predicate: Callable[[], bool], timeout: float) -> bool:
        import rclpy

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return predicate()

    def linger(self, seconds: float) -> None:
        import rclpy

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)


def _stamp(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


# ------------------------------------------------------------------ ground truth


def load_truth(path: Path) -> Dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as handle:
        rows = [row for row in csv.reader(line for line in handle if not line.startswith("#"))]
    header, body = rows[0], rows[1:]
    columns = {}
    for i, name in enumerate(header):
        values = [row[i] if i < len(row) else "" for row in body]
        columns[name] = np.array([float(v) if v != "" else np.nan for v in values])
    return columns


def truth_position(truth: Dict[str, np.ndarray], wall_time: float) -> np.ndarray:
    return np.array(
        [np.interp(wall_time, truth["wall_time"], truth[axis]) for axis in ("x", "y", "z")]
    )


def load_events(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def estimate_errors(observer: Observer, truth: Dict[str, np.ndarray]) -> np.ndarray:
    """3-D error of every nav_state against truth, while airborne above 0.3 m."""
    errors = []
    start, end = truth["wall_time"][0], truth["wall_time"][-1]
    for sample in observer.nav:
        t = sample["wall_time"]
        if not start <= t <= end:
            continue
        flying = np.interp(t, truth["wall_time"], truth["flying"]) > 0.99
        actual = truth_position(truth, t)
        if flying and actual[2] > 0.3:
            errors.append(float(np.linalg.norm(np.asarray(sample["position"]) - actual)))
    return np.asarray(errors)


def touchdown(truth: Dict[str, np.ndarray]) -> Optional[dict]:
    """Where and when the aircraft last went from flying to on the ground."""
    flying = truth["flying"] > 0.5
    changes = np.flatnonzero(flying[:-1] & ~flying[1:])
    if not len(changes):
        return None
    i = int(changes[-1]) + 1
    return {
        "wall_time": float(truth["wall_time"][i]),
        "flight_time": float(truth["flight_time"][i]),
        "position": [float(truth[a][i]) for a in ("x", "y", "z")],
        "battery": float(truth["battery"][i]),
    }


def write_summary(output: Path, scenario: str, summary: dict) -> dict:
    record = {
        "scenario": scenario,
        "ground_truth": "simulated (Tello emulator, tello_core.emulator)",
        "note": "Every number here comes from the emulator and the simulated room. "
        "None is a measurement of a DJI Tello.",
        "files": {"truth": "truth.csv", "events": "events.jsonl"},
        **summary,
    }
    with open(output / "summary.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=_jsonable)
        handle.write("\n")
    return record


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
