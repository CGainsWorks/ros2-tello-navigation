"""The mission runner: the loop that actually flies the aircraft.

One loop, one thread, no callbacks. Every cycle it polls the link, updates the
estimator, asks the failsafe machine what is allowed, computes a command if
anything is, sends it and writes a log row. It runs identically against the
simulator and against a real Tello, because everything it touches is behind the
:class:`~tello_core.link.base.TelloLink` interface, including the clock.

Order within a cycle
--------------------
The order is not arbitrary and each step is where it is for a reason:

1. **Telemetry first.** The estimator's prediction step needs the newest inertial
   data before anything reads the estimate and the failsafe needs the newest
   battery and link age before it decides anything.
2. **Vision second.** A marker fix is a correction to a prediction, so it has to
   come after one. Its timestamp is older than the filter's clock by the
   detection and pose-solve time, which is exactly what the filter's out-of-order
   path is for.
3. **Failsafe third.** After the estimate is current and before any command is
   computed, so that a command is never computed from an estimate the failsafe
   has already decided not to trust.
4. **Command last** and only if the failsafe permits it.

Reversing steps three and four is the tempting simplification and it is how a
system ends up flying one more cycle on a failed estimate. One cycle at 0.8 m/s
is 4 cm, which does not sound like much until the reason the estimate failed is
that the aircraft is about to hit something.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from ..camera import CameraModel
from ..config import AircraftConfig
from ..geometry import euler_to_matrix, wrap_angle
from ..link.base import TelloLink
from ..localisation.fusion import Ekf, FusionSettings
from ..localisation.markers import MarkerMap
from ..localisation.multi import estimate_pose, marker_range
from ..localisation.qr import QrDetector, filter_known
from ..localisation.vo import VisualOdometry, VoSettings, plane_depth_from_marker
from ..logs import FlightLog, FlightRow, provenance_path
from ..types import (
    FailsafeCause,
    FlightState,
    MarkerFix,
    NavState,
    RcCommand,
    Waypoint,
    WaypointEvent,
)
from .failsafe import DescentProfile, FailsafeInputs, FailsafeMachine, FailsafeSettings
from .pid import ControlGains
from .waypoint import WaypointController, heading_for_markers, heading_is_adequate


class MissionError(ValueError):
    """Raised for a mission file that is not flyable as written."""


@dataclass
class Mission:
    """A parsed mission file."""

    name: str
    waypoints: List[Waypoint]
    description: str = ""
    takeoff_height: float = 1.10
    home: Optional[np.ndarray] = None
    """Where ``RETURN`` goes. Defaults to the launch position, which is what an
    operator means by home unless they say otherwise."""

    loops: int = 1
    source_path: Optional[str] = None

    @staticmethod
    def from_yaml(path) -> "Mission":
        path = Path(path)
        if not path.is_file():
            raise MissionError(f"mission file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise MissionError(f"{path}: top level must be a mapping")
        return Mission.from_dict(data, source_path=str(path))

    @staticmethod
    def from_dict(data: Dict[str, Any], source_path: Optional[str] = None) -> "Mission":
        entries = data.get("waypoints")
        if not entries:
            raise MissionError("mission contains no waypoints")

        tolerance = float(data.get("default_tolerance_m", 0.20))
        dwell = float(data.get("default_dwell_s", 1.0))
        timeout = float(data.get("default_timeout_s", 30.0))

        waypoints: List[Waypoint] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise MissionError(f"waypoint {index}: entry must be a mapping")
            position = np.asarray(entry.get("position", []), dtype=float).ravel()
            if position.size != 3 or not np.isfinite(position).all():
                raise MissionError(f"waypoint {index}: 'position' must be a finite [x, y, z]")
            raw_yaw = entry.get("yaw_deg", "auto")
            if raw_yaw in (None, "auto", "AUTO"):
                yaw = None
            else:
                yaw = float(np.deg2rad(float(raw_yaw)))
            waypoints.append(
                Waypoint(
                    name=str(entry.get("name", f"wp{index + 1}")),
                    position=position,
                    yaw=yaw,
                    tolerance=float(entry.get("tolerance_m", tolerance)),
                    dwell=float(entry.get("dwell_s", dwell)),
                    timeout=float(entry.get("timeout_s", timeout)),
                    metadata={k: v for k, v in entry.items() if k not in {
                        "name", "position", "yaw_deg", "tolerance_m", "dwell_s", "timeout_s"
                    }},
                )
            )

        home = data.get("home")
        return Mission(
            name=str(data.get("name", Path(source_path).stem if source_path else "mission")),
            description=str(data.get("description", "")),
            waypoints=waypoints,
            takeoff_height=float(data.get("takeoff_height_m", 1.10)),
            home=None if home is None else np.asarray(home, dtype=float).ravel(),
            loops=int(data.get("loops", 1)),
            source_path=source_path,
        )

    def validate(self, config: AircraftConfig, marker_map: MarkerMap) -> List[str]:
        """Return a list of problems. Empty means flyable.

        Checked before take-off by ``scripts/validate_missions.py`` and by the
        runner itself, because the two failures this catches, a waypoint outside
        the geofence and a waypoint from which no marker can be seen, are both
        certain to happen in the air and cheap to find on the ground.
        """
        problems: List[str] = []
        for index, waypoint in enumerate(self.waypoints):
            if not config.geofence.contains(waypoint.position):
                distance = config.geofence.breach_distance(waypoint.position)
                problems.append(
                    f"waypoint {index} ({waypoint.name}) is {distance:.2f} m outside the "
                    "geofence, so the failsafe would trip the moment it was reached"
                )
            if waypoint.tolerance <= 0:
                problems.append(f"waypoint {index} ({waypoint.name}) has a non-positive tolerance")
        if self.home is not None and not config.geofence.contains(self.home):
            problems.append("home position is outside the geofence")
        if not marker_map.markers:
            problems.append("marker map is empty")
        return problems


@dataclass
class MissionSettings:
    """How the runner runs, as opposed to where it goes."""

    control_rate: float = 20.0
    """Hz. Twice the telemetry rate, so that a command follows every packet
    promptly without the loop spinning on nothing."""

    detect_every: int = 1
    """Run detection on one frame in this many. One means every frame that
    arrives. Raising it is the first thing to try on a machine that cannot keep
    up and it costs localisation rate directly, so the value is recorded in the
    results metadata."""

    vo_enabled: bool = True
    max_flight_time: float = 300.0
    takeoff_settle: float = 2.0
    """Seconds after take-off before the position controller is allowed to
    command anything horizontal. The aircraft's own take-off is open loop and
    fighting it achieves nothing."""

    yaw_replan_interval: float = 0.4
    """Seconds between *checks* of whether the automatic heading is still working.

    Not between recomputations. The distinction is the whole point and it was
    learned the expensive way.

    An automatic heading chosen once, at the waypoint, is the heading that sees
    the most markers *from the waypoint* and halfway along the leg the aircraft
    is somewhere else. On this map that is often somewhere the chosen heading
    sees nothing and the reference mission lost localisation four times a flight
    because of it.

    The obvious fix, recomputing the best heading every cycle from the current
    position, is worse than the problem. It leaves the aircraft permanently
    turning: the best heading changes continuously as the aircraft moves, the
    controller chases it and a camera that is always turning is a camera that is
    always blurred. That version reached three waypoints out of five and spent 43
    per cent of the flight unlocalised, against zero lost fixes for the original.

    So the heading is held and only *checked*: if at least one marker is still
    in view from where the aircraft now is, nothing changes. If none is, a new
    heading is chosen. The aircraft turns when it has a reason to and not
    otherwise.
    """

    preflight_timeout: float = 12.0
    """Seconds allowed, on the ground, to get a first fix before the run is
    abandoned. The aircraft does not leave the ground until the filter is
    localised: an aircraft that takes off not knowing where it is has to find
    out while moving and the only tool it has for finding out is a camera that
    is now being shaken. Refusing to fly is the cheap failure."""

    record_truth: bool = True
    """Copy the simulator's ground truth into the log. Silently does nothing on a
    link that has none."""

    fusion: FusionSettings = field(default_factory=FusionSettings)
    vo: VoSettings = field(default_factory=VoSettings)
    failsafe: FailsafeSettings = field(default_factory=FailsafeSettings)
    gains: ControlGains = field(default_factory=ControlGains)


class MissionRunner:
    """Flies a :class:`Mission` over a :class:`~tello_core.link.base.TelloLink`."""

    def __init__(
        self,
        link: TelloLink,
        mission: Mission,
        marker_map: MarkerMap,
        config: AircraftConfig,
        camera: Optional[CameraModel] = None,
        settings: Optional[MissionSettings] = None,
    ) -> None:
        self.link = link
        self.mission = mission
        self.marker_map = marker_map
        self.config = config
        self.camera = camera or getattr(link, "camera", None) or CameraModel.tello_nominal()
        self.settings = settings or MissionSettings()

        self.detector = QrDetector()
        self.filter = Ekf(self.settings.fusion)
        self.vo = VisualOdometry(self.camera, self.settings.vo)
        self.controller = WaypointController(config, self.settings.gains)
        self.failsafe = FailsafeMachine(config, self.settings.failsafe)
        self.descent = DescentProfile(self.settings.failsafe)

        self.log = FlightLog()
        self.waypoint_events: List[WaypointEvent] = []
        self._index = 0
        self._loop = 0
        self._waypoint_started: Optional[float] = None
        self._frame_counter = 0
        self._start_time = 0.0
        self._home: Optional[np.ndarray] = None
        self._resolved_yaw: Dict[int, float] = {}
        self._yaw_replanned_at = float("-inf")
        self._last_state = None
        self._flight_state = None
        self._hold_position = None
        self._hold_yaw = None
        self._last_fix: Optional[MarkerFix] = None
        self._last_fix_accepted = False
        self._last_vo = None
        self._unknown_payloads: Dict[str, int] = {}
        self._preflight_seconds = float("nan")
        self._ever_airborne = False
        self._refused = None

    # --------------------------------------------------------------------- run

    def run(self) -> FlightLog:
        """Fly the mission to completion, a failsafe landing, or the time limit."""
        problems = self.mission.validate(self.config, self.marker_map)
        if problems:
            raise MissionError(
                "mission is not flyable:\n  " + "\n  ".join(problems)
            )

        if not self.link.connect():
            raise MissionError("could not connect to the aircraft")
        self.link.start_video()

        period = 1.0 / self.settings.control_rate
        self._start_time = self.link.now()
        wall_start = time.monotonic()

        # Do not leave the ground until the filter has an absolute fix. See
        # MissionSettings.preflight_timeout for why this is a hard gate rather
        # than a warning.
        if not self._preflight(period):
            self._abort_preflight()
            return self.log

        takeoff_requested = True
        previous = self.link.now()

        while True:
            now = self.link.now()
            dt = max(now - previous, 1e-3)
            previous = now
            elapsed = now - self._start_time

            telemetry = self._ingest_telemetry()
            self._ingest_video()

            estimate = self.filter.nav_state()
            if self._home is None and estimate.localised:
                if self.mission.home is not None:
                    self._home = np.asarray(self.mission.home, dtype=float)
                else:
                    # The launch point's horizontal position, at the mission's
                    # cruise height rather than at the floor. Home is where the
                    # aircraft flies *to* before landing and the launch point as
                    # recorded is on the ground, below the geofence floor. Using
                    # it directly sends the aircraft into a descent that the
                    # landing logic is not driving and that never terminates,
                    # which is what the first flight did for four minutes.
                    self._home = np.array(
                        [
                            estimate.position[0],
                            estimate.position[1],
                            max(self.mission.takeoff_height, self.config.geofence.min_corner[2]),
                        ]
                    )

            inputs = FailsafeInputs(
                now=now,
                link=self.link.status(),
                estimate=estimate,
                battery=self._last_state.battery if self._last_state else 1.0,
                temperature=self._last_state.temperature if self._last_state else float("nan"),
                height=self._last_state.height if self._last_state else 0.0,
                flight_time=elapsed,
                estimator_diverged=self.filter.has_diverged(),
                mission_complete=self._mission_complete(),
                takeoff_requested=takeoff_requested,
                airborne=self._airborne(),
                home_reached=self._home_reached(estimate),
            )
            decision = self.failsafe.update(inputs)
            takeoff_requested = False

            if decision.state is not self._flight_state:
                # Every failsafe transition resets the controller. An integrator
                # that has learned the room's air currents while cruising is not
                # the integrator a descent wants and carrying it across the
                # switch means the first second of the new state is spent working
                # off an error accumulated under different rules.
                self.controller.reset()
                self._hold_position = None
                self._hold_yaw = None
                self._flight_state = decision.state

            command, target, target_yaw, waypoint = self._command_for(
                decision.state, estimate, now, dt, elapsed
            )

            if decision.command_land:
                self.link.land()
            if decision.command_emergency:
                self.link.emergency()
            if decision.send_rc:
                self.link.send_rc(command)

            self._record(
                now=now,
                decision=decision,
                estimate=estimate,
                command=command,
                sent=decision.send_rc,
                target=target,
                target_yaw=target_yaw,
                waypoint=waypoint,
                telemetry=telemetry,
            )

            if decision.terminal:
                if self.failsafe.cause is not FailsafeCause.NONE and not self._ever_airborne:
                    # The failsafe refused the flight before it started. Recorded
                    # as an abort rather than as a completed flight, because a
                    # mission that never left the ground is not a mission that
                    # ended.
                    self._refused = self.failsafe.events[-1].detail
                break
            if elapsed > self.settings.max_flight_time:
                break
            if time.monotonic() - wall_start > 3600.0:
                # A wall-clock stop, independent of simulated time. Without it a
                # bug that stops the simulated clock advancing turns a test into
                # an infinite loop rather than a failure.
                break

            self.link.advance(period)

        self._finalise()
        return self.log

    def _preflight(self, period: float) -> bool:
        """Run the estimator on the ground until it is localised, or give up.

        Returns whether the aircraft may take off. Nothing is commanded here: the
        aircraft sits still, the camera looks at whatever it is pointed at and
        either a marker is in view or the mission does not happen.
        """
        deadline = self.link.now() + self.settings.preflight_timeout
        while self.link.now() < deadline:
            self._ingest_telemetry()
            self._ingest_video()
            # Both halves. A marker fix from the first video frame can satisfy the
            # filter before any telemetry has arrived and taking off then means
            # taking off without knowing the battery, the attitude or the height
            # and with the failsafe unable to tell a quiet link from a link that
            # has not started.
            link = self.link.status()
            fresh_telemetry = (
                link.packets_received > 0
                and link.last_state_age <= self.settings.failsafe.link_timeout
            )
            if self.filter.nav_state().localised and fresh_telemetry:
                self._preflight_seconds = self.link.now() - self._start_time
                return True
            self.link.advance(period)
        return False

    def _abort_preflight(self) -> None:
        """Record why the flight did not happen and close the link."""
        self.log.metadata = {
            "mission": self.mission.name,
            "mission_file": provenance_path(self.mission.source_path),
            "marker_map": provenance_path(self.marker_map.source_path),
            "final_state": FlightState.IDLE.value,
            "final_cause": FailsafeCause.LOCALISATION_LOST.value,
            "aborted": "no marker fix before take-off",
            "preflight_timeout_s": self.settings.preflight_timeout,
            "fusion": self.filter.diagnostics.as_dict(),
            "unknown_payloads": self._unknown_payloads,
        }
        if self._refused:
            # A flight the failsafe refused before take-off is an abort, not a
            # completed mission and the evaluation reads this field to tell the
            # two apart.
            self.log.metadata["aborted"] = self._refused
        self.link.close()

    # ----------------------------------------------------------------- ingestion

    def _ingest_telemetry(self):
        state = self.link.poll_state()
        if state is None:
            return None
        self._last_state = state
        self.filter.predict(state)
        return state

    def _ingest_video(self) -> None:
        frame = self.link.poll_frame()
        if frame is None:
            return
        self._frame_counter += 1
        if self._frame_counter % max(1, self.settings.detect_every) != 0:
            return

        observations = self.detector.detect(frame)
        known, unknown = filter_known(observations, self.marker_map)
        for payload in unknown:
            self._unknown_payloads[payload] = self._unknown_payloads.get(payload, 0) + 1

        prior = self.filter.position if self.filter.timestamp is not None else None
        fix = estimate_pose(
            known,
            self.marker_map,
            self.camera,
            self.config.mount.transform(),
            timestamp=frame.timestamp,
            prior_position=prior,
            prior_sigma=max(self.filter.position_sigma(), 0.05),
        )
        self._last_fix = fix
        self._last_fix_accepted = False
        if fix is not None:
            self._last_fix_accepted = self.filter.update_marker_fix(fix)

        if self.settings.vo_enabled:
            self._run_visual_odometry(frame, known)

    def _run_visual_odometry(self, frame, known) -> None:
        """Feed the front end, with depth taken from a marker when one is visible.

        The depth hint is the whole reason this front end has scale, so where it
        comes from is recorded. A marker in view gives a depth that is as good as
        the map; without one, the filter's own position against the nearest wall
        is used and that is worth less because it is the quantity the velocity is
        meant to be helping to estimate.
        """
        angular_velocity = np.array([0.0, 0.0, 0.0])
        if self._last_state is not None:
            angular_velocity = np.array([0.0, 0.0, float(self._last_state.gyro_z)])

        depth = 2.0
        normal = None
        source = "filter"
        if known:
            marker = self.marker_map.by_payload(known[0].decoded)
            if marker is not None:
                depth, normal = plane_depth_from_marker(
                    marker,
                    self.filter.position,
                    self.filter.yaw,
                    self.config.mount.transform(),
                )
                source = "marker"

        estimate = self.vo.process(
            frame,
            angular_velocity_body=angular_velocity,
            depth=depth,
            plane_normal_camera=normal,
            scale_source=source,
        )
        self._last_vo = estimate
        if estimate is not None:
            self.filter.update_visual_odometry(estimate)

    # ------------------------------------------------------------------ commands

    def _command_for(self, state, estimate: NavState, now: float, dt: float, elapsed: float):
        """Produce a stick command appropriate to the failsafe state."""
        if state in (FlightState.IDLE, FlightState.LANDED):
            self.controller.reset()
            return RcCommand.zero(), None, float("nan"), None

        if state is FlightState.TAKEOFF:
            self.link.takeoff()
            if elapsed < self.settings.takeoff_settle:
                return RcCommand.zero(), None, float("nan"), None
            target = np.array(
                [estimate.position[0], estimate.position[1], self.mission.takeoff_height]
            )
            command = self.controller.update(target, estimate.yaw, estimate, dt)
            return command, target, estimate.yaw, None

        if state in (FlightState.LAND, FlightState.EMERGENCY_LAND):
            # Vertical only. Holding position during a descent sounds better and
            # is worse: it keeps a horizontal command alive right up to touchdown
            # and a Tello that touches down with lateral velocity tips over.
            self.controller.reset()
            command = self.controller.to_rc(
                np.array([0.0, 0.0, self.descent.rate(state)]), 0.0
            )
            return command, None, float("nan"), None

        if state is FlightState.HOLD:
            # Hold the position the aircraft had when it lost the fix and turn
            # slowly on the spot looking for a marker.
            #
            # Both halves were wrong on the first attempt and both failures were
            # invisible in the code. Setting the target to the *current* estimate
            # every cycle means the position error is identically zero, so the
            # controller never corrects the drift it is supposed to be holding
            # against; and setting the yaw target to the current heading plus one
            # cycle's worth of search rate means the error is 0.3 degrees, which
            # is inside the stick deadband, so the aircraft does not turn at all.
            # It hovered, drifting, staring at a blank wall, until the hold timer
            # expired. Both targets have to be latched and accumulated.
            if self._hold_position is None:
                self._hold_position = estimate.position.copy()
                self._hold_yaw = float(estimate.yaw)
            self._hold_yaw = wrap_angle(
                self._hold_yaw + self.settings.failsafe.hold_search_rate * dt
            )
            command = self.controller.update(
                self._hold_position, self._hold_yaw, estimate, dt
            )
            return command, self._hold_position, self._hold_yaw, None

        if state is FlightState.RETURN:
            target = self._home if self._home is not None else estimate.position
            yaw = self._auto_yaw(target, estimate.yaw)
            command = self.controller.update(target, yaw, estimate, dt)
            return command, target, yaw, None

        # NAVIGATE
        waypoint = self._current_waypoint()
        if waypoint is None:
            self.controller.reset()
            return RcCommand.zero(), None, float("nan"), None

        if self._waypoint_started is None:
            self._waypoint_started = now
            self.controller.start_waypoint()

        yaw = waypoint.yaw
        if yaw is None:
            yaw = self._resolved_yaw.get(self._index)
            due = now - self._yaw_replanned_at >= self.settings.yaw_replan_interval
            if yaw is None or (due and not self._heading_still_works(estimate, yaw)):
                yaw = self._auto_yaw(estimate.position, estimate.yaw)
                self._resolved_yaw[self._index] = yaw
            if due:
                self._yaw_replanned_at = now

        command = self.controller.update(waypoint.position, yaw, estimate, dt)

        if self.controller.check_arrival(waypoint, estimate, now):
            self._advance_waypoint(now, estimate, waypoint, "arrived")
        elif now - self._waypoint_started > waypoint.timeout:
            self._advance_waypoint(now, estimate, waypoint, "timeout")

        return command, waypoint.position, yaw, waypoint

    def _auto_yaw(self, position: np.ndarray, current_yaw: float) -> float:
        return heading_for_markers(
            position, self.marker_map, self.config, self.camera, prefer=current_yaw
        )

    def _heading_still_works(self, estimate: NavState, yaw: float) -> bool:
        """Whether the heading being held is still good enough to hold.

        An absolute standard, not a comparison with the best heading available
        and that is the conclusion of four attempts rather than a first guess:

        * recompute the best heading every cycle and the aircraft turns
          continuously because the best heading moves with it. A camera that is
          always turning is always blurred: three waypoints out of five;
        * switch whenever a substantially better heading exists and it is the
          same failure with extra steps, because "substantially better" keeps
          being true: two waypoints out of five;
        * switch only when the view is completely empty and the aircraft flies
          the whole mission while holding a heading that sees a marker pair three
          metres away: five waypoints out of five but a fix on only a third of
          frames;
        * switch when the view falls below a standard set by the detector's own
          measured performance, which is this.

        See :func:`tello_core.control.waypoint.heading_is_adequate` for the two
        conditions and where they come from.
        """
        return heading_is_adequate(
            estimate.position, yaw, self.marker_map, self.config, self.camera
        )

    def _current_waypoint(self) -> Optional[Waypoint]:
        if self._index < len(self.mission.waypoints):
            return self.mission.waypoints[self._index]
        return None

    def _advance_waypoint(
        self, now: float, estimate: NavState, waypoint: Waypoint, kind: str
    ) -> None:
        error = float(np.linalg.norm(estimate.position - waypoint.position))
        self.waypoint_events.append(
            WaypointEvent(
                timestamp=now,
                index=self._index,
                name=waypoint.name,
                kind=kind,
                position_error=error,
            )
        )
        self._index += 1
        self._waypoint_started = None
        if self._index >= len(self.mission.waypoints):
            self._loop += 1
            if self._loop < self.mission.loops:
                self._index = 0
                self._resolved_yaw.clear()

    def _mission_complete(self) -> bool:
        return self._index >= len(self.mission.waypoints) and self._loop >= self.mission.loops

    def _home_reached(self, estimate: NavState) -> bool:
        """Whether the return leg has arrived. Tolerance is deliberately loose.

        A return is not a waypoint: nothing downstream depends on hitting it
        precisely and the reason the aircraft is returning at all is usually
        that something is running out. Half a metre horizontally and a quarter
        vertically is close enough to start the descent from.
        """
        if self._home is None:
            return False
        delta = np.asarray(estimate.position, dtype=float) - self._home
        return bool(np.linalg.norm(delta[:2]) < 0.5 and abs(delta[2]) < 0.25)

    def _airborne(self) -> bool:
        if self._last_state is None:
            return False
        airborne = bool(self._last_state.height > self.settings.failsafe.landed_height)
        self._ever_airborne = self._ever_airborne or airborne
        return airborne

    # ------------------------------------------------------------------- logging

    def _record(
        self,
        now: float,
        decision,
        estimate: NavState,
        command: RcCommand,
        sent: bool,
        target,
        target_yaw: float,
        waypoint: Optional[Waypoint],
        telemetry,
    ) -> None:
        status = self.link.status()
        fix = self._last_fix
        vo = self._last_vo
        row = FlightRow(
            time_s=now - self._start_time,
            state=decision.state.value,
            cause=decision.cause.value,
            est_x=float(estimate.position[0]),
            est_y=float(estimate.position[1]),
            est_z=float(estimate.position[2]),
            est_yaw_deg=float(np.rad2deg(estimate.yaw)),
            est_vx=float(estimate.velocity[0]),
            est_vy=float(estimate.velocity[1]),
            est_vz=float(estimate.velocity[2]),
            est_sigma_m=estimate.position_sigma(),
            localised=int(estimate.localised),
            time_since_fix_s=float(min(estimate.time_since_fix, 999.0)),
            target_x=float(target[0]) if target is not None else float("nan"),
            target_y=float(target[1]) if target is not None else float("nan"),
            target_z=float(target[2]) if target is not None else float("nan"),
            target_yaw_deg=float(np.rad2deg(target_yaw)) if np.isfinite(target_yaw) else float(
                "nan"
            ),
            waypoint_index=self._index if waypoint is not None else -1,
            waypoint_name=waypoint.name if waypoint is not None else "",
            rc_lr=command.left_right,
            rc_fb=command.forward_back,
            rc_ud=command.up_down,
            rc_yaw=command.yaw_rate,
            rc_sent=int(sent),
            n_markers=len(fix.marker_ids) if fix is not None else 0,
            marker_ids="|".join(fix.marker_ids) if fix is not None else "",
            fix_rmse_px=fix.reprojection_rmse if fix is not None else float("nan"),
            fix_range_m=marker_range(fix, self.marker_map) if fix is not None else float("nan"),
            fix_accepted=int(self._last_fix_accepted),
            vo_valid=int(bool(vo is not None and vo.valid)),
            vo_tracks=int(vo.n_tracks) if vo is not None else 0,
            vo_inlier_ratio=float(vo.inlier_ratio) if vo is not None else float("nan"),
            battery=float(telemetry.battery) if telemetry is not None else float("nan"),
            temperature_c=float(telemetry.temperature) if telemetry is not None else float("nan"),
            height_m=float(telemetry.height) if telemetry is not None else float("nan"),
            link_state_age_s=float(min(status.last_state_age, 999.0)),
            link_frame_age_s=float(min(status.last_frame_age, 999.0)),
        )

        if self.settings.record_truth and hasattr(self.link, "truth"):
            truth = self.link.truth()
            row.truth_x = float(truth.position[0])
            row.truth_y = float(truth.position[1])
            row.truth_z = float(truth.position[2])
            row.truth_yaw_deg = float(np.rad2deg(truth.yaw))
            row.truth_error_m = float(np.linalg.norm(estimate.position - truth.position))

        self.log.append(row)
        # The fix is consumed by the row that reports it, so that a single fix
        # does not appear on every subsequent row until the next one arrives.
        self._last_fix = None
        self._last_fix_accepted = False
        self._last_vo = None

    def _finalise(self) -> None:
        self.log.failsafe_events = self.failsafe.event_rows()
        self.log.waypoint_events = [
            {
                "time_s": round(event.timestamp - self._start_time, 3),
                "index": event.index,
                "name": event.name,
                "kind": event.kind,
                "position_error_m": round(event.position_error, 4),
            }
            for event in self.waypoint_events
        ]
        self.log.metadata = {
            "mission": self.mission.name,
            "mission_file": provenance_path(self.mission.source_path),
            "waypoints": len(self.mission.waypoints),
            "loops": self.mission.loops,
            "marker_map": provenance_path(self.marker_map.source_path),
            "marker_count": len(self.marker_map),
            "survey_method": self.marker_map.survey.method,
            "survey_accuracy_m": self.marker_map.survey.stated_accuracy,
            "camera_source": self.camera.source,
            "camera_size": [self.camera.width, self.camera.height],
            "aircraft_config": provenance_path(self.config.source),
            "detector_backend": self.detector.backend,
            "detector_fallback_frames": int(self.detector.fallback_frames),
            "detect_every": self.settings.detect_every,
            "control_rate_hz": self.settings.control_rate,
            "vo_enabled": self.settings.vo_enabled,
            "final_state": self.failsafe.state.value,
            "final_cause": self.failsafe.cause.value,
            "preflight_seconds": self._preflight_seconds,
            "waypoints_reached": sum(
                1 for event in self.waypoint_events if event.kind == "arrived"
            ),
            "fusion": self.filter.diagnostics.as_dict(),
            "unknown_payloads": self._unknown_payloads,
        }
        if self._refused:
            # A flight the failsafe refused on the ground is an abort, not a
            # mission that ended. It reaches here rather than
            # ``_abort_preflight`` because the refusal happens *after* a
            # successful pre-flight: the aircraft localised itself, read its own
            # battery and only then declined to take off. Without this the log
            # is indistinguishable from a mission that flew and landed, because
            # both end in LANDED with zero waypoints reached.
            self.log.metadata["aborted"] = self._refused
        self.link.close()


def resolve_yaw(
    waypoint: Waypoint,
    marker_map: MarkerMap,
    config: AircraftConfig,
    camera: CameraModel,
    prefer: Optional[float] = None,
) -> float:
    """The heading a waypoint should be flown at.

    Its own if it has one, otherwise the one that sees the most marker. Exposed
    separately from the runner so that ``scripts/validate_missions.py`` can report
    the resolved headings and so that a mission author can see what the automatic
    choice would be before trusting it.
    """
    if waypoint.yaw is not None:
        return float(waypoint.yaw)
    return heading_for_markers(waypoint.position, marker_map, config, camera, prefer=prefer)


def mission_visibility(
    mission: Mission,
    marker_map: MarkerMap,
    config: AircraftConfig,
    camera: CameraModel,
) -> List[Dict[str, Any]]:
    """How many markers each waypoint can see, at its resolved heading.

    The single most useful thing to know about a mission before flying it. A
    waypoint that sees nothing will be flown blind and the failsafe will notice
    within two seconds and land, which is correct behaviour and a wasted flight.
    """
    from ..geometry import make_transform
    from ..localisation.markers import markers_in_view

    rows: List[Dict[str, Any]] = []
    previous = None
    for index, waypoint in enumerate(mission.waypoints):
        yaw = resolve_yaw(waypoint, marker_map, config, camera, prefer=previous)
        previous = yaw
        T_wc = make_transform(euler_to_matrix(yaw), waypoint.position) @ config.mount.transform()
        visible = markers_in_view(marker_map, T_wc, camera)
        rows.append(
            {
                "index": index,
                "name": waypoint.name,
                "position": [float(v) for v in waypoint.position],
                "yaw_deg": float(np.rad2deg(yaw)),
                "auto_yaw": waypoint.yaw is None,
                "visible_markers": [m.marker_id for m in visible],
                "n_visible": len(visible),
            }
        )
    return rows


def summarise(events: Sequence[WaypointEvent]) -> Dict[str, int]:
    """Counts by event kind, for a one-line mission summary."""
    counts: Dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return counts


__all__ = [
    "Mission",
    "MissionError",
    "MissionRunner",
    "MissionSettings",
    "mission_visibility",
    "resolve_yaw",
    "summarise",
    "FailsafeCause",
]
