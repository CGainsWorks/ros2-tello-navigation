"""The waypoint controller: position error in, stick command out.

One loop, not two. The obvious architecture is a cascade, position to velocity to
attitude but the Tello already closes its own velocity loop inside the airframe
and the SDK's ``rc`` command *is* a velocity request. Putting a second velocity
controller on top of the aircraft's own means two integrators fighting over the
same actuator across a wireless link with 30 to 60 ms of delay, which is a
reliable recipe for a slow oscillation nobody can tune out. So the position loop
produces a velocity directly and the aircraft's own loop is left to do its job.

The damping comes from the derivative term acting on the estimated position,
which for a fixed setpoint is the negative of the estimated velocity: exactly the
velocity feedback a cascade would provide, without the second integrator.

Arrival
-------
A waypoint counts as reached when the estimate has stayed inside its tolerance
sphere for ``dwell`` seconds continuously. Both halves are load-bearing. The
sphere alone fires the moment a noisy estimate clips the boundary, which on a real
flight means the aircraft moves on while still 30 cm out and travelling; the dwell
alone never fires if the tolerance is tight enough that the estimate only touches
it. Together they mean what an operator means by "it got there".

Heading
-------
Yaw is not a free parameter on this aircraft. The only absolute measurement comes
from markers seen by a forward-facing camera, so where the aircraft points decides
whether it knows where it is. A waypoint may fix its own heading; when it does
not, :func:`heading_for_markers` picks the one that puts the most marker area in
view, which is the difference between a mission that localises continuously and
one that spends half its time staring at a blank wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import AircraftConfig
from ..geometry import euler_to_matrix, make_transform, wrap_angle
from ..localisation.markers import MarkerMap, markers_in_view, pixel_span
from ..types import NavState, RcCommand, Waypoint
from .pid import ControlGains, Pid, VectorPid

# How much more a heading with two markers in it is worth than the sum of their
# projected areas suggests. Two markers resolve the tilt that a single one leaves
# almost unobservable and the measured fix error falls from about 15 cm to under
# 3 cm, so the bonus is set from that ratio rather than picked.
MARKER_PAIR_BONUS = 2.5


@dataclass
class ControlDebug:
    """What the controller was thinking, for the trajectory log and the plots."""

    position_error: np.ndarray
    desired_velocity_world: np.ndarray
    desired_velocity_body: np.ndarray
    yaw_error: float
    yaw_rate_command: float
    saturated: bool
    integral: np.ndarray


class WaypointController:
    """Drives the aircraft to a position and a heading."""

    def __init__(
        self,
        config: AircraftConfig,
        gains: Optional[ControlGains] = None,
    ) -> None:
        self.config = config
        self.gains = gains or ControlGains()
        self.position_pid = VectorPid.from_gains(self.gains.horizontal, self.gains.vertical)
        self.yaw_pid = Pid(self.gains.yaw)
        self._inside_since: Optional[float] = None
        self._heading: Optional[float] = None
        self._last_yaw: Optional[float] = None
        self.last_debug: Optional[ControlDebug] = None

    def reset(self) -> None:
        self.position_pid.reset()
        self.yaw_pid.reset()
        self._inside_since = None
        self._heading = None
        self._last_yaw = None

    def _unwrap_yaw(self, yaw: float) -> float:
        """A continuous heading, accumulated from wrapped measurements."""
        if self._heading is None or self._last_yaw is None:
            self._heading = float(yaw)
        else:
            self._heading += wrap_angle(float(yaw) - self._last_yaw)
        self._last_yaw = float(yaw)
        return float(self._heading)

    # ------------------------------------------------------------------ command

    def update(
        self,
        target_position: np.ndarray,
        target_yaw: float,
        estimate: NavState,
        dt: float,
    ) -> RcCommand:
        """One control cycle."""
        target_position = np.asarray(target_position, dtype=float).ravel()
        position = np.asarray(estimate.position, dtype=float).ravel()

        desired_world = self.position_pid.update(target_position, position, dt)
        speed_limit = self.config.rc.horizontal_max
        horizontal = desired_world[:2]
        magnitude = float(np.linalg.norm(horizontal))
        if magnitude > speed_limit and magnitude > 0:
            # Scale the pair together rather than clipping each axis. Clipping
            # independently changes the direction of travel, which on a diagonal
            # leg means the aircraft leaves the straight line between waypoints
            # and comes back to it only as it slows down.
            desired_world = desired_world.copy()
            desired_world[:2] = horizontal * (speed_limit / magnitude)

        yaw_error = wrap_angle(float(target_yaw) - float(estimate.yaw))
        # The PID is given an unwrapped heading and a setpoint constructed from
        # it, rather than the wrapped error directly. Both feed it the same
        # proportional term but only this one gives its derivative-on-measurement
        # a continuous signal: the wrapped error jumps by 2 pi when the aircraft
        # crosses the wrap point and a derivative term differentiating that jump
        # produces a full-scale yaw command in the wrong direction.
        heading = self._unwrap_yaw(float(estimate.yaw))
        yaw_rate = self.yaw_pid.update(heading + yaw_error, heading, dt)

        rotation = euler_to_matrix(float(estimate.yaw))
        desired_body = rotation.T @ desired_world

        command = self.to_rc(desired_body, yaw_rate)
        self.last_debug = ControlDebug(
            position_error=target_position - position,
            desired_velocity_world=desired_world,
            desired_velocity_body=desired_body,
            yaw_error=yaw_error,
            yaw_rate_command=float(yaw_rate),
            saturated=self.position_pid.saturated(),
            integral=self.position_pid.integrals(),
        )
        return command

    def to_rc(self, velocity_body: np.ndarray, yaw_rate: float) -> RcCommand:
        """Convert a body-frame velocity and a yaw rate into stick counts.

        The deadband is applied after rounding, not before: a request that rounds
        to two counts is a request the aircraft will ignore and sending it anyway
        would leave the log full of commands that had no effect.
        """
        rc = self.config.rc
        velocity_body = np.asarray(velocity_body, dtype=float).ravel()

        def channel(value: float, scale: float) -> int:
            counts = int(round(100.0 * value / scale)) if scale > 0 else 0
            counts = int(np.clip(counts, -100, 100))
            return 0 if abs(counts) < rc.deadband else counts

        return RcCommand(
            left_right=channel(velocity_body[1], rc.horizontal_max),
            forward_back=channel(velocity_body[0], rc.horizontal_max),
            up_down=channel(velocity_body[2], rc.vertical_max),
            yaw_rate=channel(float(yaw_rate), rc.yaw_rate_max),
        )

    # ------------------------------------------------------------------ arrival

    def check_arrival(
        self, waypoint: Waypoint, estimate: NavState, now: float
    ) -> bool:
        """Update and query the dwell timer. Call once per cycle."""
        distance = float(
            np.linalg.norm(np.asarray(estimate.position) - np.asarray(waypoint.position))
        )
        if distance <= waypoint.tolerance:
            if self._inside_since is None:
                self._inside_since = float(now)
            return (float(now) - self._inside_since) >= waypoint.dwell
        self._inside_since = None
        return False

    def start_waypoint(self) -> None:
        """Reset the arrival timer without resetting the integrators.

        Deliberately separate from :meth:`reset`. Moving to the next waypoint
        should not throw away an integrator that has learned the room's air
        currents; entering a failsafe state should.
        """
        self._inside_since = None


def heading_score(
    position: np.ndarray,
    yaw: float,
    marker_map: MarkerMap,
    config: AircraftConfig,
    camera,
) -> float:
    """How good a view of the markers a pose gives, in arbitrary units.

    The sum of the squares of the projected marker spans, with a bonus when two
    or more are in view. Both parts are set from measurements rather than taste:
    pose information goes roughly as the square of the projected size and a
    marker *pair* resolves the tilt ambiguity that a single marker leaves, which
    is worth about a factor of five in position error.

    Exposed separately from :func:`heading_for_markers` so that the mission
    runner can ask "is the heading I am holding still a good one?" without
    recomputing the best, which is the difference between an aircraft that turns
    when it needs to and one that turns constantly.
    """
    T_wc = make_transform(
        euler_to_matrix(float(yaw)), np.asarray(position, dtype=float).ravel()
    ) @ config.mount.transform()
    visible = markers_in_view(marker_map, T_wc, camera)
    if not visible:
        return 0.0
    score = float(sum(pixel_span(m, T_wc, camera) ** 2 for m in visible))
    if len(visible) >= 2:
        score *= MARKER_PAIR_BONUS
    return score


def heading_for_markers(
    position: np.ndarray,
    marker_map: MarkerMap,
    config: AircraftConfig,
    camera,
    candidates: int = 72,
    prefer: Optional[float] = None,
) -> float:
    """The heading from ``position`` that sees the most usable marker.

    Scores each candidate heading by the sum of the *squares* of the projected
    marker spans. The square, not the span and the difference is not cosmetic:
    with a linear score the planner prefers a heading that sees four markers
    across the room at the edge of the detector's reach over one that sees two an
    arm's length away and the far four are the ones that stop decoding the moment
    the aircraft turns. Pose information goes roughly as the square of the
    projected size, so squaring is both the right weighting and the one that picks
    the near markers.

    Headings are searched on a fixed grid rather than optimised, because the
    objective is piecewise constant, as markers enter and leave the field of view
    and a gradient method would sit on a plateau.

    ``prefer`` biases the choice towards a heading, typically the current one.
    Without it the aircraft will happily spin 180 degrees between two waypoints
    for a marginally better view and the spin costs more in motion blur and
    time than the view is worth.
    """
    position = np.asarray(position, dtype=float).ravel()
    T_bc = config.mount.transform()
    del T_bc
    best_yaw = 0.0 if prefer is None else float(prefer)
    best_score = -np.inf
    for yaw in np.linspace(-np.pi, np.pi, candidates, endpoint=False):
        score = heading_score(position, float(yaw), marker_map, config, camera)
        if prefer is not None:
            # A mild preference for the heading already held, so the aircraft does
            # not spin most of the way round for a marginally better view. Scaled
            # to the score so it stays meaningful now that the score is quadratic.
            score *= 1.0 - 0.15 * abs(wrap_angle(float(yaw) - float(prefer))) / np.pi
        if score > best_score:
            best_score = score
            best_yaw = float(yaw)
    return best_yaw


def heading_is_adequate(
    position: np.ndarray,
    yaw: float,
    marker_map: MarkerMap,
    config: AircraftConfig,
    camera,
    min_markers: int = 2,
    max_range: float = 3.0,
) -> bool:
    """Whether a pose would produce fixes worth having, judged absolutely.

    Two conditions, both taken from measurement rather than preference:

    * at least ``min_markers`` markers in view, because a single square leaves
      its own tilt nearly unobservable and the resulting fix is good to about
      15 cm where a pair is good to under 3 cm;
    * the nearest of them within ``max_range``, because the measured decode rate
      is above 90 per cent inside 3 m and falls away beyond it and because fix
      accuracy degrades with range even when the decode succeeds.

    An absolute standard rather than a comparison with the best available
    heading. Comparing against the best means turning whenever something better
    exists, which on a moving aircraft is nearly always and a camera that is
    always turning is always blurred. Asking "is this good enough" rather than
    "is this the best" is what lets the aircraft hold a heading for a whole leg
    and still turn when the view genuinely runs out.
    """
    position = np.asarray(position, dtype=float).ravel()
    T_wc = make_transform(euler_to_matrix(float(yaw)), position) @ config.mount.transform()
    visible = markers_in_view(marker_map, T_wc, camera)
    if len(visible) < min_markers:
        return False
    nearest = min(float(np.linalg.norm(m.position - position)) for m in visible)
    return nearest <= max_range
