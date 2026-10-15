"""The failsafe state machine. Written first and for a reason.

The brief for this project says failsafe logic first, autonomy second and do not
fly anything that cannot land itself. This module is that instruction made
concrete. Everything else here decides where the aircraft goes; this decides
whether it is allowed to go anywhere at all and what happens when it is not.

States
------
``IDLE`` on the ground and not commanded. ``TAKEOFF`` climbing to the first
waypoint's height. ``NAVIGATE`` flying the mission. ``HOLD`` stationary, waiting
for something to come back. ``RETURN`` flying to the launch point. ``LAND``
descending under control. ``EMERGENCY_LAND`` descending without a usable estimate.
``LANDED`` on the ground, motors stopped, terminal.

The machine is monotone in severity: once it has entered ``RETURN`` it will not go
back to ``NAVIGATE`` and once it has entered ``LAND`` it will not go back to
anything. Only ``HOLD`` is recoverable, because ``HOLD`` exists precisely for the
condition that might clear on its own. A machine that can recover from a landing
decision is a machine that can be talked out of landing by a sensor that flickers
and the whole value of a failsafe is that it cannot be.

Priorities
----------
Several conditions can be true at once. They are evaluated in a fixed order and
the first match wins and the order is by how little time is left rather than by
how bad it sounds:

1. ``ESTIMATOR_DIVERGED`` and ``CRITICAL_BATTERY``: seconds. Land immediately,
   wherever the aircraft is.
2. ``LINK_LOSS``: the aircraft is about to make its own decision. See below.
3. ``GEOFENCE_BREACH``: the aircraft is leaving the room.
4. ``OVERHEAT``: the aircraft will land itself soon.
5. ``LOW_BATTERY`` and ``MISSION_TIMEOUT``: tens of seconds. Return, then land.
6. ``LOCALISATION_LOST``: hold and look for a marker; land if it does not come
   back.
7. ``OPERATOR_ABORT``: whatever the operator asked for.

Link loss deserves its own paragraph
------------------------------------
A Tello lands itself fifteen seconds after the last command it hears. That is not
a failsafe this software can override and it is not one it should try to. So when
the link drops, this machine does two things and they are both slightly
counter-intuitive.

It commands ``land`` rather than a controlled descent and it **stops sending rc
commands entirely**. Continuing to send them is the instinctive thing to do and it
is actively harmful: on a link that is degraded rather than dead, an rc command
that gets through resets the aircraft's own dead-man timer, so a controller
faithfully transmitting into a half-broken link can keep an aircraft it has no
control over airborne indefinitely. Falling silent hands the aircraft to its own
failsafe, which is the only thing still able to act.

The cost is that the aircraft hovers, drifting, for up to fifteen seconds before
its own timer fires. In a 4 x 3 m room with 5 cm/s of drift that is well inside
the walls, which is what makes it acceptable here and would not make it acceptable
in a larger or windier space. It is in ``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import AircraftConfig
from ..link.base import LinkStatus
from ..types import FailsafeCause, FailsafeEvent, FlightState, NavState


@dataclass(frozen=True)
class FailsafeSettings:
    """Thresholds. Every one of them is justified in ``docs/FAILSAFE.md``."""

    link_timeout: float = 1.0
    """Seconds without telemetry before the link counts as lost. Ten expected
    packets at 10 Hz: long enough that ordinary loss on a 2.4 GHz link in a
    building does not trip it, short enough to leave most of the aircraft's own
    fifteen-second window for the response."""

    frame_timeout: float = 2.5
    """Seconds without video. Longer than the telemetry timeout because video is
    bulkier and drops out more readily and because losing video costs
    localisation rather than control."""

    localisation_timeout: float = 2.0
    """Seconds without a usable fix before the estimate stops being trusted."""

    hold_timeout: float = 6.0
    """Seconds allowed in ``HOLD`` to reacquire localisation before landing. The
    aircraft dead-reckons through this on a 10 Hz accelerometer and the measured
    drift rate says the estimate is worth roughly nothing beyond about eight
    seconds, so six is a margin rather than a limit."""

    hold_search_rate: float = np.deg2rad(35.0)
    """Yaw rate used while searching for a marker in ``HOLD``. Slow enough that
    motion blur does not defeat the detector, which was measured: at 90 deg/s the
    detection rate at 2 m falls by more than half."""

    overheat_temperature: float = 88.0
    """Degrees Celsius. The aircraft throttles and eventually lands itself above
    about 90."""

    geofence_grace: float = 0.15
    """Metres outside the fence tolerated before a breach is declared, to stop a
    noisy estimate on the boundary from tripping it."""

    geofence_hard: float = 0.50
    """Metres outside the fence at which returning is abandoned and the aircraft
    lands where it is. Beyond this the estimate that says it is outside is
    probably also wrong about which way is back."""

    mission_timeout: float = 240.0
    """Seconds of flight before the mission is abandoned regardless of progress."""

    land_descent_rate: float = 0.35
    """m/s commanded in ``LAND``."""

    emergency_descent_rate: float = 0.45
    """m/s commanded in ``EMERGENCY_LAND``. Faster, because there is no estimate
    worth loitering to improve."""

    landed_height: float = 0.12
    """Metres below which the aircraft is treated as down."""

    landed_dwell: float = 1.0
    """Seconds it must stay there. Without it a bounce reads as a landing."""


@dataclass
class FailsafeInputs:
    """Everything the machine looks at, gathered once per cycle."""

    now: float
    link: LinkStatus
    estimate: NavState
    battery: float
    temperature: float
    height: float
    flight_time: float
    estimator_diverged: bool = False
    operator_abort: bool = False
    mission_complete: bool = False
    takeoff_requested: bool = False
    airborne: bool = True
    home_reached: bool = False
    """True once the aircraft is at the return point. Supplied by the runner,
    which is the thing that knows where home is; the machine only needs to know
    that the return leg is over."""


@dataclass
class FailsafeDecision:
    """What the machine wants to happen this cycle.

    ``send_rc`` is the field that carries the link-loss behaviour described in the
    module docstring: when it is false the caller must not transmit a stick
    command, because staying silent is what hands the aircraft to its own
    fifteen-second timer.
    """

    state: FlightState
    cause: FailsafeCause = FailsafeCause.NONE
    send_rc: bool = True
    command_land: bool = False
    command_emergency: bool = False
    detail: str = ""

    @property
    def terminal(self) -> bool:
        return self.state is FlightState.LANDED


class FailsafeMachine:
    """The machine itself. One method, called once per control cycle."""

    # Severity order, used to enforce monotonicity.
    _RANK = {
        FlightState.IDLE: 0,
        FlightState.TAKEOFF: 1,
        FlightState.NAVIGATE: 2,
        FlightState.HOLD: 3,
        FlightState.RETURN: 4,
        FlightState.LAND: 5,
        FlightState.EMERGENCY_LAND: 6,
        FlightState.LANDED: 7,
    }

    def __init__(
        self,
        config: AircraftConfig,
        settings: Optional[FailsafeSettings] = None,
    ) -> None:
        self.config = config
        self.settings = settings or FailsafeSettings()
        self.state = FlightState.IDLE
        self.cause = FailsafeCause.NONE
        self.events: List[FailsafeEvent] = []
        self._hold_since: Optional[float] = None
        self._low_since: Optional[float] = None
        self._recoverable_from = FlightState.NAVIGATE
        self._landed_since: Optional[float] = None

    # ------------------------------------------------------------------- helpers

    def _transition(
        self, to_state: FlightState, cause: FailsafeCause, now: float, detail: str = ""
    ) -> None:
        if to_state is self.state:
            return
        self.events.append(
            FailsafeEvent(
                timestamp=float(now),
                from_state=self.state,
                to_state=to_state,
                cause=cause,
                detail=detail,
            )
        )
        self.state = to_state
        self.cause = cause

    def _escalate(
        self, to_state: FlightState, cause: FailsafeCause, now: float, detail: str = ""
    ) -> bool:
        """Move to ``to_state`` unless the machine is already at least as severe.

        Returns whether the state is now ``to_state`` or worse, so a caller can
        tell whether its condition is the one in charge.
        """
        if self._RANK[to_state] > self._RANK[self.state]:
            self._transition(to_state, cause, now, detail)
            return True
        return self._RANK[self.state] >= self._RANK[to_state]

    # -------------------------------------------------------------------- update

    def update(self, inputs: FailsafeInputs) -> FailsafeDecision:
        """Evaluate every condition and return what should happen."""
        now = float(inputs.now)
        settings = self.settings

        if self.state is FlightState.LANDED:
            return FailsafeDecision(
                state=FlightState.LANDED, cause=self.cause, send_rc=False
            )

        # Reaching the ground ends the flight from any state.
        if self._check_landed(inputs, now):
            return FailsafeDecision(
                state=FlightState.LANDED, cause=self.cause, send_rc=False
            )

        if self.state is FlightState.IDLE:
            if inputs.takeoff_requested:
                self._transition(FlightState.TAKEOFF, FailsafeCause.NONE, now, "commanded")
            else:
                return FailsafeDecision(state=self.state, cause=self.cause, send_rc=False)

        # Still on the ground. Anything that would cut a flight short is instead a
        # reason not to start one and the aircraft is already where a landing
        # would put it.
        #
        # This is not a special case bolted on; it is the correct handling of a
        # whole class of condition and getting it wrong was instructive. A
        # battery below the return threshold used to send the machine straight to
        # RETURN while the aircraft sat on the floor with its motors off. RETURN
        # does not command a take-off, so nothing moved, nothing landed and the
        # flight ran until the wall-clock limit expired ten minutes later. An
        # aircraft that cannot fly the mission should refuse it, not attempt to
        # fly home from the ground.
        if not inputs.airborne and self.state is FlightState.TAKEOFF:
            refusal = self._preflight_refusal(inputs)
            if refusal is not None:
                cause, detail = refusal
                self._transition(
                    FlightState.LANDED, cause, now, f"refused to take off: {detail}"
                )
                return FailsafeDecision(
                    state=FlightState.LANDED, cause=cause, send_rc=False
                )

        # 1. Nothing survives a diverged estimator or a critical battery.
        if inputs.estimator_diverged:
            self._escalate(
                FlightState.EMERGENCY_LAND,
                FailsafeCause.ESTIMATOR_DIVERGED,
                now,
                f"position sigma {inputs.estimate.position_sigma():.2f} m",
            )
            return self._decide(inputs)

        if inputs.battery <= self.config.battery.critical:
            self._escalate(
                FlightState.EMERGENCY_LAND,
                FailsafeCause.CRITICAL_BATTERY,
                now,
                f"battery {inputs.battery * 100:.0f} per cent",
            )
            return self._decide(inputs)

        # 1b. The return leg has arrived. Checked here, above every condition
        #     that can *cause* a return, because those conditions are all still
        #     true once the aircraft gets home: the battery is still low, the
        #     mission has still timed out. Testing it further down means the
        #     low-battery branch fires again every cycle and the aircraft hovers
        #     over the launch point until something else lands it.
        if self.state is FlightState.RETURN and inputs.home_reached:
            self._transition(FlightState.LAND, self.cause, now, "reached the return point")
            return self._decide(inputs)

        # 2. Link loss. The response is to fall silent and ask for a landing, so
        #    that the aircraft's own dead-man timer is free to act.
        if inputs.link.last_state_age > settings.link_timeout or not inputs.link.connected:
            self._escalate(
                FlightState.LAND,
                FailsafeCause.LINK_LOSS,
                now,
                f"no telemetry for {inputs.link.last_state_age:.1f} s",
            )
            return self._decide(inputs)

        # 3. Geofence. A small breach is recoverable by flying back; a large one
        #    means the estimate saying "outside" is not to be trusted about
        #    direction either.
        #
        #    Only meaningful when the aircraft is airborne and localised. On the
        #    ground it is outside the fence by construction, because the fence's
        #    floor is above the floor and an unlocalised estimate saying the
        #    aircraft is outside is not evidence that it is. Checking anyway was
        #    the first thing this machine did wrong: it landed an aircraft that
        #    had not taken off, for leaving a volume it was never in.
        breach = (
            self.config.geofence.breach_distance(
                inputs.estimate.position,
                ignore_floor=self.state
                in (
                    FlightState.TAKEOFF,
                    FlightState.LAND,
                    FlightState.EMERGENCY_LAND,
                ),
            )
            if (inputs.airborne and inputs.estimate.localised)
            else 0.0
        )
        if breach > settings.geofence_hard:
            self._escalate(
                FlightState.LAND,
                FailsafeCause.GEOFENCE_BREACH,
                now,
                f"{breach:.2f} m outside the fence",
            )
            return self._decide(inputs)
        if breach > settings.geofence_grace:
            self._escalate(
                FlightState.RETURN,
                FailsafeCause.GEOFENCE_BREACH,
                now,
                f"{breach:.2f} m outside the fence",
            )
            return self._decide(inputs)

        # 4. Overheat.
        if np.isfinite(inputs.temperature) and inputs.temperature >= settings.overheat_temperature:
            self._escalate(
                FlightState.RETURN,
                FailsafeCause.OVERHEAT,
                now,
                f"{inputs.temperature:.0f} C",
            )
            return self._decide(inputs)

        # 5. Battery and mission time: tens of seconds of margin, so return.
        if inputs.battery <= self.config.battery.low:
            if self._low_since is None:
                self._low_since = now
            reserve = self.config.battery.reserve_seconds
            remaining = self._estimate_endurance(inputs.battery)
            if remaining < reserve:
                self._escalate(
                    FlightState.LAND,
                    FailsafeCause.LOW_BATTERY,
                    now,
                    f"{remaining:.0f} s left, less than the {reserve:.0f} s reserve",
                )
            else:
                self._escalate(
                    FlightState.RETURN,
                    FailsafeCause.LOW_BATTERY,
                    now,
                    f"battery {inputs.battery * 100:.0f} per cent",
                )
            return self._decide(inputs)

        if inputs.flight_time > settings.mission_timeout:
            self._escalate(
                FlightState.RETURN,
                FailsafeCause.MISSION_TIMEOUT,
                now,
                f"{inputs.flight_time:.0f} s elapsed",
            )
            return self._decide(inputs)

        # 6. Operator abort. Below the automatic conditions because the operator
        #    asking for a landing while the battery is critical should still get
        #    the critical-battery response, which is the more urgent one.
        if inputs.operator_abort:
            self._escalate(FlightState.LAND, FailsafeCause.OPERATOR_ABORT, now, "operator")
            return self._decide(inputs)

        # 7. Localisation. The only recoverable condition, so it is the only one
        #    that uses HOLD.
        stale_video = inputs.link.last_frame_age > settings.frame_timeout
        lost = (not inputs.estimate.localised) or stale_video
        if lost and not inputs.airborne and self.state is FlightState.TAKEOFF:
            # Not yet off the ground and not localised. This is not a failure in
            # the air, it is a take-off that has not happened and the runner's
            # pre-flight check is what decides how long to wait for it.
            return self._decide(inputs)
        if lost:
            if self.state in (FlightState.NAVIGATE, FlightState.TAKEOFF):
                self._recoverable_from = self.state
                self._hold_since = now
                self._transition(
                    FlightState.HOLD,
                    FailsafeCause.LOCALISATION_LOST,
                    now,
                    "no video" if stale_video else
                    f"no fix for {inputs.estimate.time_since_fix:.1f} s",
                )
            elif self.state is FlightState.HOLD and self._hold_since is not None:
                if now - self._hold_since > settings.hold_timeout:
                    self._transition(
                        FlightState.LAND,
                        FailsafeCause.LOCALISATION_LOST,
                        now,
                        f"held {now - self._hold_since:.1f} s without reacquiring",
                    )
            return self._decide(inputs)

        if self.state is FlightState.HOLD:
            self._hold_since = None
            self._transition(
                self._recoverable_from,
                FailsafeCause.NONE,
                now,
                "localisation reacquired",
            )

        if self.state is FlightState.TAKEOFF and inputs.airborne and inputs.height > 0.5:
            self._transition(FlightState.NAVIGATE, FailsafeCause.NONE, now, "airborne")

        if inputs.mission_complete and self.state is FlightState.NAVIGATE:
            self._transition(FlightState.LAND, FailsafeCause.NONE, now, "mission complete")

        return self._decide(inputs)

    # ------------------------------------------------------------------ outputs

    def _decide(self, inputs: FailsafeInputs) -> FailsafeDecision:
        state = self.state
        if state is FlightState.LAND:
            # Silent on rc if the reason is link loss, for the reason in the
            # module docstring. Otherwise a controlled descent, which needs rc.
            silent = self.cause is FailsafeCause.LINK_LOSS
            return FailsafeDecision(
                state=state,
                cause=self.cause,
                send_rc=not silent,
                command_land=True,
                detail="silent descent" if silent else "controlled descent",
            )
        if state is FlightState.EMERGENCY_LAND:
            return FailsafeDecision(
                state=state,
                cause=self.cause,
                send_rc=True,
                command_land=True,
                detail="descending without a usable estimate",
            )
        return FailsafeDecision(state=state, cause=self.cause, send_rc=True)

    def _preflight_refusal(self, inputs: FailsafeInputs):
        """Reasons not to leave the ground, as ``(cause, detail)`` or ``None``.

        Deliberately the same conditions and the same thresholds the airborne
        ladder uses, read in the same order of urgency. A threshold that is worth
        landing for is worth not taking off for and keeping one list means the
        two cannot drift apart.

        Lost localisation is *not* here. On the ground that is a take-off that has
        not happened yet rather than a refusal and how long to wait for a first
        fix is the mission runner's decision; see
        ``MissionSettings.preflight_timeout``.
        """
        if inputs.estimator_diverged:
            return (
                FailsafeCause.ESTIMATOR_DIVERGED,
                f"position sigma {inputs.estimate.position_sigma():.2f} m",
            )
        if inputs.battery <= self.config.battery.critical:
            return (
                FailsafeCause.CRITICAL_BATTERY,
                f"battery {inputs.battery * 100:.0f} per cent",
            )
        if inputs.battery <= self.config.battery.low:
            return (
                FailsafeCause.LOW_BATTERY,
                f"battery {inputs.battery * 100:.0f} per cent, already below the "
                f"{self.config.battery.low * 100:.0f} per cent return threshold",
            )
        # A link that has gone quiet, not one that has not started. Before the
        # first packet the age is infinite and treating that as a link loss
        # refuses every flight on a lossy link: the pre-flight check can be
        # satisfied by a marker fix from the first video frame, which arrives
        # before any telemetry has been polled. How long to wait for the link to
        # come up is the runner's decision, the same as for the first fix.
        if inputs.link.packets_received > 0 and (
            not inputs.link.connected
            or inputs.link.last_state_age > self.settings.link_timeout
        ):
            return (
                FailsafeCause.LINK_LOSS,
                f"no telemetry for {inputs.link.last_state_age:.1f} s",
            )
        if (
            np.isfinite(inputs.temperature)
            and inputs.temperature >= self.settings.overheat_temperature
        ):
            return (FailsafeCause.OVERHEAT, f"{inputs.temperature:.0f} C")
        if inputs.operator_abort:
            return (FailsafeCause.OPERATOR_ABORT, "operator")
        return None

    def _check_landed(self, inputs: FailsafeInputs, now: float) -> bool:
        descending = self.state in (
            FlightState.LAND,
            FlightState.EMERGENCY_LAND,
        )
        if not descending:
            self._landed_since = None
            return False
        if inputs.height > self.settings.landed_height and inputs.airborne:
            self._landed_since = None
            return False
        if self._landed_since is None:
            self._landed_since = now
            return False
        if now - self._landed_since >= self.settings.landed_dwell:
            self._transition(FlightState.LANDED, self.cause, now, "on the ground")
            return True
        return False

    def _estimate_endurance(self, battery: float) -> float:
        """Seconds of flight left, from the published endurance.

        Crude on purpose. A battery model good enough to trust would need
        current, which the SDK does not expose and a curve fitted to a battery
        this project does not have. What this gives is a monotone decreasing
        number with roughly the right slope, which is all the reserve check needs
        in order to distinguish "enough to fly home" from "not".
        """
        full_flight_seconds = 480.0
        return float(max(0.0, battery) * full_flight_seconds)

    # --------------------------------------------------------------- reporting

    def event_rows(self) -> List[dict]:
        """The event log as plain dictionaries, for ``results/failsafe_log.csv``."""
        return [
            {
                "time_s": round(event.timestamp, 3),
                "from_state": event.from_state.value,
                "to_state": event.to_state.value,
                "cause": event.cause.value,
                "detail": event.detail,
            }
            for event in self.events
        ]


@dataclass
class DescentProfile:
    """The open-loop part of a landing.

    Used by ``LAND`` and ``EMERGENCY_LAND``. Horizontal channels are held at zero
    rather than at whatever the position controller last wanted, because a
    descent with a horizontal component is how an aircraft lands on its side.
    """

    settings: FailsafeSettings = field(default_factory=FailsafeSettings)

    def rate(self, state: FlightState) -> float:
        if state is FlightState.EMERGENCY_LAND:
            return -self.settings.emergency_descent_rate
        return -self.settings.land_descent_rate
