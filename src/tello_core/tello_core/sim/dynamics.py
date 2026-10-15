"""Flight dynamics for the simulated Tello.

A velocity-tracking model, not a rigid-body one and that is the right level of
fidelity for what this simulator is for. The Tello's own flight controller closes
an attitude and velocity loop the SDK never exposes; from outside, the aircraft
behaves as a first-order lag from stick input to body velocity, with a rate limit
and a disturbance. Modelling rotors and thrust would add parameters nobody can
measure through the SDK and would not change a single conclusion about the
navigation stack, which is what this repository is actually about.

What the model does include is everything that makes navigation hard:

* **First-order lag** on each velocity channel, so a step command does not
  produce a step velocity and the position loop has real phase lag to cope with;
* **Attitude coupling**, because the aircraft has to tilt to accelerate, which
  swings the camera exactly when the controller is asking for the largest
  correction;
* **Drift**, as an Ornstein-Uhlenbeck disturbance, standing in for the air
  currents in a real room and for the aircraft's own trim errors. Without it a
  position controller with no integral term looks perfect;
* **Accelerometer bias and noise**, so that dead reckoning between marker fixes
  degrades the way it does in the air;
* **Battery drain** at a rate that gives roughly the Tello's published endurance,
  because the failsafe logic is meaningless against a battery that never runs
  down;
* **Ground contact**, so that a landing is a state the aircraft reaches rather
  than a command it acknowledges.

Every stochastic element takes a seed. A simulated flight is reproducible from
its seed and its mission file alone, which is what lets ``results/`` be checked
in and CI verify it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import AircraftConfig
from ..geometry import euler_to_matrix, wrap_angle
from ..types import RcCommand, TelloState

GRAVITY = 9.80665


@dataclass(frozen=True)
class DynamicsSettings:
    """Model parameters.

    UNVERIFIED against hardware, all of them. The time constants are chosen so
    that the closed-loop step response matches the qualitative behaviour reported
    for the Tello, which is a much weaker statement than "measured". They are the
    single biggest reason the controller gains in this repository must be treated
    as a starting point for a real aircraft rather than as tuned values. See
    ``docs/PID_TUNING.md``, which sets out how much of the tuning survives a
    plausible range of these numbers.
    """

    horizontal_tau: float = 0.35
    """Seconds, first-order lag from stick to horizontal body velocity."""

    vertical_tau: float = 0.45
    yaw_tau: float = 0.25

    horizontal_rate_limit: float = 3.0
    """m/s^2 on the velocity setpoint, the aircraft's own acceleration limit."""

    vertical_rate_limit: float = 2.0
    yaw_rate_limit: float = np.deg2rad(300.0)

    drift_sigma: float = 0.05
    """m/s, stationary standard deviation of the horizontal disturbance."""

    drift_tau: float = 4.0
    """Seconds, correlation time of the disturbance. Long compared with the
    control loop, which is what makes it look like a slowly wandering trim error
    rather than like noise the controller can ignore."""

    accel_noise: float = 0.30
    """m/s^2, one axis, on the reported acceleration."""

    accel_bias_sigma: float = 0.12
    """m/s^2, one axis, of the constant bias drawn once per flight."""

    velocity_report_noise: float = 0.06
    """m/s on the aircraft's own reported body velocity."""

    tof_noise: float = 0.012
    """m on the downward time-of-flight sensor."""

    yaw_report_quantum: float = np.deg2rad(1.0)
    """The aircraft reports whole degrees."""

    takeoff_height: float = 1.0
    """m. Published as roughly 1 m for the ``takeoff`` command."""

    resting_height: float = 0.030
    """m. Height of the body origin above the floor when the aircraft is sitting
    on its landing gear. Small and not negligible: the camera sits below the body
    origin, so with a resting height of zero the camera would be below the floor
    plane and the pre-flight localisation check, which is the thing that decides
    whether take-off is allowed at all, would never pass. The Tello is 41 mm tall
    and its body origin is taken at the propeller plane."""

    takeoff_speed: float = 0.6
    land_speed: float = 0.35

    hover_current: float = 1.0 / 480.0
    """Battery fraction per second while flying, giving about eight minutes from
    full, which is close to the published endurance."""

    idle_current: float = 1.0 / 6000.0
    temperature_idle: float = 62.0
    temperature_rise: float = 22.0
    temperature_tau: float = 90.0


@dataclass
class DroneState:
    """Ground truth. The estimator never sees this; the evaluation does."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """World frame, m/s."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    yaw_rate: float = 0.0
    battery: float = 1.0
    temperature: float = 62.0
    flying: bool = False
    motors_cut: bool = False
    time: float = 0.0

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            yaw_rate=self.yaw_rate,
            battery=self.battery,
            temperature=self.temperature,
            flying=self.flying,
            motors_cut=self.motors_cut,
            time=self.time,
        )


class TelloDynamics:
    """The simulated aircraft."""

    def __init__(
        self,
        config: AircraftConfig,
        settings: Optional[DynamicsSettings] = None,
        start_position: Optional[np.ndarray] = None,
        start_yaw: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.settings = settings or DynamicsSettings()
        self._rng = np.random.default_rng(seed)
        self.state = DroneState(
            position=np.asarray(
                start_position
                if start_position is not None
                else [2.0, 1.5, (settings or DynamicsSettings()).resting_height],
                dtype=float,
            ),
            yaw=float(start_yaw),
        )
        self.accel_bias = self._rng.normal(0.0, self.settings.accel_bias_sigma, 3)
        self._drift = np.zeros(3)
        self._velocity_body = np.zeros(3)
        self._command = RcCommand.zero()
        self._takeoff_pending = False
        self._land_pending = False
        self._last_acceleration = np.zeros(3)

    # ----------------------------------------------------------------- commands

    def set_rc(self, command: RcCommand) -> None:
        self._command = command

    def takeoff(self) -> None:
        if not self.state.flying and not self.state.motors_cut:
            self._takeoff_pending = True
            self._land_pending = False

    def land(self) -> None:
        if self.state.flying:
            self._land_pending = True
            self._takeoff_pending = False

    def emergency(self) -> None:
        """Cut the motors. The aircraft then falls, which is the point."""
        self.state.motors_cut = True
        self._takeoff_pending = False
        self._land_pending = False

    # -------------------------------------------------------------- integration

    def step(self, dt: float) -> DroneState:
        """Advance by ``dt`` seconds and return the new ground-truth state."""
        dt = float(dt)
        if dt <= 0:
            return self.state.copy()

        settings = self.settings
        state = self.state
        state.time += dt

        if state.motors_cut:
            self._free_fall(dt)
            return state.copy()

        target_body, target_yaw_rate = self._targets()

        # Vertical automation. Take-off and landing are open-loop climbs and
        # descents in the real aircraft too: the SDK's `takeoff` does not accept
        # a height and does not report progress.
        if self._takeoff_pending:
            state.flying = True
            target_body = np.array([0.0, 0.0, settings.takeoff_speed])
            target_yaw_rate = 0.0
            if state.position[2] >= settings.takeoff_height:
                self._takeoff_pending = False
        elif self._land_pending:
            target_body = np.array([0.0, 0.0, -settings.land_speed])
            target_yaw_rate = 0.0

        if not state.flying:
            self._on_ground(dt)
            return state.copy()

        # First-order lag with a rate limit on the setpoint, per channel.
        horizontal = self._approach(
            self._velocity_body[:2],
            target_body[:2],
            settings.horizontal_tau,
            settings.horizontal_rate_limit,
            dt,
        )
        vertical = self._approach(
            self._velocity_body[2:],
            target_body[2:],
            settings.vertical_tau,
            settings.vertical_rate_limit,
            dt,
        )
        previous_body = self._velocity_body.copy()
        self._velocity_body = np.concatenate([horizontal, vertical])

        state.yaw_rate = float(
            self._approach(
                np.array([state.yaw_rate]),
                np.array([target_yaw_rate]),
                settings.yaw_tau,
                settings.yaw_rate_limit,
                dt,
            )[0]
        )
        state.yaw = wrap_angle(state.yaw + state.yaw_rate * dt)

        # Disturbance: Ornstein-Uhlenbeck in the world frame.
        decay = np.exp(-dt / settings.drift_tau)
        sigma = settings.drift_sigma * np.sqrt(max(1.0 - decay**2, 1e-9))
        self._drift = decay * self._drift + self._rng.normal(0.0, sigma, 3)
        self._drift[2] *= 0.4

        rotation = euler_to_matrix(state.yaw, state.pitch, state.roll)
        world_velocity = rotation @ self._velocity_body + self._drift
        acceleration = (self._velocity_body - previous_body) / dt

        state.position = state.position + world_velocity * dt
        state.velocity = world_velocity
        self._last_acceleration = rotation @ acceleration

        # The aircraft tilts to accelerate. Small-angle, which is exact enough
        # below the 20 degree limit and keeps the sign obvious: nose down to go
        # forward, left wing down to go left.
        state.pitch = float(
            np.clip(-acceleration[0] / GRAVITY, -self.config.max_tilt, self.config.max_tilt)
        )
        state.roll = float(
            np.clip(acceleration[1] / GRAVITY, -self.config.max_tilt, self.config.max_tilt)
        )

        if state.position[2] <= settings.resting_height:
            state.position[2] = settings.resting_height
            if self._land_pending or self._velocity_body[2] < 0:
                self._touch_down()

        state.battery = max(0.0, state.battery - settings.hover_current * dt)
        target_temperature = settings.temperature_idle + settings.temperature_rise
        state.temperature += (
            (target_temperature - state.temperature) * dt / settings.temperature_tau
        )
        if state.battery <= 0.0:
            # A flat battery is not a landing. The aircraft stops flying wherever
            # it is, which is exactly the outcome the failsafe exists to avoid.
            self.emergency()
        return state.copy()

    def _targets(self) -> tuple:
        rc = self.config.rc
        command = self._command
        return (
            np.array(
                [
                    command.forward_back / 100.0 * rc.horizontal_max,
                    command.left_right / 100.0 * rc.horizontal_max,
                    command.up_down / 100.0 * rc.vertical_max,
                ]
            ),
            command.yaw_rate / 100.0 * rc.yaw_rate_max,
        )

    @staticmethod
    def _approach(
        current: np.ndarray, target: np.ndarray, tau: float, rate_limit: float, dt: float
    ) -> np.ndarray:
        delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
        step = delta * (dt / max(tau, 1e-3))
        magnitude = float(np.linalg.norm(step))
        cap = rate_limit * dt
        if magnitude > cap and magnitude > 0:
            step = step * (cap / magnitude)
        return np.asarray(current, dtype=float) + step

    def _on_ground(self, dt: float) -> None:
        state = self.state
        state.velocity = np.zeros(3)
        self._velocity_body = np.zeros(3)
        state.position[2] = self.settings.resting_height
        state.pitch = 0.0
        state.roll = 0.0
        state.yaw_rate = 0.0
        state.battery = max(0.0, state.battery - self.settings.idle_current * dt)
        state.temperature += (
            (self.settings.temperature_idle - state.temperature)
            * dt
            / self.settings.temperature_tau
        )

    def _touch_down(self) -> None:
        self.state.flying = False
        self._land_pending = False
        self._velocity_body = np.zeros(3)
        self.state.velocity = np.zeros(3)
        self.state.yaw_rate = 0.0
        self._command = RcCommand.zero()

    def _free_fall(self, dt: float) -> None:
        state = self.state
        state.velocity = state.velocity + np.array([0.0, 0.0, -GRAVITY]) * dt
        state.position = state.position + state.velocity * dt
        self._last_acceleration = np.zeros(3)
        if state.position[2] <= self.settings.resting_height:
            state.position[2] = self.settings.resting_height
            state.velocity = np.zeros(3)
            state.flying = False

    # -------------------------------------------------------------- observation

    def telemetry(self, timestamp: float) -> TelloState:
        """Synthesise the aircraft's own view of itself.

        Deliberately *not* the ground truth with noise added to the position: the
        Tello does not know its position and neither does this. What it reports is
        a body velocity from its downward optical flow, a height from its
        time-of-flight sensor, an attitude from its own filter and a specific
        force from its accelerometer, all with the biases and quantisation of the
        real thing.
        """
        state = self.state
        rotation = euler_to_matrix(state.yaw, state.pitch, state.roll)

        specific_force = rotation.T @ (self._last_acceleration + np.array([0.0, 0.0, GRAVITY]))
        accel = (
            specific_force
            + self.accel_bias
            + self._rng.normal(0.0, self.settings.accel_noise, 3)
        )

        body_velocity = rotation.T @ state.velocity + self._rng.normal(
            0.0, self.settings.velocity_report_noise, 3
        )

        # The time-of-flight sensor looks down from the underside, so it reads
        # the gap to the floor, which is the body height minus the resting
        # height. On the ground that is zero, which is what makes `landed` a
        # thing the aircraft can report rather than something inferred.
        height = max(
            0.0,
            state.position[2]
            - self.settings.resting_height
            + self._rng.normal(0.0, self.settings.tof_noise),
        )
        quantum = self.settings.yaw_report_quantum
        reported_yaw = float(np.round(state.yaw / quantum) * quantum)

        return TelloState(
            timestamp=float(timestamp),
            accel=accel,
            gyro_z=0.0,
            attitude=np.array([reported_yaw, state.pitch, state.roll]),
            velocity_body=body_velocity,
            height=float(height),
            barometer=float(state.position[2]),
            battery=float(state.battery),
            temperature=float(state.temperature),
            valid=True,
        )
