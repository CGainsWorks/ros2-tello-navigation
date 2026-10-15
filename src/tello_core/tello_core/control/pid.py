"""A PID controller with the three details that matter on a real vehicle.

**Derivative on measurement, not on error.** A step change in the setpoint makes
the error jump and a derivative term acting on that jump produces an impulse on
the output. On this aircraft that is a full-scale stick deflection for one control
cycle every time a waypoint is reached, which is both visible in the trajectory
and unnecessary: the derivative term exists to damp the *vehicle's* motion and
the vehicle's motion is the measurement. Differentiating the measurement and
negating gives the damping without the kick.

**Anti-windup by back-calculation.** Clamping the output without telling the
integrator is the classic mistake: the integrator keeps accumulating while the
actuator is saturated and when the error finally reverses the controller spends
several seconds unwinding before it responds at all. Back-calculation feeds the
difference between the requested and the delivered output back into the
integrator, so it stops growing at exactly the point the actuator stops
responding. On a Tello this matters more than it might, because full stick is only
1 m/s and a controller asked to cross a 4 m room saturates immediately.

**Explicit ``dt``.** The controller does not read a clock. Every call takes the
interval since the last one, which is what makes it testable, makes a simulated
run reproducible and makes it behave correctly when the loop rate wobbles, which
on a wireless link it always does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PidGains:
    """Gains and limits for one axis."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float = float("inf")
    """Symmetric saturation on the output."""

    integral_limit: float = float("inf")
    """Hard clamp on the integrator's own contribution, in output units. Belt and
    braces alongside back-calculation: back-calculation stops the integrator
    growing when the *output* saturates and this stops it growing when a
    persistent small error would otherwise integrate to something large without
    ever saturating anything."""

    derivative_tau: float = 0.10
    """Seconds. First-order filter on the derivative term. The estimate this
    controller differentiates comes from a Kalman filter at 10 Hz and
    differentiating it raw amplifies the estimator's noise straight onto the
    sticks."""

    back_calculation: float = 1.0
    """Gain on the anti-windup feedback, as a fraction of ``ki``. One is the usual
    choice and means the integrator unwinds on roughly the same timescale it
    wound up on."""


@dataclass
class PidState:
    """Everything the controller carries between calls. Exposed so that a log can
    record it and a test can assert on it."""

    integral: float = 0.0
    derivative: float = 0.0
    last_measurement: Optional[float] = None
    last_output: float = 0.0
    last_unsaturated: float = 0.0
    saturated: bool = False


class Pid:
    """One axis."""

    def __init__(self, gains: PidGains) -> None:
        self.gains = gains
        self.state = PidState()

    def reset(self) -> None:
        """Forget the integrator and the derivative history.

        Called on every failsafe transition. A controller that keeps its
        integrator across a switch from navigating to returning will spend the
        first second of the return leg working off an error it accumulated
        somewhere else.
        """
        self.state = PidState()

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        gains = self.gains
        state = self.state
        if dt <= 0.0:
            return state.last_output

        error = float(setpoint) - float(measurement)

        if state.last_measurement is None:
            derivative = 0.0
        else:
            raw = (float(measurement) - state.last_measurement) / dt
            alpha = dt / max(gains.derivative_tau + dt, 1e-9)
            derivative = state.derivative + alpha * (raw - state.derivative)
        state.derivative = derivative
        state.last_measurement = float(measurement)

        integral_term = float(np.clip(state.integral, -gains.integral_limit, gains.integral_limit))
        unsaturated = gains.kp * error + integral_term - gains.kd * derivative
        output = float(np.clip(unsaturated, -gains.output_limit, gains.output_limit))

        if gains.ki != 0.0:
            # Back-calculation: the integrator sees the error it is meant to
            # remove, minus however much of the last request the actuator refused.
            windup = (output - unsaturated) * gains.back_calculation
            state.integral += (gains.ki * error + windup) * dt
            state.integral = float(
                np.clip(state.integral, -gains.integral_limit, gains.integral_limit)
            )

        state.last_unsaturated = unsaturated
        state.last_output = output
        state.saturated = abs(unsaturated - output) > 1e-9
        return output


@dataclass
class VectorPid:
    """Three independent :class:`Pid` axes, for a position or velocity loop.

    Independent per axis rather than a single gain on the error's magnitude. The
    axes of this problem genuinely are different: the aircraft accelerates
    horizontally by tilting and vertically by changing thrust, the two have
    different time constants and different limits and the vertical axis has a
    much better sensor behind it.
    """

    x: Pid
    y: Pid
    z: Pid

    @staticmethod
    def from_gains(
        horizontal: PidGains, vertical: PidGains
    ) -> "VectorPid":
        return VectorPid(x=Pid(horizontal), y=Pid(horizontal), z=Pid(vertical))

    def reset(self) -> None:
        self.x.reset()
        self.y.reset()
        self.z.reset()

    def update(self, setpoint: np.ndarray, measurement: np.ndarray, dt: float) -> np.ndarray:
        setpoint = np.asarray(setpoint, dtype=float).ravel()
        measurement = np.asarray(measurement, dtype=float).ravel()
        return np.array(
            [
                self.x.update(setpoint[0], measurement[0], dt),
                self.y.update(setpoint[1], measurement[1], dt),
                self.z.update(setpoint[2], measurement[2], dt),
            ]
        )

    def saturated(self) -> bool:
        return bool(self.x.state.saturated or self.y.state.saturated or self.z.state.saturated)

    def integrals(self) -> np.ndarray:
        return np.array([self.x.state.integral, self.y.state.integral, self.z.state.integral])


@dataclass
class ControlGains:
    """The whole controller's gains, loaded from
    ``src/tello_bringup/config/control.yaml``."""

    horizontal: PidGains = field(
        default_factory=lambda: PidGains(
            kp=1.10, ki=0.22, kd=0.42, output_limit=0.75, integral_limit=0.30
        )
    )
    vertical: PidGains = field(
        default_factory=lambda: PidGains(
            kp=1.30, ki=0.30, kd=0.30, output_limit=0.55, integral_limit=0.30
        )
    )
    yaw: PidGains = field(
        default_factory=lambda: PidGains(
            kp=1.60, ki=0.0, kd=0.18, output_limit=np.deg2rad(30.0)
        )
    )
    """No integral term: yaw has no steady disturbance to reject.

    The 30 deg/s output limit is the important number here and it is not a
    control decision at all. The camera's 1/60 s exposure turns a yaw rate into
    image blur of ``fx * omega * exposure`` pixels and the measured detection
    rate falls off a cliff at about eight pixels: 85 per cent at 30 deg/s, 35 per
    cent at 40, zero at 50. ``results/detection_yaw_rate.csv`` has the curve. The
    aircraft turning quickly to look at a marker is the aircraft blinding itself
    on the way to it and it cost this project its first autonomous flight."""

    @staticmethod
    def from_dict(data: dict) -> "ControlGains":
        def build(section: dict, fallback: PidGains) -> PidGains:
            if not section:
                return fallback
            return PidGains(
                kp=float(section.get("kp", fallback.kp)),
                ki=float(section.get("ki", fallback.ki)),
                kd=float(section.get("kd", fallback.kd)),
                output_limit=float(section.get("output_limit", fallback.output_limit)),
                integral_limit=float(section.get("integral_limit", fallback.integral_limit)),
                derivative_tau=float(section.get("derivative_tau", fallback.derivative_tau)),
                back_calculation=float(
                    section.get("back_calculation", fallback.back_calculation)
                ),
            )

        defaults = ControlGains()
        yaw_section = dict(data.get("yaw", {}) or {})
        if "output_limit_dps" in yaw_section:
            yaw_section["output_limit"] = float(np.deg2rad(yaw_section.pop("output_limit_dps")))
        return ControlGains(
            horizontal=build(data.get("horizontal", {}), defaults.horizontal),
            vertical=build(data.get("vertical", {}), defaults.vertical),
            yaw=build(yaw_section, defaults.yaw),
        )

    @staticmethod
    def from_yaml(path) -> "ControlGains":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return ControlGains.from_dict(data.get("gains", data))
