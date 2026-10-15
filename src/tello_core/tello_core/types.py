"""The data that moves between the driver, the estimator and the controller.

Plain dataclasses rather than ROS 2 messages, for one reason: the algorithms
have to be runnable, testable and reproducible with no ROS 2 installation. The
ROS 2 packages under ``src/tello_localisation`` and ``src/tello_control`` convert
these to and from the messages in ``src/tello_msgs`` at the node boundary and
nowhere else.

Every field carries SI units and the frame it is expressed in. The Tello's own
telemetry is in centimetres, decimetres per second and degrees; that conversion
happens once, in :mod:`tello_core.link.state` and no value in SI-unit form ever
travels alongside a value that is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TelloState:
    """One decoded telemetry packet, converted to SI units and REP-103 axes.

    The Tello broadcasts this at roughly 10 Hz on UDP port 8890. That rate is
    the hard constraint the whole estimator is designed around: 10 Hz is far too
    slow to integrate acceleration into position on its own, which is why the
    filter leans on the marker fixes for position and uses the accelerometer
    only to carry the estimate between them.
    """

    timestamp: float
    """Seconds, host monotonic clock at the moment the packet was received."""

    accel: np.ndarray
    """Specific force in the body frame, m/s^2, gravity *not* removed."""

    gyro_z: float
    """Yaw rate about the body z axis, rad/s. Differentiated from the reported
    yaw: the Tello does not publish a rate directly."""

    attitude: np.ndarray
    """``(yaw, pitch, roll)`` in radians, from the Tello's own attitude filter."""

    velocity_body: np.ndarray
    """``(vx, vy, vz)`` in the body frame, m/s, from the Tello's optical-flow
    and barometer estimate. Reliable in translation over a textured floor and
    close to useless over a plain one, which is why it enters the filter with a
    large covariance rather than as a direct measurement of truth."""

    height: float
    """Metres above the take-off surface, from the downward time-of-flight
    sensor. Valid roughly 0.1 m to 8 m and reported as its floor value outside
    that."""

    barometer: float
    """Metres, absolute barometric altitude. Drifts; used only as a fallback
    when the time-of-flight reading is out of range."""

    battery: float
    """Remaining charge, 0.0 to 1.0."""

    temperature: float
    """Degrees Celsius, mean of the two reported values. The Tello throttles
    and eventually lands itself above about 90 C."""

    valid: bool = True
    """False for a packet that arrived but failed to parse, so that a malformed
    packet is distinguishable from no packet at all."""


@dataclass(frozen=True)
class Frame:
    """One camera image with the host time it was received."""

    timestamp: float
    image: np.ndarray
    """``(H, W, 3)`` uint8, BGR, as OpenCV wants it."""

    index: int = -1
    """Sequence number within the stream, or -1 if unknown."""


@dataclass(frozen=True)
class MarkerObservation:
    """One QR marker seen in one frame.

    ``corners`` are in the order OpenCV's ``QRCodeDetector`` returns them, which
    is top-left, top-right, bottom-right, bottom-left of the QR symbol as
    printed. The marker map defines the same order in the marker's own frame, so
    the correspondence is fixed and no orientation search is needed.
    """

    marker_id: str
    corners: np.ndarray
    """``(4, 2)`` float, pixels."""

    timestamp: float
    decoded: str = ""
    """The raw payload string, kept because a marker whose payload does not
    match any map entry is a different failure from a marker that was not seen
    at all and the operator wants to know which."""


@dataclass(frozen=True)
class MarkerFix:
    """An absolute pose derived from one or more markers.

    This is the only measurement in the system that is not relative. Everything
    else, the accelerometer and the visual odometry alike, tells you how the
    state changed; only this tells you where you are.
    """

    timestamp: float
    position: np.ndarray
    """``(x, y, z)`` of the body origin in the world frame, metres."""

    yaw: float
    """Body heading in the world frame, radians."""

    marker_ids: tuple
    """Which map entries contributed, in the order they were used."""

    reprojection_rmse: float
    """Pixels. The single best quality signal available without ground truth
    and what the gate in :mod:`tello_core.localisation.multi` thresholds on."""

    position_covariance: np.ndarray
    """``(3, 3)``, m^2. Scaled from the reprojection residual and the marker
    geometry, see :func:`tello_core.localisation.multi.fix_covariance`."""

    yaw_variance: float
    """rad^2."""

    n_points: int = 0
    """Number of image points in the pose solve, four per marker."""


@dataclass(frozen=True)
class VoEstimate:
    """A body-frame velocity from the visual-odometry front end.

    Visual odometry from a single camera has no scale of its own. This front end
    takes its scale from the distance to the tracked plane, which comes from the
    filter's current position estimate and the marker map, so the estimate is
    only as good as that distance. ``scale_source`` records where it came from
    so that a velocity resting on a stale depth is identifiable afterwards.
    """

    timestamp: float
    velocity_body: np.ndarray
    """``(vx, vy, vz)`` m/s in the body frame."""

    n_tracks: int
    inlier_ratio: float
    scale_source: str
    """``"marker"``, ``"filter"`` or ``"height"``."""

    valid: bool = True


@dataclass
class NavState:
    """The fused estimate the controller flies on."""

    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    """World frame, m/s."""

    yaw: float
    accel_bias: np.ndarray
    gyro_bias: float
    covariance: np.ndarray
    """``(11, 11)`` over ``[p(3), v(3), yaw, accel_bias(3), gyro_bias]``, in that
    order, so the position block the controller and the failsafe read is the
    plain top-left 3x3 slice."""

    localised: bool = False
    """True while the filter has had a marker fix recently enough to trust its
    absolute position. The failsafe reads this and nothing else to decide
    whether localisation has been lost."""

    time_since_fix: float = float("inf")

    def position_sigma(self) -> float:
        """Scalar position uncertainty in metres: the **worst** of the three axes.

        The worst rather than the average, because this number is what the
        failsafe reads to decide whether the estimate can be acted on and a
        safety decision should be made on the axis that is least well known. The
        two are not close on this aircraft: the vertical is held by a
        time-of-flight sensor good to a few centimetres while the horizontal comes
        from markers alone, so an RMS over the three can sit comfortably inside
        the threshold while one horizontal axis has drifted half a metre.
        """
        var = np.clip(np.diag(self.covariance)[:3], 0.0, None)
        return float(np.sqrt(np.max(var)))


@dataclass(frozen=True)
class RcCommand:
    """A Tello ``rc a b c d`` command.

    All four channels are integers in ``[-100, 100]``. The Tello interprets them
    as a fraction of its maximum rate and the maximum depends on the speed
    setting, which is why :mod:`tello_core.control.waypoint` converts from m/s
    using a calibration constant rather than assuming 100 means one metre per
    second.

    Channel order follows the SDK: left-right, forward-back, up-down, yaw. The
    signs here are in the *body* frame of :mod:`tello_core.geometry`, so
    ``left_right`` is positive to the left and the driver flips it on the way
    out, because the SDK's own channel is positive to the right.
    """

    left_right: int = 0
    forward_back: int = 0
    up_down: int = 0
    yaw_rate: int = 0

    @staticmethod
    def zero() -> "RcCommand":
        return RcCommand(0, 0, 0, 0)

    def as_tuple(self) -> tuple:
        return (self.left_right, self.forward_back, self.up_down, self.yaw_rate)


class FailsafeCause(str, Enum):
    """Why the aircraft stopped doing what it was asked to do.

    These are the cause codes that appear in the trajectory log and in
    ``results/failsafe_log.csv``. They are strings rather than integers so that
    a log is readable without the enum to hand.
    """

    NONE = "NONE"
    LINK_LOSS = "LINK_LOSS"
    LOW_BATTERY = "LOW_BATTERY"
    CRITICAL_BATTERY = "CRITICAL_BATTERY"
    LOCALISATION_LOST = "LOCALISATION_LOST"
    LOCALISATION_DEGRADED = "LOCALISATION_DEGRADED"
    GEOFENCE_BREACH = "GEOFENCE_BREACH"
    MISSION_TIMEOUT = "MISSION_TIMEOUT"
    OPERATOR_ABORT = "OPERATOR_ABORT"
    OVERHEAT = "OVERHEAT"
    ESTIMATOR_DIVERGED = "ESTIMATOR_DIVERGED"


class FlightState(str, Enum):
    """States of the failsafe machine in :mod:`tello_core.control.failsafe`."""

    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    NAVIGATE = "NAVIGATE"
    HOLD = "HOLD"
    RETURN = "RETURN"
    LAND = "LAND"
    EMERGENCY_LAND = "EMERGENCY_LAND"
    LANDED = "LANDED"


@dataclass(frozen=True)
class FailsafeEvent:
    """A transition of the failsafe machine, with the reason."""

    timestamp: float
    from_state: FlightState
    to_state: FlightState
    cause: FailsafeCause
    detail: str = ""


@dataclass(frozen=True)
class WaypointEvent:
    """A waypoint reached, timed out or abandoned."""

    timestamp: float
    index: int
    name: str
    kind: str
    """``"arrived"``, ``"timeout"`` or ``"abandoned"``."""

    position_error: float = float("nan")
    """Metres between the estimate and the waypoint at the moment of the
    event."""


@dataclass
class Waypoint:
    """One entry in a mission."""

    name: str
    position: np.ndarray
    yaw: Optional[float] = None
    """Radians. ``None`` means hold whatever heading gets the most markers into
    view, which is resolved by :func:`tello_core.control.mission.resolve_yaw`."""

    tolerance: float = 0.20
    """Arrival radius, metres."""

    dwell: float = 1.0
    """Seconds the estimate must stay inside the radius before the waypoint
    counts as reached. Without this a waypoint fires the moment a noisy estimate
    clips the sphere, which on a real flight means it fires early and the
    aircraft never actually gets there."""

    timeout: float = 30.0
    """Seconds before the waypoint is abandoned and the mission decides what to
    do about it."""

    metadata: dict = field(default_factory=dict)
