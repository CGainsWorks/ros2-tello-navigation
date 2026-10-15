"""The fusion filter: one absolute measurement, several relative ones.

An extended Kalman filter over eleven states::

    [ p(3)  v(3)  yaw  accel_bias(3)  gyro_bias ]

all in the world frame except the biases, which are in the body frame where the
sensors live.

Why these states and not others
-------------------------------
Roll and pitch are **not** estimated. The Tello runs its own attitude filter and
publishes roll and pitch that are well observed, because gravity is a large
constant vector that a stationary accelerometer sees directly and re-estimating
them here would mean fighting a filter that has better data than we do. Yaw is
different: nothing observes absolute heading on this aircraft. The magnetometer
is not exposed over the SDK and would be useless indoors regardless, so the
Tello's yaw is an integrated rate that drifts and the markers are the only thing
that pins it down. Hence yaw is a state and roll and pitch are inputs.

The accelerometer bias is a state because it has to be. Ten-hertz accelerometer
data integrated twice turns a 0.05 m/s^2 bias into 25 cm of position error in ten
seconds, which is longer than this aircraft routinely goes without seeing a
marker. Estimating the bias is what makes the dead-reckoning between fixes worth
anything.

Why not an error-state filter
-----------------------------
The usual argument for an error-state formulation is that a three-parameter
attitude error avoids the singularities and the normalisation constraint of a
full rotation state. Here the only attitude state is a scalar yaw, which has no
singularity and needs no constraint, only wrapping. An error-state filter would
be more machinery for the same answer.

Timing
------
Measurements do not arrive in order. The telemetry is 10 Hz over one UDP socket,
the video is 30 Hz over another and a marker fix is only available after
detection and the pose solve have run on a frame, which on a laptop is 15 to 30
milliseconds after that frame arrived. So a fix whose timestamp is in the past
relative to the filter is normal, not exceptional. It is handled by keeping a
short buffer of past states and re-running the prediction forward from the state
at the fix's timestamp, which is exact up to the buffer length and is what
``max_buffer_age`` bounds. Beyond that the fix is dropped, because applying an
out-of-date measurement at the current time is worse than not applying it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from ..geometry import euler_to_matrix, wrap_angle
from ..types import MarkerFix, NavState, TelloState, VoEstimate

GRAVITY = 9.80665

# State layout. Written out rather than left implicit because every slice in this
# module depends on it and an off-by-one here is the kind of bug that produces a
# filter that works well enough to be believed.
IP = slice(0, 3)
IV = slice(3, 6)
IYAW = 6
IBA = slice(7, 10)
IBG = 10
N_STATES = 11


@dataclass(frozen=True)
class FusionSettings:
    """Noise model and gating. Every figure is justified in ``docs/PID_TUNING.md``
    or, for the ones that are sensor properties rather than tuning, here."""

    accel_noise: float = 0.35
    """m/s^2 per sqrt(Hz), one axis. The Tello's reported acceleration is already
    filtered and quantised, so this is not a datasheet figure; it is the standard
    deviation of the residual after removing gravity while the aircraft sits
    still. ``scripts/estimate_noise.py`` measures it from a flight log, which
    works on a simulated one and will work on a real one."""

    gyro_noise: float = 0.02
    """rad/s. Differentiated from the reported yaw, so it carries the yaw
    quantisation of 0.1 degrees at 10 Hz, which alone is 0.017 rad/s."""

    accel_bias_walk: float = 0.02
    """m/s^2 per sqrt(s)."""

    gyro_bias_walk: float = 0.002
    """rad/s per sqrt(s)."""

    velocity_noise: float = 0.25
    """m/s, on the Tello's own body-frame velocity estimate. Deliberately large:
    that estimate comes from the downward optical-flow sensor and is good over a
    textured floor and poor over a plain one and the filter has no way to tell
    which it is looking at."""

    height_noise: float = 0.03
    """m, on the downward time-of-flight sensor inside its valid range."""

    height_valid_range: Tuple[float, float] = (0.15, 4.0)

    vo_noise: float = 0.15
    """m/s, on the visual-odometry velocity. Scaled up further by the front end's
    own inlier ratio, see :meth:`Ekf.update_visual_odometry`."""

    fix_gate_chi2: float = 16.27
    """Chi-square gate on the four-dimensional marker-fix innovation, at the
    99.7th percentile of chi2(4). Fixes beyond it are rejected and counted."""

    max_consecutive_rejections: int = 5
    """How many fixes in a row may be gated out before the filter concludes that
    it and not the fixes, is the thing that is wrong.

    This is the most important line in this file and it exists because of a
    deadlock that is easy to build and hard to see.

    The gate compares an innovation against ``HPH' + R``. Once the marker layout
    was changed to put markers in pairs, the fixes became good to about a
    centimetre, so ``R`` collapsed; and ``P`` has a floor of
    ``min_position_sigma``. The gate therefore sits at roughly three times that
    floor, about 11 cm. Dead reckoning on a 10 Hz accelerometer drifts further
    than that in a couple of seconds, so after any gap in the fixes the estimate
    is outside the gate, every recovering fix is rejected as an outlier and the
    filter never gets back. The reference mission reached two waypoints out of
    five with 47 per cent of its fixes thrown away, every one of them correct.

    Five consecutive rejections is not a plausible run of outliers when fixes
    arrive at 10 Hz from a map that does not move. It means the filter has lost
    track and the correct response is to believe the measurement and
    re-initialise from it rather than to keep defending a state that is wrong.
    ``FusionDiagnostics.resets`` counts how often that happens: a flight with
    several is a flight whose estimate was being rescued rather than maintained
    and that is worth seeing in the log."""

    max_buffer_age: float = 1.0
    """Seconds of history kept for out-of-order measurements."""

    fix_timeout: float = 2.0
    """Seconds without an accepted fix before ``localised`` goes false."""

    min_position_sigma: float = 0.035
    """Metres. A floor on the filter's own position standard deviation.

    Not a fudge, a correction for an assumption the filter makes and reality does
    not honour. A Kalman filter shrinks its covariance as the square root of the
    number of independent measurements and at 14 marker fixes a second it shrinks
    a long way. Those fixes are not independent: consecutive frames see the same
    marker at almost the same angle, so they share whatever error the survey, the
    camera calibration and the detector's corner bias are contributing and no
    amount of averaging removes a systematic error. Without the floor the filter
    reported 1.5 cm of uncertainty against a measured 5.8 cm of error and the
    failsafe, which reads that number to decide whether the estimate can be
    trusted, believed it.

    The value is the residual the system cannot average away: roughly the survey
    accuracy and the camera-mount uncertainty added in quadrature. Raising it
    makes the filter lean harder on new fixes; lowering it makes the failsafe
    credulous."""

    max_position_sigma: float = 0.45
    """Metres. Above this the filter reports itself unlocalised even if a fix
    arrived recently, because a fix that barely constrains anything is not the
    same as knowing where you are."""

    diverged_position_sigma: float = 2.0
    """Metres. Above this the estimate is not merely uncertain, it is useless
    and the failsafe treats it as an estimator failure rather than as lost
    localisation."""


@dataclass
class FusionDiagnostics:
    """Counters the mission log and the failsafe both read."""

    fixes_accepted: int = 0
    fixes_rejected: int = 0
    fixes_stale: int = 0
    vo_accepted: int = 0
    vo_rejected: int = 0
    predictions: int = 0
    resets: int = 0
    consecutive_rejections: int = 0
    last_fix_time: float = float("-inf")
    last_rejected_nis: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "fixes_accepted": self.fixes_accepted,
            "fixes_rejected": self.fixes_rejected,
            "fixes_stale": self.fixes_stale,
            "vo_accepted": self.vo_accepted,
            "vo_rejected": self.vo_rejected,
            "predictions": self.predictions,
            "resets": self.resets,
        }


@dataclass
class _Snapshot:
    """Enough to rewind the filter to a past instant and replay from it."""

    timestamp: float
    state: np.ndarray
    covariance: np.ndarray
    attitude: np.ndarray
    accel: np.ndarray
    gyro_z: float


class Ekf:
    """The fusion filter.

    Usage is deliberately plain: call :meth:`predict` with each telemetry packet,
    call the ``update_*`` methods with whatever else arrives and read
    :meth:`nav_state` whenever the controller needs an estimate. The filter does
    not own a thread and does not sleep.
    """

    def __init__(
        self,
        settings: Optional[FusionSettings] = None,
        initial_position: Optional[np.ndarray] = None,
        initial_yaw: float = 0.0,
        initial_position_sigma: float = 1.5,
    ) -> None:
        self.settings = settings or FusionSettings()
        self.state = np.zeros(N_STATES)
        if initial_position is not None:
            self.state[IP] = np.asarray(initial_position, dtype=float)
        self.state[IYAW] = float(initial_yaw)

        self.covariance = np.diag(
            np.concatenate(
                [
                    np.full(3, initial_position_sigma**2),
                    np.full(3, 0.5**2),
                    [np.deg2rad(45.0) ** 2],
                    np.full(3, 0.3**2),
                    [np.deg2rad(5.0) ** 2],
                ]
            )
        )
        self.timestamp: Optional[float] = None
        self.attitude = np.zeros(3)
        self.last_accel = np.array([0.0, 0.0, GRAVITY])
        self.last_gyro_z = 0.0
        self.diagnostics = FusionDiagnostics()
        self._history: Deque[_Snapshot] = deque()
        self._initialised_from_fix = initial_position is not None

    # ------------------------------------------------------------------ state

    @property
    def position(self) -> np.ndarray:
        return self.state[IP].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[IV].copy()

    @property
    def yaw(self) -> float:
        return float(self.state[IYAW])

    def rotation(self) -> np.ndarray:
        """Body-to-world rotation from the estimated yaw and the measured
        roll and pitch."""
        return euler_to_matrix(self.state[IYAW], self.attitude[1], self.attitude[2])

    def position_sigma(self) -> float:
        """The worst of the three position standard deviations, metres.

        See :meth:`tello_core.types.NavState.position_sigma` for why the worst
        axis rather than the average of the three.
        """
        return float(np.sqrt(np.max(np.clip(np.diag(self.covariance)[IP], 0.0, None))))

    def time_since_fix(self) -> float:
        if self.timestamp is None or not np.isfinite(self.diagnostics.last_fix_time):
            return float("inf")
        return float(self.timestamp - self.diagnostics.last_fix_time)

    def is_localised(self) -> bool:
        return bool(
            self._initialised_from_fix
            and self.time_since_fix() <= self.settings.fix_timeout
            and self.position_sigma() <= self.settings.max_position_sigma
        )

    def has_diverged(self) -> bool:
        return bool(
            not np.all(np.isfinite(self.state))
            or self.position_sigma() > self.settings.diverged_position_sigma
        )

    def nav_state(self) -> NavState:
        return NavState(
            timestamp=float(self.timestamp if self.timestamp is not None else 0.0),
            position=self.position,
            velocity=self.velocity,
            yaw=self.yaw,
            accel_bias=self.state[IBA].copy(),
            gyro_bias=float(self.state[IBG]),
            covariance=self.covariance.copy(),
            localised=self.is_localised(),
            time_since_fix=self.time_since_fix(),
        )

    # -------------------------------------------------------------- prediction

    def predict(self, telemetry: TelloState) -> None:
        """Advance the filter to ``telemetry.timestamp`` using its inertial data.

        Also applies the two measurements that come free in the same packet: the
        Tello's own body-frame velocity estimate and the downward time-of-flight
        height when it is in range. Keeping them here rather than in separate
        calls means a caller cannot accidentally use the inertial half of a
        packet and forget the rest.
        """
        if not telemetry.valid:
            return
        stamp = float(telemetry.timestamp)
        if self.timestamp is None:
            self.timestamp = stamp
            self.attitude = np.asarray(telemetry.attitude, dtype=float).copy()
            self.last_accel = np.asarray(telemetry.accel, dtype=float).copy()
            self.last_gyro_z = float(telemetry.gyro_z)
            self._push_history()
            return

        dt = stamp - self.timestamp
        if dt <= 0.0:
            # A packet from the past. The telemetry socket does not reorder in
            # practice, so this means a clock step or a duplicated packet and
            # integrating a negative interval would run the filter backwards.
            return
        dt = min(dt, 0.5)

        self.attitude = np.asarray(telemetry.attitude, dtype=float).copy()
        self.last_accel = np.asarray(telemetry.accel, dtype=float).copy()
        self.last_gyro_z = float(telemetry.gyro_z)
        self._propagate(dt, self.last_accel, self.last_gyro_z)
        self.timestamp = stamp
        self.diagnostics.predictions += 1

        self.update_body_velocity(np.asarray(telemetry.velocity_body, dtype=float))
        low, high = self.settings.height_valid_range
        if low <= telemetry.height <= high:
            self.update_height(float(telemetry.height))

        self._push_history()

    def _propagate(self, dt: float, accel: np.ndarray, gyro_z: float) -> None:
        R = self.rotation()
        specific_force = np.asarray(accel, dtype=float) - self.state[IBA]
        acceleration = R @ specific_force + np.array([0.0, 0.0, -GRAVITY])

        self.state[IP] = self.state[IP] + self.state[IV] * dt + 0.5 * acceleration * dt**2
        self.state[IV] = self.state[IV] + acceleration * dt
        self.state[IYAW] = wrap_angle(self.state[IYAW] + (gyro_z - self.state[IBG]) * dt)

        F = np.eye(N_STATES)
        F[IP, IV] = np.eye(3) * dt
        F[IP, IBA] = -0.5 * R * dt**2
        F[IV, IBA] = -R * dt
        # Sensitivity of the world-frame acceleration to the yaw state: rotating
        # the body frame rotates the measured specific force with it.
        d_yaw = np.array(
            [
                [-np.sin(self.state[IYAW]), -np.cos(self.state[IYAW]), 0.0],
                [np.cos(self.state[IYAW]), -np.sin(self.state[IYAW]), 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        levelled = euler_to_matrix(0.0, self.attitude[1], self.attitude[2]) @ specific_force
        F[IV, IYAW] = (d_yaw @ levelled) * dt
        F[IP, IYAW] = 0.5 * (d_yaw @ levelled) * dt**2
        F[IYAW, IBG] = -dt

        Q = np.zeros((N_STATES, N_STATES))
        accel_var = (self.settings.accel_noise**2) * dt
        Q[IV, IV] = np.eye(3) * accel_var
        Q[IP, IP] = np.eye(3) * accel_var * dt**2 / 3.0
        Q[IP, IV] = np.eye(3) * accel_var * dt / 2.0
        Q[IV, IP] = Q[IP, IV]
        Q[IYAW, IYAW] = (self.settings.gyro_noise**2) * dt
        Q[IBA, IBA] = np.eye(3) * (self.settings.accel_bias_walk**2) * dt
        Q[IBG, IBG] = (self.settings.gyro_bias_walk**2) * dt

        self.covariance = F @ self.covariance @ F.T + Q
        self._symmetrise()

    # ------------------------------------------------------------------ updates

    def update_marker_fix(self, fix: MarkerFix) -> bool:
        """Apply an absolute fix. Returns whether it was accepted.

        Three ways a fix can fail to be applied and they are counted separately,
        because they mean different things to the operator: it can be too old for
        the history buffer, it can fail the chi-square gate, or the filter can be
        uninitialised, in which case the fix is not an update at all but the
        initialisation.
        """
        if not self._initialised_from_fix:
            self._initialise_from(fix)
            return True

        stamp = float(fix.timestamp)
        if self.timestamp is not None and stamp < self.timestamp:
            age = self.timestamp - stamp
            if age > self.settings.max_buffer_age:
                self.diagnostics.fixes_stale += 1
                return False
            return self._apply_delayed(fix)

        return self._apply_fix_now(fix)

    def _initialise_from(self, fix: MarkerFix) -> None:
        self.state[IP] = np.asarray(fix.position, dtype=float)
        self.state[IV] = np.zeros(3)
        self.state[IYAW] = float(fix.yaw)
        self.covariance[IP, IP] = np.asarray(fix.position_covariance, dtype=float)
        self.covariance[IYAW, IYAW] = float(fix.yaw_variance)
        self.timestamp = float(fix.timestamp)
        self.diagnostics.last_fix_time = float(fix.timestamp)
        self.diagnostics.fixes_accepted += 1
        self._initialised_from_fix = True
        self._push_history()

    def _apply_fix_now(self, fix: MarkerFix) -> bool:
        H = np.zeros((4, N_STATES))
        H[0:3, IP] = np.eye(3)
        H[3, IYAW] = 1.0

        innovation = np.empty(4)
        innovation[0:3] = np.asarray(fix.position, dtype=float) - self.state[IP]
        innovation[3] = wrap_angle(float(fix.yaw) - self.state[IYAW])

        R = np.zeros((4, 4))
        R[0:3, 0:3] = np.asarray(fix.position_covariance, dtype=float)
        R[3, 3] = max(float(fix.yaw_variance), 1e-6)

        S = H @ self.covariance @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.diagnostics.fixes_rejected += 1
            return False

        nis = float(innovation @ S_inv @ innovation)
        if not np.isfinite(nis) or nis > self.settings.fix_gate_chi2:
            self.diagnostics.fixes_rejected += 1
            self.diagnostics.last_rejected_nis = nis
            self.diagnostics.consecutive_rejections += 1
            if (
                self.diagnostics.consecutive_rejections
                >= self.settings.max_consecutive_rejections
            ):
                # The filter, not the fixes. See
                # FusionSettings.max_consecutive_rejections.
                self._initialise_from(fix)
                self.diagnostics.resets += 1
                self.diagnostics.consecutive_rejections = 0
                return True
            return False
        self.diagnostics.consecutive_rejections = 0

        K = self.covariance @ H.T @ S_inv
        self.state = self.state + K @ innovation
        self.state[IYAW] = wrap_angle(self.state[IYAW])
        self._joseph(K, H, R)

        self.diagnostics.fixes_accepted += 1
        self.diagnostics.last_fix_time = max(
            self.diagnostics.last_fix_time, float(fix.timestamp)
        )
        return True

    def _apply_delayed(self, fix: MarkerFix) -> bool:
        """Rewind to the fix's timestamp, apply it and replay forward.

        Exact rather than approximate, which matters because the delay is a
        substantial fraction of the interval between fixes. The alternative
        commonly used, applying the fix at the current time and pretending it is
        fresh, injects an error equal to however far the aircraft moved during
        the processing delay: at 0.8 m/s and 30 ms that is 2.4 cm, which is
        comparable with the fix's own accuracy.
        """
        stamp = float(fix.timestamp)
        index = None
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].timestamp <= stamp:
                index = i
                break
        if index is None:
            self.diagnostics.fixes_stale += 1
            return False

        replay = list(self._history)[index + 1 :]
        anchor = self._history[index]

        saved = (self.state.copy(), self.covariance.copy(), self.timestamp, self.attitude.copy())
        self.state = anchor.state.copy()
        self.covariance = anchor.covariance.copy()
        self.timestamp = anchor.timestamp
        self.attitude = anchor.attitude.copy()

        gap = stamp - anchor.timestamp
        if gap > 0:
            self._propagate(gap, anchor.accel, anchor.gyro_z)
            self.timestamp = stamp

        if not self._apply_fix_now(fix):
            self.state, self.covariance, self.timestamp, self.attitude = saved
            return False

        # Replay the inertial data that arrived after the fix, recording fresh
        # snapshots as we go. Re-using the old snapshots here would leave the
        # history holding pre-correction states, so a second delayed fix would
        # rewind to a state the filter no longer believes.
        rebuilt: List[_Snapshot] = []
        for snapshot in replay:
            step = snapshot.timestamp - self.timestamp
            if step > 0:
                self.attitude = snapshot.attitude.copy()
                self._propagate(step, snapshot.accel, snapshot.gyro_z)
                self.timestamp = snapshot.timestamp
            rebuilt.append(
                _Snapshot(
                    timestamp=float(self.timestamp),
                    state=self.state.copy(),
                    covariance=self.covariance.copy(),
                    attitude=self.attitude.copy(),
                    accel=snapshot.accel.copy(),
                    gyro_z=float(snapshot.gyro_z),
                )
            )
        self._history = deque(rebuilt)
        self._trim_history()
        return True

    def update_body_velocity(self, velocity_body: np.ndarray) -> None:
        """The Tello's own velocity estimate, rotated into the world."""
        R = self.rotation()
        measurement = R @ np.asarray(velocity_body, dtype=float)
        H = np.zeros((3, N_STATES))
        H[:, IV] = np.eye(3)
        R_noise = np.eye(3) * self.settings.velocity_noise**2
        self._linear_update(H, measurement - self.state[IV], R_noise)

    def update_height(self, height: float) -> None:
        """The downward time-of-flight sensor.

        Measures height above the *floor beneath the aircraft*, which is the same
        thing as world z only over a flat floor. The room this project targets has
        one; a room with a table in it does not and that is one of the
        limitations in ``docs/LIMITATIONS.md`` rather than something the filter
        can fix.
        """
        H = np.zeros((1, N_STATES))
        H[0, 2] = 1.0
        innovation = np.array([float(height) - self.state[2]])
        self._linear_update(H, innovation, np.array([[self.settings.height_noise**2]]))

    def update_visual_odometry(self, estimate: VoEstimate) -> bool:
        """A body-frame velocity from the visual-odometry front end.

        The noise scales with the inverse of the inlier ratio, so a frame pair
        where the tracker mostly failed contributes proportionally less. A front
        end that reports a velocity it does not believe in is worse than one that
        reports nothing, so ``valid`` is honoured before anything else.
        """
        if not estimate.valid or estimate.n_tracks < 8:
            self.diagnostics.vo_rejected += 1
            return False
        quality = max(float(estimate.inlier_ratio), 0.05)
        sigma = self.settings.vo_noise / quality
        R = self.rotation()
        measurement = R @ np.asarray(estimate.velocity_body, dtype=float)
        H = np.zeros((3, N_STATES))
        H[:, IV] = np.eye(3)
        self._linear_update(H, measurement - self.state[IV], np.eye(3) * sigma**2)
        self.diagnostics.vo_accepted += 1
        return True

    # ----------------------------------------------------------------- plumbing

    def _linear_update(self, H: np.ndarray, innovation: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.covariance @ H.T + R
        try:
            K = self.covariance @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.state = self.state + K @ innovation
        self.state[IYAW] = wrap_angle(self.state[IYAW])
        self._joseph(K, H, R)

    def _joseph(self, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Joseph-form covariance update.

        The short form ``(I - KH)P`` is algebraically equivalent and numerically
        is not: it loses symmetry and positive-definiteness over long runs and
        this filter runs for whole missions at 10 Hz with a badly conditioned
        covariance whenever the aircraft has not seen a marker for a while.
        """
        A = np.eye(N_STATES) - K @ H
        self.covariance = A @ self.covariance @ A.T + K @ R @ K.T
        self._symmetrise()

    def _symmetrise(self) -> None:
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self._apply_covariance_floor()

    def _apply_covariance_floor(self) -> None:
        """Stop the position covariance shrinking below the systematic error.

        Applied to the diagonal only and only upwards. Scaling the whole block
        would break the correlations the filter has legitimately learned between
        axes; raising the diagonal leaves them intact and simply refuses to claim
        more precision than the measurements can support. See
        ``FusionSettings.min_position_sigma``.
        """
        floor = self.settings.min_position_sigma**2
        if floor <= 0.0:
            return
        diagonal = np.diag(self.covariance)[IP]
        deficit = np.clip(floor - diagonal, 0.0, None)
        if np.any(deficit > 0):
            self.covariance[IP, IP] = self.covariance[IP, IP] + np.diag(deficit)

    def _push_history(self) -> None:
        self._history.append(
            _Snapshot(
                timestamp=float(self.timestamp),
                state=self.state.copy(),
                covariance=self.covariance.copy(),
                attitude=self.attitude.copy(),
                accel=self.last_accel.copy(),
                gyro_z=float(self.last_gyro_z),
            )
        )
        self._trim_history()

    def _trim_history(self) -> None:
        cutoff = float(self.timestamp) - self.settings.max_buffer_age
        while len(self._history) > 2 and self._history[0].timestamp < cutoff:
            self._history.popleft()
