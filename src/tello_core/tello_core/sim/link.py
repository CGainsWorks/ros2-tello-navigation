"""A :class:`~tello_core.link.base.TelloLink` backed by the simulator.

Everything above this class believes it is talking to an aircraft over a
lossy wireless link, because it is talking to something that behaves like one.
Telemetry is generated at 10 Hz and video at 30 Hz on a virtual clock, telemetry
is serialised into genuine Tello state strings and parsed back by the same decoder
the real driver uses, packets are dropped, the link goes down for whole seconds at
a time on a schedule and commands sent during an outage do not arrive.

The last of those is the one that makes the failsafe testable. It is easy to write
a link-loss test where the aircraft stops answering but still obeys; that tests
nothing. Here, when the link is down, the ``rc`` commands the controller is
cheerfully still sending go nowhere and the simulated aircraft does what a real
Tello does after fifteen seconds without a command, which is land itself. The
failsafe has to get the aircraft down before that happens, using the last
estimate it had, or the aircraft comes down somewhere nobody chose.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from ..camera import CameraModel
from ..config import AircraftConfig
from ..geometry import euler_to_matrix, make_transform
from ..link.base import LinkStatus, TelloLink
from ..link.state import decode_state, encode_state
from ..types import Frame, RcCommand, TelloState
from .dynamics import DroneState, DynamicsSettings, TelloDynamics
from .render import Renderer, RenderSettings
from .world import Room

# What a real Tello does when it stops hearing from the controller. Documented in
# the SDK: the aircraft lands after fifteen seconds without a command.
COMMAND_TIMEOUT = 15.0


@dataclass(frozen=True)
class LinkFaults:
    """Everything that can go wrong with the wireless link, on a schedule.

    Scheduled rather than purely random, because a failsafe test has to be
    repeatable and "run it enough times and the link will drop somewhere useful"
    is not a test. Random loss is available too, for the background attrition that
    a 2.4 GHz link in a building always has.
    """

    outages: Tuple[Tuple[float, float], ...] = ()
    """``(start, duration)`` in seconds of flight time. Nothing gets through in
    either direction during an outage."""

    video_outages: Tuple[Tuple[float, float], ...] = ()
    """``(start, duration)`` for the video stream alone.

    A separate schedule because losing video is a different failure from losing
    the link and the aircraft's response has to be different too. With the
    telemetry still arriving the aircraft is fully controllable and completely
    unable to localise, so the correct response is to hold, look for a marker
    and land under control if one does not appear. With the whole link down it
    cannot be commanded at all and the correct response is to fall silent.

    Modelling only total link loss would leave the more common failure untested:
    the Tello's video is the first thing to go when the signal degrades, because
    it is by far the bulkier of the two streams."""

    state_loss: float = 0.02
    """Probability an individual telemetry packet is lost."""

    frame_loss: float = 0.01
    command_loss: float = 0.0
    """Probability an rc command is lost. Zero by default: on a healthy link the
    Tello's command channel is reliable enough that modelling loss on it would be
    modelling the wrong failure. The outage schedule is how command loss is
    tested."""

    latency: float = 0.03
    """Seconds of one-way delay applied to telemetry and video. The processing
    delay this creates is the reason the fusion filter has an out-of-order
    measurement path at all."""

    malformed_state: float = 0.005
    """Probability a telemetry packet arrives corrupted rather than lost. Exercises
    the decoder's tolerance, which is otherwise dead code until the day a real
    link produces one."""


@dataclass
class SimSettings:
    """Rates and quality of the simulated aircraft's two streams."""

    telemetry_rate: float = 10.0
    video_rate: float = 30.0
    physics_rate: float = 100.0
    """The dynamics are integrated faster than either stream is sampled, so the
    integration error is not part of what the estimator is being judged on."""

    exposure_time: float = 1.0 / 60.0
    render_scale: float = 1.0
    """Renders at a fraction of full resolution. Anything below 1.0 changes the
    detection range, so results are only comparable at the same value and the
    value is recorded in the results metadata."""

    faults: LinkFaults = field(default_factory=LinkFaults)
    seed: int = 0


class SimLink(TelloLink):
    """The simulator behind the link interface."""

    def __init__(
        self,
        room: Room,
        config: AircraftConfig,
        camera: Optional[CameraModel] = None,
        settings: Optional[SimSettings] = None,
        dynamics_settings: Optional[DynamicsSettings] = None,
        render_settings: Optional[RenderSettings] = None,
        start_position: Optional[np.ndarray] = None,
        start_yaw: float = 0.0,
    ) -> None:
        self.room = room
        self.config = config
        self.settings = settings or SimSettings()
        base_camera = camera or CameraModel.tello_nominal()
        self.camera = (
            base_camera
            if self.settings.render_scale == 1.0
            else base_camera.scaled(self.settings.render_scale)
        )
        self.dynamics = TelloDynamics(
            config,
            dynamics_settings,
            start_position=start_position,
            start_yaw=start_yaw,
            seed=self.settings.seed,
        )
        self.renderer = Renderer(
            room,
            self.camera,
            render_settings or RenderSettings(seed=self.settings.seed + 1),
        )

        self._rng = np.random.default_rng(self.settings.seed + 2)
        self._time = 0.0
        self._connected = False
        self._video = False
        self._next_state = 0.0
        self._next_frame = 0.0
        self._frame_index = 0
        self._state_queue: Deque[Tuple[float, str]] = deque()
        self._frame_queue: Deque[Tuple[float, int, np.ndarray, np.ndarray, np.ndarray]] = deque()
        self.frames_rendered = 0
        self.frames_discarded = 0
        self._pending_state: Optional[TelloState] = None
        self._pending_frame: Optional[Frame] = None
        self._previous_state: Optional[TelloState] = None
        self._last_state_time = float("-inf")
        self._last_frame_time = float("-inf")
        self._last_command_time = 0.0
        self._packets_received = 0
        self._packets_dropped = 0
        self._commands_sent = 0
        self.truth_log: List[DroneState] = []

    # ------------------------------------------------------------------- clock

    def now(self) -> float:
        return self._time

    def advance(self, dt: float) -> None:
        """Step the simulation forward by ``dt`` seconds of flight time."""
        if dt <= 0:
            return
        steps = max(1, int(np.ceil(dt * self.settings.physics_rate)))
        step = dt / steps
        for _ in range(steps):
            self._time += step
            self._enforce_command_timeout()
            truth = self.dynamics.step(step)
            self.truth_log.append(truth)
            self._maybe_emit_state()
            self._maybe_emit_frame()

    # ------------------------------------------------------------------- link

    def connect(self) -> bool:
        self._connected = True
        self._next_state = self._time
        self._next_frame = self._time
        self._last_command_time = self._time
        return True

    def close(self) -> None:
        self._connected = False
        self._video = False
        self._state_queue.clear()
        self._frame_queue.clear()

    def start_video(self) -> bool:
        if not self._connected:
            return False
        self._video = True
        return True

    def _link_up(self) -> bool:
        if not self._connected:
            return False
        return not self._inside(self.settings.faults.outages)

    def _video_up(self) -> bool:
        """The video stream specifically, which can fail on its own."""
        return self._link_up() and not self._inside(self.settings.faults.video_outages)

    def _inside(self, windows) -> bool:
        return any(start <= self._time < start + duration for start, duration in windows)

    def _enforce_command_timeout(self) -> None:
        """The aircraft's own dead-man behaviour.

        Fifteen seconds without a command and a real Tello lands itself. It is
        modelled because it changes what a link-loss failsafe has to achieve: the
        aircraft is not going to hover politely until the link comes back and a
        failsafe that assumes it will is a failsafe that has never been tested
        against the aircraft it is flying.
        """
        if (
            self.dynamics.state.flying
            and self._time - self._last_command_time > COMMAND_TIMEOUT
        ):
            self.dynamics.land()

    # -------------------------------------------------------------- production

    def _maybe_emit_state(self) -> None:
        interval = 1.0 / self.settings.telemetry_rate
        while self._time >= self._next_state:
            stamp = self._next_state
            self._next_state += interval
            if not self._link_up():
                self._packets_dropped += 1
                continue
            if self._rng.random() < self.settings.faults.state_loss:
                self._packets_dropped += 1
                continue
            payload = encode_state(self.dynamics.telemetry(stamp))
            if self._rng.random() < self.settings.faults.malformed_state:
                cut = int(self._rng.integers(1, max(2, len(payload) // 2)))
                payload = payload[:cut]
            self._state_queue.append((stamp + self.settings.faults.latency, payload))

    def _maybe_emit_frame(self) -> None:
        if not self._video:
            return
        interval = 1.0 / self.settings.video_rate
        while self._time >= self._next_frame:
            stamp = self._next_frame
            self._next_frame += interval
            if not self._video_up():
                continue
            if self._rng.random() < self.settings.faults.frame_loss:
                continue
            # The pose is captured now; the image is not rendered until somebody
            # actually polls for it. Rendering eagerly means rendering every frame
            # the aircraft emits and a control loop at 20 Hz against a camera at
            # 30 Hz throws a third of them away unseen. That third was a third of
            # the entire cost of a simulated flight, spent on images nothing ever
            # looked at.
            truth = self.dynamics.state
            T_wc = (
                make_transform(
                    euler_to_matrix(truth.yaw, truth.pitch, truth.roll), truth.position
                )
                @ self.config.mount.transform()
            )
            self._frame_queue.append(
                (
                    stamp + self.settings.faults.latency,
                    self._frame_index,
                    T_wc,
                    np.array([0.0, 0.0, truth.yaw_rate]),
                    truth.velocity.copy(),
                )
            )
            self._frame_index += 1

    # ---------------------------------------------------------------- polling

    def poll_state(self) -> Optional[TelloState]:
        if not self._link_up():
            # A packet already in flight when the link drops does not arrive
            # either. Delivering the queue during an outage would make the
            # telemetry age tick over exactly when the failsafe is watching it.
            self._packets_dropped += len(self._state_queue)
            self._state_queue.clear()
            return None
        newest: Optional[TelloState] = None
        while self._state_queue and self._state_queue[0][0] <= self._time:
            _, payload = self._state_queue.popleft()
            # Stamped with the arrival time, not the emission time. That is all a
            # real driver knows: the aircraft's own clock is not in the packet in
            # any usable form, so the latency shows up as a lag the estimator has
            # to tolerate rather than as a correction it can apply.
            state = decode_state(payload, self._time, self._previous_state)
            if state.valid:
                self._previous_state = state
                self._last_state_time = self._time
                self._packets_received += 1
                newest = state
            else:
                self._packets_dropped += 1
        return newest

    def poll_frame(self) -> Optional[Frame]:
        if not self._video_up():
            self._frame_queue.clear()
            return None
        pending = None
        while self._frame_queue and self._frame_queue[0][0] <= self._time:
            if pending is not None:
                self.frames_discarded += 1
            pending = self._frame_queue.popleft()
        if pending is None:
            return None

        _, index, pose, angular_velocity, linear_velocity = pending
        image = self.renderer.render(
            pose,
            angular_velocity=angular_velocity,
            linear_velocity=linear_velocity,
            exposure_time=self.settings.exposure_time,
        )
        self.frames_rendered += 1
        self._last_frame_time = self._time
        # Stamped on arrival for the same reason the telemetry is. The delay
        # between this stamp and the moment a marker fix derived from the frame
        # reaches the filter is the processing time and that one *is* measurable
        # by the host, which is what the filter's out-of-order path uses.
        return Frame(timestamp=self._time, image=image, index=index)

    # --------------------------------------------------------------- commands

    def send_rc(self, command: RcCommand) -> None:
        self._commands_sent += 1
        if not self._link_up():
            return
        if self._rng.random() < self.settings.faults.command_loss:
            return
        self._last_command_time = self._time
        self.dynamics.set_rc(command)

    def takeoff(self) -> None:
        self._commands_sent += 1
        if not self._link_up():
            return
        self._last_command_time = self._time
        self.dynamics.takeoff()

    def land(self) -> None:
        self._commands_sent += 1
        if not self._link_up():
            return
        self._last_command_time = self._time
        self.dynamics.land()

    def emergency(self) -> None:
        self._commands_sent += 1
        if not self._link_up():
            return
        self._last_command_time = self._time
        self.dynamics.emergency()

    def status(self) -> LinkStatus:
        return LinkStatus(
            connected=self._link_up(),
            last_state_age=self._time - self._last_state_time,
            last_frame_age=self._time - self._last_frame_time,
            packets_received=self._packets_received,
            packets_dropped=self._packets_dropped,
            commands_sent=self._commands_sent,
            detail="simulated",
        )

    # ------------------------------------------------------------------ truth

    def truth(self) -> DroneState:
        """Ground truth, for the evaluation only. Nothing in the flight software
        may call this and nothing does: the mission runner receives it through a
        separate recorder so that the temptation is not even available."""
        return self.dynamics.state.copy()
