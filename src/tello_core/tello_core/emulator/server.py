"""The emulated aircraft: three UDP ports, a physics thread and a camera.

What it speaks
--------------
Tello SDK 2.0 as the driver in :mod:`tello_core.link.udp` assumes it, which is
DJI's published document plus the behaviour that is widely described for the
aircraft:

* **Commands** arrive as text on the command port (8889). Nothing is answered
  until ``command`` has put the aircraft into SDK mode. Control commands answer
  ``ok`` or ``error``; ``takeoff``, ``land`` and the movement commands answer
  when the manoeuvre has finished, not when it starts; ``rc`` is never answered;
  read commands (``battery?``, ``height?`` and so on) answer with a value.
  Replies go back to whichever address and port the command came from, from
  the command port, as the aircraft's do.
* **State** strings go to port 8890 on the host that sent ``command``, at 10 Hz,
  in the format :func:`tello_core.link.state.encode_state` writes and the
  driver's decoder reads.
* **Video** goes to port 11111 on the same host after ``streamon``: frames
  rendered from the simulated room by :class:`~tello_core.sim.render.Renderer`,
  encoded to H.264 by libx264 through PyAV and cut into datagrams of at most
  1460 bytes.
* The aircraft **lands itself** fifteen seconds after the last command it heard.

What it is not
--------------
It is an emulator of the protocol as this repository understands it, built from
the same assumptions as the driver. Where the SDK document is silent (the sign of
the ``rc`` yaw channel, the units of ``agx``, the exact error strings, the
datagram size) the emulator and the driver agree because they were written to
the same assumption, so a green test here cannot catch a wrong assumption.
``docs/HARDWARE.md`` lists them. Every number it produces is simulated.

Faults
------
Link outages are scheduled in seconds after the first accepted ``takeoff``,
because the time a stack spends on the ground waiting for its first marker fix
varies from run to run in real time and a fault meant to hit mid-flight has to be
timed from something that does not. During an outage nothing gets through in
either direction: commands are not received, replies, state and video are not
sent. The encoder keeps running, so the decoder on the far side meets the gap the
way it would on a real link, with missing reference frames until the next
keyframe.

Ground truth
------------
Written to a CSV file, never to the Tello ports. Nothing on the driver side can
see it; the launch test reads the file after the flight to measure position
error against it.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import socket
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..camera import CameraModel
from ..config import AircraftConfig
from ..geometry import euler_to_matrix, make_transform, wrap_angle
from ..link.state import encode_state
from ..link.video import TELLO_VIDEO_DATAGRAM, H264Encoder, packetise
from ..sim.dynamics import DynamicsSettings, TelloDynamics
from ..sim.render import Renderer, RenderSettings
from ..sim.world import Room
from ..types import RcCommand

MOVE_COMMANDS = ("forward", "back", "left", "right", "up", "down")
"""SDK distance commands, each taking a distance in centimetres from 20 to 500."""


@dataclass(frozen=True)
class EmulatorFaults:
    """What goes wrong and when. Times are seconds after the first take-off."""

    outages: Tuple[Tuple[float, float], ...] = ()
    """``(start, duration)``. Nothing gets through in either direction."""

    video_outages: Tuple[Tuple[float, float], ...] = ()
    """``(start, duration)`` for the video stream alone."""

    state_loss: float = 0.0
    """Probability an individual state datagram is not sent."""

    frame_loss: float = 0.0
    """Probability an encoded frame's datagrams are not sent."""


@dataclass(frozen=True)
class EmulatorSettings:
    """Ports, rates and the simulated aircraft's starting condition."""

    bind_address: str = "127.0.0.1"
    """Where the command socket listens. Loopback by default: this is a test
    fixture and has no business answering on a real network."""

    command_port: int = 8889
    state_port: int = 8890
    """Destination port for state strings on the host that sent ``command``."""

    video_port: int = 11111
    """Destination port for the H.264 stream."""

    telemetry_rate: float = 10.0
    video_rate: float = 30.0
    """Target frame rate. The renderer is numpy and on a slow machine the
    achieved rate is lower; :meth:`TelloEmulator.stats` reports it."""

    physics_rate: float = 100.0
    render_workers: int = 0
    """Processes rendering frames in parallel. Zero renders in the video thread,
    which is simplest and enough for the unit tests; the launch tests use a pool
    because the numpy ray caster manages only a handful of 960x720 frames per
    second on one core."""

    bitrate: int = 1_500_000
    """libx264 target bitrate in bits per second. An emulator setting, not a
    figure measured on an aircraft."""

    keyframe_interval: int = 30
    exposure_time: float = 1.0 / 60.0
    seed: int = 1
    battery: float = 1.0
    """Starting charge as a fraction."""

    battery_drain: float = 1.0
    """Multiplier on the simulated drain. Above one only to reach the battery
    failsafes in a test of reasonable length; recorded in the truth log header
    so a battery figure cannot be mistaken for an endurance claim."""

    command_timeout: float = 15.0
    """Seconds without a command before the aircraft lands itself."""

    move_timeout: float = 20.0
    """Seconds a distance or rotation command may take before it is answered
    with ``error``."""

    faults: EmulatorFaults = field(default_factory=EmulatorFaults)
    truth_path: Optional[str] = None
    """CSV of ground truth at ``truth_rate``. ``None`` disables it."""

    events_path: Optional[str] = None
    """JSON lines: every datagram received or dropped, every reply, every
    automatic action. ``None`` disables it."""

    truth_rate: float = 20.0


class TelloEmulator:
    """One emulated aircraft. ``start()`` it, point a driver at it, ``close()`` it."""

    def __init__(
        self,
        room: Room,
        config: AircraftConfig,
        camera: Optional[CameraModel] = None,
        settings: Optional[EmulatorSettings] = None,
        start_position: Optional[np.ndarray] = None,
        start_yaw: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.room = room
        self.config = config
        self.camera = camera or CameraModel.tello_nominal()
        self.settings = settings or EmulatorSettings()
        nominal = DynamicsSettings()
        self.dynamics = TelloDynamics(
            config,
            replace(
                nominal,
                # In-flight drain only, as scripts/run_mission.py --battery-drain
                # does, so a battery scenario means the same thing in both.
                hover_current=nominal.hover_current * max(self.settings.battery_drain, 0.0),
            ),
            start_position=start_position,
            start_yaw=start_yaw,
            seed=self.settings.seed,
        )
        self.dynamics.state.battery = float(np.clip(self.settings.battery, 0.0, 1.0))
        self.renderer = Renderer(
            room,
            self.camera,
            # No JPEG stage: the H.264 encoder is the compression now and doing
            # both would put two generations of artefacts on every frame.
            RenderSettings(jpeg_quality=0, seed=self.settings.seed + 1),
        )
        self._clock = clock
        self._rng = np.random.default_rng(self.settings.seed + 2)
        self._lock = threading.RLock()
        self._running = threading.Event()
        self._threads: List[threading.Thread] = []
        self._command_socket: Optional[socket.socket] = None
        self._send_socket: Optional[socket.socket] = None
        self._encoder: Optional[H264Encoder] = None

        self._sdk_mode = False
        self._client_ip: Optional[str] = None
        self._streaming = False
        self._takeoff_time: Optional[float] = None
        self._last_command = self._clock()
        self._motor_time = 0.0
        self._speed_cm = 100.0
        self._pending: Dict[str, tuple] = {}
        self._move: Optional[dict] = None
        self._deferred: List[tuple] = []
        """Replies decided under the lock and sent after it is released."""
        self._was_link_up = True
        self._auto_landed = False

        self._truth_file = None
        self._truth_writer = None
        self._events_file = None

        self.counters: Dict[str, int] = {
            "commands_received": 0,
            "commands_dropped": 0,
            "replies_sent": 0,
            "state_sent": 0,
            "frames_rendered": 0,
            "frames_skipped": 0,
            "frames_sent": 0,
            "video_datagrams": 0,
            "video_bytes": 0,
        }
        self._video_started_at: Optional[float] = None
        self._render_delay = 0.1
        """Running estimate of seconds from capture to send; see _video_loop."""

    # ------------------------------------------------------------------ lifecycle

    @property
    def command_address(self) -> Tuple[str, int]:
        """The address a driver should send commands to, with the bound port."""
        if self._command_socket is None:
            return (self.settings.bind_address, self.settings.command_port)
        return self._command_socket.getsockname()[:2]

    @property
    def running(self) -> bool:
        """False once closed or once any of its threads has failed."""
        return self._running.is_set()

    def start(self) -> "TelloEmulator":
        self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._command_socket.bind((self.settings.bind_address, self.settings.command_port))
        self._command_socket.settimeout(0.1)
        self._send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._open_logs()
        self._running.set()
        for target, name in (
            (self._command_loop, "emulator-command"),
            (self._physics_loop, "emulator-physics"),
            (self._state_loop, "emulator-state"),
            (self._video_loop, "emulator-video"),
        ):
            thread = threading.Thread(target=self._guard(target), name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._event("started", f"listening on {self.command_address[0]}:{self.command_address[1]}")
        return self

    def close(self) -> None:
        if not self._running.is_set() and not self._threads:
            return
        self._running.clear()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)
        self._threads = []
        for handle in (self._command_socket, self._send_socket):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._command_socket = None
        self._send_socket = None
        self._event("stopped", json.dumps(self.stats()))
        for handle in (self._truth_file, self._events_file):
            if handle is not None:
                handle.close()
        self._truth_file = None
        self._events_file = None

    def __enter__(self) -> "TelloEmulator":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def _guard(self, target):
        """A thread that dies takes the emulator down loudly rather than quietly."""

        def run() -> None:
            try:
                target()
            except Exception:  # noqa: BLE001 - reported, then the emulator stops
                traceback.print_exc(file=sys.stderr)
                self._event("thread_crashed", threading.current_thread().name)
                self._running.clear()

        return run

    # -------------------------------------------------------------------- faults

    def flight_clock(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds since the first accepted take-off, or ``None`` before it."""
        if self._takeoff_time is None:
            return None
        return (self._clock() if now is None else now) - self._takeoff_time

    def _inside(self, windows, now: float) -> bool:
        t = self.flight_clock(now)
        if t is None:
            return False
        return any(start <= t < start + duration for start, duration in windows)

    def link_up(self, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else now
        return not self._inside(self.settings.faults.outages, now)

    def _video_up(self, now: float) -> bool:
        return self.link_up(now) and not self._inside(self.settings.faults.video_outages, now)

    # ------------------------------------------------------------------ commands

    def _command_loop(self) -> None:
        while self._running.is_set():
            try:
                payload, sender = self._command_socket.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                if not self._running.is_set():
                    return
                continue
            now = self._clock()
            text = payload.decode("ascii", "replace").strip()
            if not self.link_up(now):
                self.counters["commands_dropped"] += 1
                self._event("dropped", text)
                continue
            self.counters["commands_received"] += 1
            self._event("command", text)
            with self._lock:
                reply = self._handle(text, sender, now)
            if reply is not None:
                self._reply(reply, sender)

    def _reply(self, text: str, address) -> None:
        if not self.link_up() or self._command_socket is None:
            self._event("reply_lost", text)
            return
        try:
            self._command_socket.sendto(text.encode("ascii"), address)
        except OSError:
            return
        self.counters["replies_sent"] += 1
        self._event("reply", text)

    def _handle(self, text: str, sender, now: float) -> Optional[str]:
        """Apply one command and return the immediate reply, if there is one."""
        words = text.split()
        if not words:
            return None
        verb = words[0].lower()
        if verb == "command":
            self._sdk_mode = True
            self._client_ip = sender[0]
            self._last_command = now
            return "ok"
        if not self._sdk_mode:
            # Outside SDK mode the aircraft does not answer text commands.
            return None
        self._last_command = now
        state = self.dynamics.state

        if verb == "rc":
            self._handle_rc(words)
            return None
        if verb == "takeoff":
            if state.flying or state.motors_cut or "takeoff" in self._pending:
                return "error"
            if self._takeoff_time is None:
                self._takeoff_time = now
            self.dynamics.takeoff()
            self._pending["takeoff"] = (sender, now)
            return None
        if verb == "land":
            if not state.flying:
                return "error"
            self._cancel_move("error")
            self.dynamics.land()
            self._pending["land"] = (sender, now)
            return None
        if verb == "emergency":
            self._cancel_move("error")
            self.dynamics.emergency()
            return "ok"
        if verb == "stop":
            self._cancel_move("error")
            self.dynamics.set_rc(RcCommand.zero())
            return "ok"
        if verb == "streamon":
            self._streaming = True
            return "ok"
        if verb == "streamoff":
            self._streaming = False
            return "ok"
        if verb == "speed" and len(words) == 2:
            value = _as_number(words[1])
            if value is None or not 10 <= value <= 100:
                return "error"
            self._speed_cm = float(value)
            return "ok"
        if verb in MOVE_COMMANDS and len(words) == 2:
            return self._start_move(verb, words[1], sender, now)
        if verb in ("cw", "ccw") and len(words) == 2:
            return self._start_turn(verb, words[1], sender, now)
        if verb.endswith("?"):
            return self._query(verb)
        return f"unknown command: {verb}"

    def _handle_rc(self, words) -> None:
        if len(words) != 5:
            return
        values = [_as_number(word) for word in words[1:]]
        if any(v is None or not -100 <= v <= 100 for v in values):
            return
        a, b, c, d = (int(round(v)) for v in values)
        self._cancel_move("error")
        # SDK channel a is positive to the right; the simulator's body frame has
        # y to the left. The inverse of the flip in UdpLink.send_rc.
        self.dynamics.set_rc(RcCommand(left_right=-a, forward_back=b, up_down=c, yaw_rate=d))

    def _query(self, verb: str) -> str:
        state = self.dynamics.state
        height = max(0.0, state.position[2] - self.dynamics.settings.resting_height)
        if verb == "battery?":
            return str(int(round(state.battery * 100)))
        if verb == "speed?":
            return f"{self._speed_cm:.1f}"
        if verb == "time?":
            return f"{int(self._motor_time)}s"
        if verb == "height?":
            return f"{int(round(height * 10))}dm"
        if verb == "tof?":
            return f"{int(round(height * 1000))}mm"
        if verb == "temp?":
            return f"{int(round(state.temperature - 1))}~{int(round(state.temperature + 1))}C"
        if verb == "baro?":
            return f"{state.position[2]:.2f}"
        if verb in ("attitude?", "acceleration?"):
            fields = dict(
                item.split(":", 1)
                for item in encode_state(self.dynamics.telemetry(self._motor_time)).split(";")
                if ":" in item
            )
            keys = ("pitch", "roll", "yaw") if verb == "attitude?" else ("agx", "agy", "agz")
            return "".join(f"{key}:{fields[key]};" for key in keys)
        # wifi?, sdk?, sn? and the rest describe hardware the emulator does not
        # have. Answering with a made-up value would look like data.
        return "error Not supported by the emulator"

    # ------------------------------------------------------------- movements

    def _start_move(self, verb: str, raw: str, sender, now: float) -> Optional[str]:
        distance = _as_number(raw)
        if distance is None or not 20 <= distance <= 500:
            return "error"
        if not self.dynamics.state.flying or self._busy():
            return "error"
        metres = distance / 100.0
        state = self.dynamics.state
        forward = np.array([np.cos(state.yaw), np.sin(state.yaw), 0.0])
        left = np.array([-np.sin(state.yaw), np.cos(state.yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        direction = {
            "forward": forward,
            "back": -forward,
            "left": left,
            "right": -left,
            "up": up,
            "down": -up,
        }[verb]
        self._move = {
            "target": state.position + direction * metres,
            "yaw": state.yaw,
            "sender": sender,
            "started": now,
        }
        return None

    def _start_turn(self, verb: str, raw: str, sender, now: float) -> Optional[str]:
        degrees = _as_number(raw)
        if degrees is None or not 1 <= degrees <= 360:
            return "error"
        if not self.dynamics.state.flying or self._busy():
            return "error"
        state = self.dynamics.state
        # Clockwise seen from above is a negative rotation about z-up.
        sign = -1.0 if verb == "cw" else 1.0
        self._move = {
            "target": state.position.copy(),
            "yaw": wrap_angle(state.yaw + sign * np.deg2rad(degrees)),
            "sender": sender,
            "started": now,
        }
        return None

    def _busy(self) -> bool:
        return self._move is not None or bool(self._pending) or self.dynamics.taking_off

    def _cancel_move(self, reply: str) -> None:
        if self._move is not None:
            sender = self._move["sender"]
            self._move = None
            self.dynamics.set_rc(RcCommand.zero())
            self._deferred.append((reply, sender))

    def _steer_move(self, now: float) -> None:
        """A small position loop on the true state, standing in for the Tello's
        own vision-positioning hold that executes these commands on the
        aircraft."""
        move = self._move
        state = self.dynamics.state
        rc = self.config.rc
        error = move["target"] - state.position
        yaw_error = wrap_angle(move["yaw"] - state.yaw)
        if np.linalg.norm(error) < 0.05 and abs(yaw_error) < np.deg2rad(2.0):
            self._move = None
            self.dynamics.set_rc(RcCommand.zero())
            self._deferred.append(("ok", move["sender"]))
            return
        if now - move["started"] > self.settings.move_timeout:
            self._cancel_move("error")
            return
        speed = self._speed_cm / 100.0
        velocity = 1.5 * error
        horizontal = np.linalg.norm(velocity[:2])
        if horizontal > speed:
            velocity[:2] *= speed / horizontal
        rotation = euler_to_matrix(state.yaw)
        body = rotation.T @ velocity
        yaw_rate = float(np.clip(2.0 * yaw_error, -0.5 * rc.yaw_rate_max, 0.5 * rc.yaw_rate_max))
        self.dynamics.set_rc(
            RcCommand(
                left_right=_counts(body[1], rc.horizontal_max),
                forward_back=_counts(body[0], rc.horizontal_max),
                up_down=_counts(body[2], rc.vertical_max),
                yaw_rate=_counts(yaw_rate, rc.yaw_rate_max),
            )
        )

    # ------------------------------------------------------------------- physics

    def _physics_loop(self) -> None:
        period = 1.0 / self.settings.physics_rate
        truth_period = 1.0 / self.settings.truth_rate
        last = self._clock()
        next_truth = last
        while self._running.is_set():
            tick_start = self._clock()
            dt = min(tick_start - last, 0.1)
            last = tick_start
            with self._lock:
                self._step(dt, tick_start)
                if tick_start >= next_truth:
                    next_truth = tick_start + truth_period
                    self._record_truth(tick_start)
                deferred = self._deferred
                self._deferred = []
            for reply, address in deferred:
                self._reply(reply, address)
            remaining = period - (self._clock() - tick_start)
            if remaining > 0:
                time.sleep(remaining)

    def _step(self, dt: float, now: float) -> None:
        state = self.dynamics.state
        link_up = self.link_up(now)
        if link_up != self._was_link_up:
            self._event(
                "link_up" if link_up else "link_down", f"t={self.flight_clock(now) or 0.0:.2f}"
            )
            self._was_link_up = link_up

        if (
            state.flying
            and not self.dynamics.landing
            and now - self._last_command > self.settings.command_timeout
        ):
            self._cancel_move("error")
            self.dynamics.land()
            self._auto_landed = True
            self._event("auto_land", f"{now - self._last_command:.1f} s without a command")

        if self._move is not None:
            self._steer_move(now)

        steps = max(1, int(np.ceil(dt * self.settings.physics_rate))) if dt > 0 else 0
        for _ in range(steps):
            self.dynamics.step(dt / steps)
        if self.dynamics.state.flying:
            self._motor_time += max(dt, 0.0)

        if "takeoff" in self._pending and not self.dynamics.taking_off:
            sender, _ = self._pending.pop("takeoff")
            reply = "ok" if self.dynamics.state.flying else "error"
            self._deferred.append((reply, sender))
            self._event("takeoff_complete", reply)
        if "land" in self._pending and not self.dynamics.state.flying:
            sender, _ = self._pending.pop("land")
            self._deferred.append(("ok", sender))
        if self.dynamics.state.motors_cut:
            for key in list(self._pending):
                sender, _ = self._pending.pop(key)
                self._deferred.append(("error", sender))

    # --------------------------------------------------------------------- state

    def _state_loop(self) -> None:
        period = 1.0 / self.settings.telemetry_rate
        next_time = self._clock()
        while self._running.is_set():
            next_time += period
            delay = next_time - self._clock()
            if delay > 0:
                time.sleep(delay)
            else:
                next_time = self._clock()
            now = self._clock()
            if not self._sdk_mode or self._client_ip is None or not self.link_up(now):
                continue
            if self._rng.random() < self.settings.faults.state_loss:
                continue
            with self._lock:
                payload = encode_state(self.dynamics.telemetry(self._motor_time))
                destination = (self._client_ip, self.settings.state_port)
            try:
                self._send_socket.sendto(payload.encode("ascii"), destination)
            except OSError:
                continue
            self.counters["state_sent"] += 1

    # --------------------------------------------------------------------- video

    def _video_loop(self) -> None:
        """Capture on a fixed schedule, render, encode in capture order, send.

        With ``render_workers`` above zero the frames are rendered in a process
        pool, so several are in flight at once. A capture tick that finds every
        worker busy is skipped, as a camera whose pipeline is full drops a frame.

        Rendering takes the emulator tens of milliseconds of its own and on a
        loaded machine a good deal more. That delay is an artefact of a numpy
        ray caster, not a property of a camera. Left alone it showed up as
        a marker fix stamped on arrival but describing where the aircraft had
        been: at cruise speed the launch tests saw fixes 0.3 m behind the truth
        and the failsafe holding for lost localisation. So each frame is
        rendered at the pose extrapolated, from the true velocity and yaw rate,
        to when it is expected to be sent, using a running estimate of the
        delay from capture to send. The stream then has the latency of the link
        and the decoder only. A real camera's own latency is not modelled;
        ``docs/HARDWARE.md`` lists measuring it.
        """
        period = 1.0 / self.settings.video_rate
        workers = max(0, int(self.settings.render_workers))
        pool = self._start_render_pool(workers) if workers else None
        pending: Deque[Tuple[object, Tuple[str, int], float]] = deque()
        next_time = self._clock()
        try:
            while self._running.is_set():
                if not self._streaming or self._client_ip is None:
                    pending.clear()
                    time.sleep(0.02)
                    next_time = self._clock()
                    continue
                if self._video_started_at is None:
                    self._video_started_at = self._clock()

                while pending and (pool is None or pending[0][0].done()):
                    job, destination, captured = pending.popleft()
                    image = job if pool is None else job.result()
                    self._encode_and_send(image, destination)
                    delay = self._clock() - captured
                    self._render_delay += 0.2 * (delay - self._render_delay)

                now = self._clock()
                if now < next_time:
                    time.sleep(min(next_time - now, 0.005))
                    continue
                # Behind schedule means the renderer cannot keep up. Skip ahead
                # rather than bursting to catch up, which is what a camera does.
                next_time = max(next_time + period, now)
                if pool is not None and len(pending) >= workers:
                    self.counters["frames_skipped"] += 1
                    continue

                with self._lock:
                    truth = self.dynamics.state.copy()
                    destination = (self._client_ip, self.settings.video_port)
                ahead = self._render_delay
                pose = (
                    make_transform(
                        euler_to_matrix(
                            truth.yaw + truth.yaw_rate * ahead, truth.pitch, truth.roll
                        ),
                        truth.position + truth.velocity * ahead,
                    )
                    @ self.config.mount.transform()
                )
                arguments = (
                    pose,
                    np.array([0.0, 0.0, truth.yaw_rate]),
                    truth.velocity.copy(),
                    self.settings.exposure_time,
                )
                if pool is None:
                    job = _render_with(self.renderer, *arguments)
                else:
                    job = pool.submit(_render_in_worker, *arguments)
                pending.append((job, destination, now))
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

    def _start_render_pool(self, workers: int):
        context = multiprocessing.get_context("spawn")
        return ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_render_worker,
            initargs=(self.room, self.camera, self.settings.seed + 1),
        )

    def _encode_and_send(self, image: np.ndarray, destination) -> None:
        self.counters["frames_rendered"] += 1
        if self._encoder is None:
            self._encoder = H264Encoder(
                self.camera.width,
                self.camera.height,
                fps=self.settings.video_rate,
                bitrate=self.settings.bitrate,
                keyframe_interval=self.settings.keyframe_interval,
            )
        stream = self._encoder.encode(image)
        if not stream or not self._video_up(self._clock()):
            return
        if self._rng.random() < self.settings.faults.frame_loss:
            return
        for datagram in packetise(stream, TELLO_VIDEO_DATAGRAM):
            try:
                self._send_socket.sendto(datagram, destination)
            except OSError:
                return
            self.counters["video_datagrams"] += 1
            self.counters["video_bytes"] += len(datagram)
        self.counters["frames_sent"] += 1

    # -------------------------------------------------------------------- output

    def stats(self) -> dict:
        """Counters plus the achieved video rate."""
        result = dict(self.counters)
        if self._video_started_at is not None:
            elapsed = max(self._clock() - self._video_started_at, 1e-6)
            result["video_fps_achieved"] = round(self.counters["frames_rendered"] / elapsed, 2)
            result["render_delay_s"] = round(self._render_delay, 3)
        result["auto_landed"] = self._auto_landed
        return result

    def truth(self):
        """Ground truth now. For tests; nothing on the driver side can call it."""
        with self._lock:
            return self.dynamics.state.copy()

    def _open_logs(self) -> None:
        settings = self.settings
        if settings.truth_path:
            self._truth_file = open(settings.truth_path, "w", newline="\n", encoding="utf-8")
            self._truth_file.write(
                "# simulated ground truth from the Tello emulator; "
                f"seed={settings.seed} battery_start={settings.battery} "
                f"battery_drain={settings.battery_drain}\n"
            )
            self._truth_writer = csv.writer(self._truth_file, lineterminator="\n")
            self._truth_writer.writerow(
                [
                    "wall_time",
                    "flight_time",
                    "x",
                    "y",
                    "z",
                    "yaw",
                    "vx",
                    "vy",
                    "vz",
                    "flying",
                    "motors_cut",
                    "battery",
                    "link_up",
                ]
            )
        if settings.events_path:
            self._events_file = open(settings.events_path, "w", newline="\n", encoding="utf-8")

    def _record_truth(self, now: float) -> None:
        if self._truth_writer is None:
            return
        state = self.dynamics.state
        flight = self.flight_clock(now)
        self._truth_writer.writerow(
            [
                f"{time.time():.3f}",
                "" if flight is None else f"{flight:.3f}",
                f"{state.position[0]:.4f}",
                f"{state.position[1]:.4f}",
                f"{state.position[2]:.4f}",
                f"{state.yaw:.4f}",
                f"{state.velocity[0]:.3f}",
                f"{state.velocity[1]:.3f}",
                f"{state.velocity[2]:.3f}",
                int(state.flying),
                int(state.motors_cut),
                f"{state.battery:.4f}",
                int(self.link_up(now)),
            ]
        )
        self._truth_file.flush()

    def _event(self, kind: str, detail: str = "") -> None:
        if self._events_file is None:
            return
        record = {"wall_time": round(time.time(), 3), "event": kind, "detail": detail}
        with self._lock:
            if self._events_file is not None:
                self._events_file.write(json.dumps(record) + "\n")
                self._events_file.flush()


def _as_number(text: str) -> Optional[float]:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _counts(value: float, maximum: float) -> int:
    return int(np.clip(round(value / maximum * 100.0), -100, 100))


# ------------------------------------------------------------- render workers

_WORKER_RENDERER: Optional[Renderer] = None


def _init_render_worker(room: Room, camera: CameraModel, seed: int) -> None:
    """Build one renderer per worker process.

    SIGINT is ignored here: ROS 2 launch interrupts the whole process group on
    shutdown and the parent, not each worker, decides when rendering stops.
    """
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_RENDERER
    _WORKER_RENDERER = Renderer(
        room, camera, RenderSettings(jpeg_quality=0, seed=seed + os.getpid() % 10007)
    )


def _render_in_worker(pose, angular_velocity, linear_velocity, exposure_time) -> np.ndarray:
    return _render_with(_WORKER_RENDERER, pose, angular_velocity, linear_velocity, exposure_time)


def _render_with(renderer, pose, angular_velocity, linear_velocity, exposure_time) -> np.ndarray:
    image = renderer.render(
        pose,
        angular_velocity=angular_velocity,
        linear_velocity=linear_velocity,
        exposure_time=exposure_time,
    )
    return np.ascontiguousarray(image[:, :, 0])

