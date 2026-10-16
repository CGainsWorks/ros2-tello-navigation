"""The real driver: Tello SDK 2.0 over UDP.

**Never run against an aircraft.** This project has no Tello. Everything here is
written from DJI's published SDK 2.0 document and from the protocol as it is
widely described, it is exercised against a loopback socket and against the
emulator in the test suite and it has never exchanged a packet with a real one.
``docs/HARDWARE.md`` lists what would have to be checked before it is trusted
and the first item on that list is this file.

Why not djitellopy
------------------
``djitellopy`` is the obvious choice and it is a good library. It is not used
here for one reason: it presents the state packet as a set of typed accessors and
does not expose the raw broadcast with a receive timestamp. This estimator needs
the timestamp, because the interval between packets is what the prediction step
integrates over and it needs the raw string, because a malformed packet has to
be distinguishable from a missing one. Rebuilding those on top of a library that
has discarded them is more work than speaking the protocol, which is three
sockets and a text format.

``djitellopy`` is still listed in ``requirements-optional.txt`` as the fastest way
to check that an aircraft is reachable before running any of this against it.

Protocol
--------
Three UDP sockets, all to 192.168.10.1 while joined to the aircraft's own access
point:

* **8889**, commands and their responses. Text. ``command`` first, then anything
  else. Responses are ``ok`` or ``error ...``, except for query commands which
  answer with a value.
* **8890**, the state broadcast, roughly 10 Hz, decoded by
  :mod:`tello_core.link.state`.
* **11111**, the H.264 video stream, after ``streamon``.

Video decoding is the part that cannot be done in the standard library. The
stream is raw H.264 in UDP datagrams with no container, so it needs a decoder.
This class takes one as a callable, so the transport does not import a codec;
:class:`tello_core.link.video.H264Decoder` is the one this repository ships,
FFmpeg's H.264 parser and decoder through PyAV, and it is tested against the H.264
that the emulator in :mod:`tello_core.emulator` encodes with libx264.

The emulator is also what this driver is exercised against end to end: the same
three ports, the same text protocol, state strings and an H.264 stream, on the
loopback interface. That is evidence the driver and the emulator agree with each
other. Neither has met an aircraft.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..types import Frame, RcCommand, TelloState
from .base import LinkStatus, TelloLink
from .state import decode_state

TELLO_ADDRESS = ("192.168.10.1", 8889)
STATE_PORT = 8890
VIDEO_PORT = 11111


@dataclass(frozen=True)
class UdpSettings:
    """Addresses, ports and timeouts."""

    address: tuple = TELLO_ADDRESS
    """Where to send commands. The aircraft's own address and port."""

    local_command_port: int = 8889
    """The local port the command socket binds to.

    Separate from ``address[1]`` even though both are 8889 against a real
    aircraft and the reason is testability rather than pedantry. The Tello
    replies to port 8889 on the host, so the driver has to bind the same number it
    sends to; binding ``address[1]`` directly works against hardware and makes the
    driver impossible to point at a fake aircraft on loopback, because the fake is
    already bound to that port. The loopback tests in ``tests/test_udp_link.py``
    found this immediately, which is most of the value of having written them.

    Zero asks the operating system for any free port, which is what a test wants
    and what no aircraft would tolerate."""

    state_port: int = STATE_PORT
    video_port: int = VIDEO_PORT
    command_timeout: float = 0.5
    """Seconds to wait for a response to a command. Short: a command that has not
    been answered in half a second on a link this simple is not going to be."""

    keepalive_interval: float = 4.0
    """Seconds between keepalive commands when nothing else is being sent. The
    aircraft lands itself after fifteen seconds of silence, so something has to
    keep talking while it hovers. The failsafe deliberately stops this on a link
    loss, see :mod:`tello_core.control.failsafe`."""

    receive_buffer: int = 4096


class UdpLink(TelloLink):
    """Speaks Tello SDK 2.0 to a real aircraft.

    ``video_decoder`` is a callable taking a bytes datagram and returning either
    a decoded ``(H, W, 3)`` BGR image or ``None`` while it accumulates. Supplying
    one is the caller's job; see the module docstring.
    """

    def __init__(
        self,
        settings: Optional[UdpSettings] = None,
        video_decoder: Optional[Callable[[bytes], Optional[np.ndarray]]] = None,
    ) -> None:
        self.settings = settings or UdpSettings()
        self.video_decoder = video_decoder

        self._command_socket: Optional[socket.socket] = None
        self._state_socket: Optional[socket.socket] = None
        self._video_socket: Optional[socket.socket] = None

        self._running = threading.Event()
        self._threads: list = []
        self._lock = threading.Lock()

        self._latest_state: Optional[TelloState] = None
        self._previous_state: Optional[TelloState] = None
        self._latest_frame: Optional[Frame] = None
        self._frame_index = 0

        self._last_state_time = float("-inf")
        self._last_frame_time = float("-inf")
        self._last_command_time = float("-inf")
        self._packets_received = 0
        self._packets_dropped = 0
        self._commands_sent = 0
        self._detail = ""

    # ------------------------------------------------------------------ lifecycle

    def connect(self) -> bool:
        """Open the sockets and enter SDK mode."""
        try:
            self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._command_socket.settimeout(self.settings.command_timeout)
            self._command_socket.bind(("", self.settings.local_command_port))

            self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._state_socket.settimeout(0.5)
            self._state_socket.bind(("", self.settings.state_port))
        except OSError as error:
            self._detail = f"could not open sockets: {error}"
            self.close()
            return False

        self._running.set()
        self._start_thread(self._state_loop, "tello-state")
        self._start_thread(self._keepalive_loop, "tello-keepalive")

        response = self._command("command")
        if response is None or not response.lower().startswith("ok"):
            self._detail = f"aircraft did not enter SDK mode: {response!r}"
            self.close()
            return False
        self._detail = "connected"
        return True

    def close(self) -> None:
        self._running.clear()
        for thread in self._threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._threads = []
        for name in ("_command_socket", "_state_socket", "_video_socket"):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, name, None)

    def start_video(self) -> bool:
        if self.video_decoder is None:
            self._detail = (
                "no video decoder supplied. The Tello streams raw H.264 with no "
                "container, so a decoder has to be passed in, for example "
                "tello_core.link.video.H264Decoder."
            )
            return False
        # The socket is bound before streamon, not after. The stream starts the
        # moment the aircraft accepts the command and its first datagrams carry
        # the parameter sets and the keyframe the decoder needs; binding late
        # means waiting a whole keyframe interval for the next one.
        try:
            self._video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._video_socket.settimeout(0.5)
            # A keyframe arrives as a burst of a few dozen datagrams and the
            # default receive buffer on some systems is smaller than that.
            try:
                self._video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
            except OSError:
                pass
            self._video_socket.bind(("", self.settings.video_port))
        except OSError as error:
            self._detail = f"could not open the video socket: {error}"
            return False
        response = self._command("streamon")
        if response is None or not response.lower().startswith("ok"):
            self._detail = f"streamon refused: {response!r}"
            self._video_socket.close()
            self._video_socket = None
            return False
        self._start_thread(self._video_loop, "tello-video")
        return True

    def _start_thread(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    # -------------------------------------------------------------------- threads

    def _state_loop(self) -> None:
        while self._running.is_set():
            try:
                payload, _ = self._state_socket.recvfrom(self.settings.receive_buffer)
            except (socket.timeout, OSError):
                continue
            now = time.monotonic()
            state = decode_state(payload.decode("ascii", "replace"), now, self._previous_state)
            with self._lock:
                if state.valid:
                    self._previous_state = state
                    self._latest_state = state
                    self._last_state_time = now
                    self._packets_received += 1
                else:
                    self._packets_dropped += 1

    def _video_loop(self) -> None:
        while self._running.is_set():
            try:
                packet, _ = self._video_socket.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            try:
                image = self.video_decoder(packet)
            except Exception:  # noqa: BLE001 - a decoder fault must not kill the link
                continue
            if image is None:
                continue
            now = time.monotonic()
            with self._lock:
                self._latest_frame = Frame(timestamp=now, image=image, index=self._frame_index)
                self._frame_index += 1
                self._last_frame_time = now

    def _keepalive_loop(self) -> None:
        """Send a harmless command when nothing else has been sent recently.

        Only when nothing else has. This must never fight the failsafe: on a link
        loss the failsafe deliberately falls silent so the aircraft's own
        fifteen-second timer can land it and a keepalive thread that carries on
        regardless would undo exactly that. It is gated on
        ``self._last_command_time``, which only real traffic updates.
        """
        while self._running.is_set():
            time.sleep(0.5)
            now = time.monotonic()
            if now - self._last_command_time < self.settings.keepalive_interval:
                continue
            # Gated on whether the aircraft has *ever* been heard from, not on
            # whether an unread packet is sitting in the buffer. Checking the
            # buffer was the first version and it was wrong: `poll_state`
            # consumes, so a caller that reads its telemetry, which is every
            # caller, left the keepalive permanently disabled. The loopback test
            # caught it; an aircraft would have caught it by landing itself
            # fifteen seconds into every hover.
            if not np.isfinite(self._last_state_time):
                continue
            self._send("rc 0 0 0 0", expect_response=False)

    # ------------------------------------------------------------------- polling

    def poll_state(self) -> Optional[TelloState]:
        with self._lock:
            state, self._latest_state = self._latest_state, None
        return state

    def poll_frame(self) -> Optional[Frame]:
        with self._lock:
            frame, self._latest_frame = self._latest_frame, None
        return frame

    # ------------------------------------------------------------------ commands

    def send_rc(self, command: RcCommand) -> None:
        """Send a stick command.

        The sign flip on the lateral channel happens here and only here: this
        stack's body frame has y to the left, the SDK's first channel is positive
        to the right.
        """
        self._send(
            f"rc {-command.left_right} {command.forward_back} "
            f"{command.up_down} {command.yaw_rate}",
            expect_response=False,
        )

    def takeoff(self) -> None:
        self._send("takeoff", expect_response=False)

    def land(self) -> None:
        self._send("land", expect_response=False)

    def emergency(self) -> None:
        self._send("emergency", expect_response=False)

    def _command(self, text: str) -> Optional[str]:
        """Send and wait for the reply. Used only during connection setup."""
        if self._command_socket is None:
            return None
        # Discard replies nobody waited for. takeoff, land and the rc keepalive
        # are sent without reading the answer and the aircraft still sends one,
        # so without this the next command that does wait reads a stale "ok"
        # belonging to an earlier command, for example streamon sent after a
        # takeoff reporting the takeoff's reply as its own.
        self._drain_replies()
        self._send(text, expect_response=False)
        try:
            payload, _ = self._command_socket.recvfrom(self.settings.receive_buffer)
        except (socket.timeout, OSError):
            return None
        return payload.decode("ascii", "replace").strip()

    def _drain_replies(self) -> None:
        handle = self._command_socket
        if handle is None:
            return
        handle.setblocking(False)
        try:
            while True:
                handle.recvfrom(self.settings.receive_buffer)
        except (BlockingIOError, OSError):
            pass
        finally:
            handle.settimeout(self.settings.command_timeout)

    def _send(self, text: str, expect_response: bool = False) -> None:
        if self._command_socket is None:
            return
        try:
            self._command_socket.sendto(text.encode("ascii"), self.settings.address)
        except OSError:
            self._packets_dropped += 1
            return
        self._commands_sent += 1
        self._last_command_time = time.monotonic()
        if expect_response:
            try:
                self._command_socket.recvfrom(self.settings.receive_buffer)
            except (socket.timeout, OSError):
                pass

    # -------------------------------------------------------------------- status

    def status(self) -> LinkStatus:
        now = time.monotonic()
        with self._lock:
            return LinkStatus(
                connected=self._running.is_set()
                and (now - self._last_state_time) < 2.0,
                last_state_age=now - self._last_state_time,
                last_frame_age=now - self._last_frame_time,
                packets_received=self._packets_received,
                packets_dropped=self._packets_dropped,
                commands_sent=self._commands_sent,
                detail=self._detail,
            )
