"""The real driver, against a fake aircraft on the loopback interface.

This is the closest thing to a hardware test that exists in this repository and
it is important to be precise about what it does and does not show.

**It does show** that the driver opens its sockets, enters SDK mode, parses the
state broadcast into SI units and REP-103 axes, serialises stick commands into the
SDK's text format with the correct channel order and the correct sign flip and
tears its threads down without leaking a socket. Those are real properties of the
code and every one of them is a way a driver can be silently wrong.

**It does not show** that a DJI Tello accepts any of it. The protocol here is
written from DJI's published SDK 2.0 document; the aircraft's actual tolerance for
timing, for malformed commands, for command order and for the ``rc`` rate is
unknown. See ``docs/HARDWARE.md``.

The fake aircraft is a plain UDP socket that answers ``ok`` and broadcasts state
strings, which is all the protocol needs it to be.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from tello_core.link.state import encode_state
from tello_core.link.udp import UdpLink, UdpSettings
from tello_core.types import RcCommand, TelloState


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class FakeTello:
    """A UDP socket that behaves enough like a Tello to drive the real driver.

    Answers every command with ``ok`` unless told otherwise, records what it was
    sent and broadcasts state strings on demand.
    """

    def __init__(self, reply: bytes = b"ok") -> None:
        self.command_port = _free_port()
        self.reply = reply
        self.received: list = []
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.2)
        self._socket.bind(("127.0.0.1", self.command_port))
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._last_sender = None

    def _serve(self) -> None:
        while self._running.is_set():
            try:
                payload, sender = self._socket.recvfrom(4096)
            except (socket.timeout, OSError):
                continue
            self.received.append(payload.decode("ascii", "replace"))
            self._last_sender = sender
            if self.reply:
                try:
                    self._socket.sendto(self.reply, sender)
                except OSError:
                    pass

    def broadcast(self, payload: str, port: int) -> None:
        """Send a state string to the driver's telemetry socket."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(payload.encode("ascii"), ("127.0.0.1", port))

    def close(self) -> None:
        self._running.clear()
        self._thread.join(timeout=1.0)
        self._socket.close()

    def wait_for(self, predicate, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(predicate(message) for message in list(self.received)):
                return True
            time.sleep(0.01)
        return False


@pytest.fixture
def aircraft():
    fake = FakeTello()
    yield fake
    fake.close()


@pytest.fixture
def link(aircraft):
    settings = UdpSettings(
        address=("127.0.0.1", aircraft.command_port),
        local_command_port=_free_port(),
        state_port=_free_port(),
        video_port=_free_port(),
        command_timeout=0.5,
        keepalive_interval=1e6,  # off: the keepalive is tested on its own
    )
    handle = UdpLink(settings)
    yield handle
    handle.close()


def _state_string() -> str:
    state = TelloState(
        timestamp=0.0,
        accel=np.array([0.1, -0.2, 9.8]),
        gyro_z=0.0,
        attitude=np.array([np.deg2rad(-14.0), np.deg2rad(2.0), np.deg2rad(-1.0)]),
        velocity_body=np.array([0.3, 0.2, -0.1]),
        height=1.20,
        barometer=1.2,
        battery=0.87,
        temperature=70.0,
    )
    return encode_state(state)


# ------------------------------------------------------------------ connection


def test_connect_enters_sdk_mode(link, aircraft):
    assert link.connect()
    assert aircraft.wait_for(lambda m: m.strip() == "command")
    assert link.status().detail == "connected"


def test_connect_fails_when_the_aircraft_does_not_answer(aircraft):
    aircraft.reply = b""
    settings = UdpSettings(
        address=("127.0.0.1", aircraft.command_port),
        local_command_port=_free_port(),
        state_port=_free_port(),
        video_port=_free_port(),
        command_timeout=0.2,
        keepalive_interval=1e6,
    )
    handle = UdpLink(settings)
    try:
        assert not handle.connect()
        assert "SDK mode" in handle.status().detail
    finally:
        handle.close()


def test_connect_fails_on_a_refused_reply(aircraft):
    aircraft.reply = b"error"
    settings = UdpSettings(
        address=("127.0.0.1", aircraft.command_port),
        local_command_port=_free_port(),
        state_port=_free_port(),
        video_port=_free_port(),
        command_timeout=0.2,
        keepalive_interval=1e6,
    )
    handle = UdpLink(settings)
    try:
        assert not handle.connect()
    finally:
        handle.close()


def test_close_is_idempotent(link):
    link.connect()
    link.close()
    link.close()


# ------------------------------------------------------------------- telemetry


def test_a_broadcast_state_arrives_decoded(link, aircraft):
    assert link.connect()
    aircraft.broadcast(_state_string(), link.settings.state_port)

    deadline = time.monotonic() + 1.0
    state = None
    while state is None and time.monotonic() < deadline:
        state = link.poll_state()
        time.sleep(0.01)

    assert state is not None and state.valid
    assert state.battery == pytest.approx(0.87, abs=0.01)
    assert state.height == pytest.approx(1.20, abs=0.01)
    # The axis flips, which are the thing most worth checking here.
    assert state.velocity_body[1] == pytest.approx(0.2, abs=0.06)
    assert state.accel[2] > 9.0


def test_polling_twice_returns_the_packet_once(link, aircraft):
    """``poll_state`` consumes. A stale packet returned twice would look to the
    filter like a fresh measurement and would be integrated twice."""
    assert link.connect()
    aircraft.broadcast(_state_string(), link.settings.state_port)
    deadline = time.monotonic() + 1.0
    while link.poll_state() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert link.poll_state() is None


def test_a_malformed_broadcast_is_counted_not_raised(link, aircraft):
    assert link.connect()
    aircraft.broadcast("not a state packet at all", link.settings.state_port)
    time.sleep(0.3)
    assert link.poll_state() is None
    assert link.status().packets_dropped >= 1


def test_link_status_reports_the_telemetry_age(link, aircraft):
    assert link.connect()
    assert link.status().last_state_age > 1.0
    aircraft.broadcast(_state_string(), link.settings.state_port)
    deadline = time.monotonic() + 1.0
    while link.poll_state() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert link.status().last_state_age < 1.0
    assert link.status().connected


# -------------------------------------------------------------------- commands


def test_rc_is_sent_in_the_sdk_format_with_the_lateral_sign_flipped(link, aircraft):
    """This stack's body frame has y to the left; the SDK's channel is positive
    to the right. The flip happens in the driver and nowhere else, so if it is
    wrong the aircraft steers away from every marker it sees."""
    assert link.connect()
    link.send_rc(RcCommand(left_right=20, forward_back=30, up_down=-10, yaw_rate=5))
    assert aircraft.wait_for(lambda m: m.strip() == "rc -20 30 -10 5")


def test_zero_rc_is_still_sent(link, aircraft):
    """A zero command is a command. It is what keeps the aircraft's own
    fifteen-second dead-man timer from firing during a hover."""
    assert link.connect()
    link.send_rc(RcCommand.zero())
    assert aircraft.wait_for(lambda m: m.strip() == "rc 0 0 0 0")


@pytest.mark.parametrize(
    "method,expected", [("takeoff", "takeoff"), ("land", "land"), ("emergency", "emergency")]
)
def test_the_flight_commands_are_sent_verbatim(link, aircraft, method, expected):
    assert link.connect()
    getattr(link, method)()
    assert aircraft.wait_for(lambda m: m.strip() == expected)


def test_commands_are_counted(link, aircraft):
    assert link.connect()
    before = link.status().commands_sent
    for _ in range(5):
        link.send_rc(RcCommand.zero())
    assert link.status().commands_sent >= before + 5


def test_start_video_without_a_decoder_is_refused(link, aircraft):
    """The Tello streams raw H.264 with no container, so a decoder has to be
    supplied. Returning False with an explanation beats accepting the call and
    producing no frames."""
    assert link.connect()
    assert not link.start_video()
    assert "decoder" in link.status().detail


def test_start_video_with_a_decoder_sends_streamon(aircraft):
    def decoder(packet):
        del packet
        return None

    settings = UdpSettings(
        address=("127.0.0.1", aircraft.command_port),
        local_command_port=_free_port(),
        state_port=_free_port(),
        video_port=_free_port(),
        command_timeout=0.3,
        keepalive_interval=1e6,
    )
    handle = UdpLink(settings, video_decoder=decoder)
    try:
        assert handle.connect()
        assert handle.start_video()
        assert aircraft.wait_for(lambda m: m.strip() == "streamon")
    finally:
        handle.close()


# ------------------------------------------------------------------- keepalive


def test_the_keepalive_only_fires_when_nothing_else_is_talking(aircraft):
    """The keepalive must never fight the failsafe.

    On a link loss the failsafe deliberately falls silent so the aircraft's own
    dead-man timer can land it. A keepalive thread that carried on regardless
    would undo exactly that, so it is gated on the time of the last real command.
    """
    settings = UdpSettings(
        address=("127.0.0.1", aircraft.command_port),
        local_command_port=_free_port(),
        state_port=_free_port(),
        video_port=_free_port(),
        command_timeout=0.3,
        keepalive_interval=0.3,
    )
    handle = UdpLink(settings)
    try:
        assert handle.connect()
        # No telemetry has arrived, so the keepalive must not start: there is no
        # aircraft to keep alive.
        time.sleep(0.9)
        assert not aircraft.wait_for(lambda m: m.strip() == "rc 0 0 0 0", timeout=0.1)

        aircraft.broadcast(_state_string(), settings.state_port)
        deadline = time.monotonic() + 1.0
        while handle.poll_state() is None and time.monotonic() < deadline:
            time.sleep(0.01)

        assert aircraft.wait_for(lambda m: m.strip() == "rc 0 0 0 0", timeout=2.0)
    finally:
        handle.close()
