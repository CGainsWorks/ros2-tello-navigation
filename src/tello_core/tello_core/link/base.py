"""The one interface everything above the driver talks to.

Two implementations exist. :class:`~tello_core.sim.link.SimLink` drives the
simulator and is what every result in this repository was produced with;
:class:`~tello_core.link.udp.UdpLink` speaks Tello SDK 2.0 to a real aircraft and
has never been run against one, for the reason in ``docs/HARDWARE.md``.

A third, replaying a recorded flight, is the obvious addition the moment there is
real footage to replay and this interface is shaped for it. It is not here
because there is no recording: writing a player for a file format nobody has
produced would be code presented as working that has never worked.

The mission runner, the controller and the failsafe machine cannot tell which
implementation they have and that is the entire point: the failsafe logic that
will one day fly a real Tello is the same object the test suite drives through a
link loss, a flat battery and a lost fix.

The interface is deliberately narrow and deliberately non-blocking. Every method
either returns immediately or returns ``None`` to mean "nothing available yet".
There is no method that waits, because a waiting call inside a control loop is how
a failsafe ends up unable to run at the moment it is needed.

Command semantics
-----------------
``send_rc`` is fire-and-forget, as it is on the aircraft: the Tello does not
acknowledge rc commands. ``takeoff``, ``land`` and ``emergency`` do get a response
on the real aircraft but waiting for it would block, so they return immediately
and the caller watches the telemetry to find out what happened. That is not a
simplification, it is what the aircraft forces: a Tello that has stopped
responding will not answer ``land`` either and a controller that waits for the
answer is a controller that has stopped flying.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from ..types import Frame, RcCommand, TelloState


@dataclass(frozen=True)
class LinkStatus:
    """What the link itself knows about its own health.

    Distinct from the aircraft's state. A link that has heard nothing for two
    seconds is a link problem; a battery at 12 per cent is an aircraft problem;
    and the failsafe machine handles them differently, so they are reported
    separately.
    """

    connected: bool
    """False before ``connect`` succeeds and after the link gives up."""

    last_state_age: float
    """Seconds since the last telemetry packet, or ``inf``."""

    last_frame_age: float
    """Seconds since the last video frame, or ``inf``."""

    packets_received: int = 0
    packets_dropped: int = 0
    commands_sent: int = 0
    detail: str = ""


class TelloLink(abc.ABC):
    """Abstract transport to an aircraft, real or otherwise."""

    @abc.abstractmethod
    def connect(self) -> bool:
        """Enter SDK mode and start the telemetry stream. Returns success."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release sockets, threads and any other resource. Idempotent."""

    @abc.abstractmethod
    def start_video(self) -> bool:
        """Begin the video stream."""

    @abc.abstractmethod
    def poll_state(self) -> Optional[TelloState]:
        """The newest unread telemetry packet, or ``None``."""

    @abc.abstractmethod
    def poll_frame(self) -> Optional[Frame]:
        """The newest unread video frame, or ``None``."""

    @abc.abstractmethod
    def send_rc(self, command: RcCommand) -> None:
        """Send a stick command. Fire and forget."""

    @abc.abstractmethod
    def takeoff(self) -> None:
        """Ask the aircraft to take off to its default height."""

    @abc.abstractmethod
    def land(self) -> None:
        """Ask for a controlled descent and motor stop."""

    @abc.abstractmethod
    def emergency(self) -> None:
        """Cut the motors immediately.

        This is not a landing. It is the last resort for an aircraft that is
        about to do something worse than fall and on a 80 gram Tello falling
        from 1.5 m it is a survivable outcome. The failsafe machine reaches for
        it only when a controlled descent is not available.
        """

    @abc.abstractmethod
    def status(self) -> LinkStatus:
        """Health of the link itself."""

    # ------------------------------------------------------------------- clock
    #
    # The mission runner needs two things from time: what it is now and a way to
    # wait until the next control cycle. Both come from the link rather than from
    # ``time`` directly and that one indirection is what lets the identical
    # mission runner fly a real aircraft in real time and a simulated one in
    # whatever fraction of it the machine can manage. A runner that called
    # ``time.monotonic`` and ``time.sleep`` itself would need a parallel
    # simulated version and the two would drift apart.

    def now(self) -> float:
        """The link's notion of the current time, in seconds."""
        import time as _time

        return _time.monotonic()

    def advance(self, dt: float) -> None:
        """Wait, or step the simulation, until ``dt`` seconds have passed."""
        import time as _time

        if dt > 0:
            _time.sleep(dt)

    # Context-manager sugar, so a test or a script cannot leak a socket by
    # taking an early return out of the middle of a flight.
    def __enter__(self) -> "TelloLink":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
