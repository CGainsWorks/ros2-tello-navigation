"""A Tello on the loopback interface: SDK 2.0 over UDP, backed by the simulator.

Where :class:`~tello_core.sim.link.SimLink` replaces the driver,
:class:`TelloEmulator` replaces the aircraft. It binds the command port, answers
SDK commands, broadcasts state strings to port 8890 and streams H.264 to port
11111, so the real :class:`~tello_core.link.udp.UdpLink` and the ROS 2 driver node
talk to it exactly as they would talk to a drone. See ``docs/EMULATOR.md``.
"""

from .server import EmulatorFaults, EmulatorSettings, TelloEmulator

__all__ = ["EmulatorFaults", "EmulatorSettings", "TelloEmulator"]
