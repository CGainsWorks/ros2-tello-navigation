"""Ways of getting frames, telemetry and commands to and from an aircraft.

Three implementations behind one interface: the real Tello over UDP, the
simulator and a replay of a recorded flight. The controller cannot tell them
apart, which is the point: the failsafe logic that will one day fly a real Tello
is the same object the test suite drives through a link loss.
"""
