"""The simulator, which is the whole stand-in for hardware.

Its job is not to be realistic in general. Its job is to be wrong in the ways a
Tello is wrong, so that the failsafe and the estimator meet the failures they will
meet in the air: lag, drift, bias, dropped packets, dead links and a battery that
runs out.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.link.state import decode_state
from tello_core.sim.dynamics import DynamicsSettings, TelloDynamics
from tello_core.sim.link import COMMAND_TIMEOUT, LinkFaults, SimLink, SimSettings
from tello_core.sim.world import Room, wall_texture
from tello_core.types import RcCommand

# ------------------------------------------------------------------- the world


def test_room_has_six_surfaces(room: Room):
    names = {plane.name for plane in room.planes}
    assert {"wall_x0", "wall_x1", "wall_y0", "wall_y1", "floor", "ceiling"} <= names


def test_every_marker_is_printed_somewhere(room: Room, marker_map):
    """A map entry the aircraft can never see is a bug in the map.

    Finding it at build time is much cheaper than finding it as an unexplained
    localisation gap halfway through a mission.
    """
    dark = sum(int((plane.texture < 40).sum()) for plane in room.planes)
    assert dark > 1000 * len(marker_map)


def test_wall_texture_is_reproducible():
    a = wall_texture(2.4, 3.0, 100.0, seed=5)
    b = wall_texture(2.4, 3.0, 100.0, seed=5)
    c = wall_texture(2.4, 3.0, 100.0, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_wall_texture_has_something_to_track():
    """A bare white wall gives Lucas-Kanade nothing and a visual-odometry front
    end tested against one reports a perfect zero and looks excellent."""
    texture = wall_texture(2.4, 3.0, 200.0, seed=5)
    assert texture.std() > 5.0


def test_room_contains():
    room = Room(length_x=4.0, length_y=3.0, height_z=2.4)
    assert room.contains([2.0, 1.5, 1.0])
    assert not room.contains([4.5, 1.5, 1.0])
    assert not room.contains([0.1, 1.5, 1.0], margin=0.5)


# ---------------------------------------------------------------- the dynamics


def _dynamics(config, **kwargs) -> TelloDynamics:
    return TelloDynamics(config, DynamicsSettings(), seed=3, **kwargs)


def test_an_idle_aircraft_stays_on_the_ground(config):
    dynamics = _dynamics(config)
    for _ in range(100):
        dynamics.step(0.01)
    assert not dynamics.state.flying
    assert dynamics.state.position[2] == pytest.approx(DynamicsSettings().resting_height)


def test_resting_height_keeps_the_camera_above_the_floor(config):
    """Zero resting height puts the camera below the floor plane.

    The mount sits a centimetre below the body origin, so with the origin at
    z = 0 the pre-flight localisation check, which is what decides whether the
    aircraft may take off at all, could never pass.
    """
    assert DynamicsSettings().resting_height > abs(config.mount.z)


def test_takeoff_climbs_and_stops(config):
    dynamics = _dynamics(config)
    dynamics.takeoff()
    for _ in range(600):
        dynamics.step(0.01)
    assert dynamics.state.flying
    # A tolerance rather than a target: the aircraft's own take-off is open loop
    # and overshoots, which is what a real one does too.
    assert 0.8 < dynamics.state.position[2] < 1.6
    settled = dynamics.state.position[2]
    for _ in range(200):
        dynamics.step(0.01)
    assert abs(dynamics.state.position[2] - settled) < 0.15


def test_land_reaches_the_ground(config):
    dynamics = _dynamics(config)
    dynamics.takeoff()
    for _ in range(400):
        dynamics.step(0.01)
    dynamics.land()
    for _ in range(800):
        dynamics.step(0.01)
    assert not dynamics.state.flying
    assert dynamics.state.position[2] == pytest.approx(DynamicsSettings().resting_height)


def test_emergency_cuts_the_motors_and_the_aircraft_falls(config):
    """Not a landing. The failsafe reaches for this only when a controlled
    descent is not available."""
    dynamics = _dynamics(config)
    dynamics.takeoff()
    for _ in range(400):
        dynamics.step(0.01)
    height = dynamics.state.position[2]
    dynamics.emergency()
    for _ in range(30):
        dynamics.step(0.01)
    assert dynamics.state.position[2] < height
    assert dynamics.state.velocity[2] < -1.0


def test_a_forward_command_moves_the_aircraft_forward(config):
    dynamics = _dynamics(config, start_yaw=0.0)
    dynamics.takeoff()
    for _ in range(400):
        dynamics.step(0.01)
    start = dynamics.state.position.copy()
    dynamics.set_rc(RcCommand(forward_back=60))
    for _ in range(200):
        dynamics.step(0.01)
    assert dynamics.state.position[0] - start[0] > 0.3


def test_the_command_frame_follows_the_aircraft(config):
    """The same stick input must move the aircraft in the direction it points."""
    dynamics = _dynamics(config, start_yaw=np.pi / 2)
    dynamics.takeoff()
    for _ in range(400):
        dynamics.step(0.01)
    start = dynamics.state.position.copy()
    dynamics.set_rc(RcCommand(forward_back=60))
    for _ in range(200):
        dynamics.step(0.01)
    delta = dynamics.state.position - start
    assert delta[1] > 0.3
    assert abs(delta[0]) < 0.15


def test_velocity_response_is_lagged_not_instant(config):
    """A step command must not produce a step velocity.

    Real phase lag is what the position loop has to be tuned against; without it
    any set of gains looks stable.
    """
    dynamics = _dynamics(config)
    dynamics.takeoff()
    for _ in range(400):
        dynamics.step(0.01)
    dynamics.set_rc(RcCommand(forward_back=100))
    dynamics.step(0.02)
    immediate = float(np.linalg.norm(dynamics.state.velocity[:2]))
    for _ in range(150):
        dynamics.step(0.01)
    settled = float(np.linalg.norm(dynamics.state.velocity[:2]))
    assert immediate < 0.25 * settled


def test_the_aircraft_tilts_to_accelerate(config):
    """The camera swings exactly when the controller asks for the largest
    correction, which is a real coupling and not an artefact."""
    dynamics = _dynamics(config)
    dynamics.takeoff()
    for _ in range(400):
        dynamics.step(0.01)
    dynamics.set_rc(RcCommand(forward_back=100))
    pitches = []
    for _ in range(40):
        dynamics.step(0.01)
        pitches.append(dynamics.state.pitch)
    assert min(pitches) < -0.005


def test_battery_drains_while_flying(config):
    dynamics = _dynamics(config)
    dynamics.takeoff()
    for _ in range(2000):
        dynamics.step(0.01)
    assert dynamics.state.battery < 1.0


def test_a_flat_battery_stops_the_aircraft_flying(config):
    dynamics = _dynamics(config)
    dynamics.state.battery = 0.001
    dynamics.takeoff()
    for _ in range(500):
        dynamics.step(0.01)
    assert dynamics.state.motors_cut


def test_telemetry_reports_gravity_when_level(config):
    dynamics = _dynamics(config)
    state = dynamics.telemetry(0.0)
    assert state.accel[2] > 8.0


def test_telemetry_height_is_zero_on_the_ground(config):
    dynamics = _dynamics(config)
    assert dynamics.telemetry(0.0).height == pytest.approx(0.0, abs=0.05)


def test_accelerometer_bias_is_constant_within_a_flight(config):
    dynamics = _dynamics(config)
    first = dynamics.accel_bias.copy()
    for _ in range(100):
        dynamics.step(0.01)
    assert np.allclose(dynamics.accel_bias, first)


def test_simulation_is_reproducible_from_its_seed(config):
    def run(seed):
        dynamics = TelloDynamics(config, DynamicsSettings(), seed=seed)
        dynamics.takeoff()
        for _ in range(300):
            dynamics.step(0.01)
        return dynamics.state.position.copy()

    assert np.allclose(run(11), run(11))
    assert not np.allclose(run(11), run(12))


# -------------------------------------------------------------------- the link


def _link(room, config, **kwargs) -> SimLink:
    settings = SimSettings(seed=4, **kwargs)
    link = SimLink(room, config, settings=settings, start_position=np.array([1.0, 0.9, 0.03]))
    link.connect()
    link.start_video()
    return link


def test_link_produces_telemetry_and_video(room, config):
    link = _link(room, config)
    link.advance(1.0)
    assert link.poll_state() is not None
    assert link.poll_frame() is not None


def test_telemetry_goes_through_the_real_decoder(room, config):
    """The simulator emits genuine Tello state strings.

    That keeps the unit conversions, the sign flips and the malformed-packet
    handling exercised by every simulated flight, rather than being dead code
    until an aircraft arrives.
    """
    link = _link(room, config, faults=LinkFaults(state_loss=0.0, malformed_state=0.0))
    link.advance(1.0)
    state = link.poll_state()
    assert state is not None and state.valid
    assert decode_state("garbage", 0.0).valid is False


def test_frames_are_rendered_lazily(room, config):
    """Frames the control loop never polls must never be drawn.

    A 20 Hz loop against a 30 Hz camera discards a third of them and drawing
    those was a third of the entire cost of a simulated flight.
    """
    link = _link(room, config)
    link.advance(1.0)
    assert link.frames_rendered == 0
    link.poll_frame()
    assert link.frames_rendered == 1


def test_video_needs_starting(room, config):
    settings = SimSettings(seed=4)
    link = SimLink(room, config, settings=settings, start_position=np.array([1.0, 0.9, 0.03]))
    link.connect()
    link.advance(0.5)
    assert link.poll_frame() is None


def test_an_outage_stops_everything_in_both_directions(room, config):
    link = _link(room, config, faults=LinkFaults(outages=((1.0, 2.0),), state_loss=0.0))
    link.advance(0.9)
    assert link.poll_state() is not None
    link.advance(1.0)
    assert link.poll_state() is None
    assert not link.status().connected


def test_commands_sent_during_an_outage_do_not_arrive(room, config):
    """The property the whole link-loss failsafe rests on."""
    link = _link(room, config, faults=LinkFaults(outages=((0.5, 5.0),), state_loss=0.0))
    link.advance(1.0)
    assert not link.status().connected
    link.takeoff()
    link.advance(2.0)
    assert not link.dynamics.state.flying


def test_the_aircraft_lands_itself_after_the_command_timeout(room, config):
    """A real Tello lands fifteen seconds after the last command it hears.

    Modelled because it changes what a link-loss failsafe has to achieve: the
    aircraft is not going to hover politely until the link comes back.
    """
    link = _link(room, config)
    link.takeoff()
    link.advance(3.0)
    assert link.dynamics.state.flying
    link.advance(COMMAND_TIMEOUT + 8.0)
    assert not link.dynamics.state.flying


def test_a_keepalive_command_prevents_the_timeout(room, config):
    link = _link(room, config)
    link.takeoff()
    link.advance(3.0)
    for _ in range(30):
        link.send_rc(RcCommand.zero())
        link.advance(0.4)
    assert link.dynamics.state.flying


def test_packet_loss_is_counted(room, config):
    link = _link(room, config, faults=LinkFaults(state_loss=0.5, frame_loss=0.0))
    link.advance(5.0)
    while link.poll_state() is not None:
        pass
    assert link.status().packets_dropped > 0


def test_link_status_ages_grow_without_traffic(room, config):
    link = _link(room, config, faults=LinkFaults(outages=((0.5, 10.0),)))
    link.advance(3.0)
    link.poll_state()
    assert link.status().last_state_age > 1.0


def test_the_clock_only_moves_when_advanced(room, config):
    link = _link(room, config)
    assert link.now() == pytest.approx(0.0)
    link.advance(1.5)
    assert link.now() == pytest.approx(1.5, abs=0.05)


def test_truth_is_available_for_evaluation_only(room, config):
    link = _link(room, config)
    link.advance(0.5)
    truth = link.truth()
    assert truth.position.shape == (3,)
