"""Decoding the Tello's telemetry string.

The only place in the repository that knows about centimetres, decimetres per
second, degrees and the aircraft's own axis convention. Everything above it is SI
and REP-103, so a sign error here is a sign error everywhere and it would show
up as an estimator that works in one axis and diverges in another rather than as
anything that looks like a parsing bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.link.state import decode_state, encode_state, parse_fields
from tello_core.types import TelloState

SAMPLE = (
    "pitch:2;roll:-1;yaw:-14;vgx:3;vgy:-2;vgz:1;templ:69;temph:71;tof:120;h:118;"
    "bat:87;baro:-30.61;time:42;agx:-3.00;agy:-4.00;agz:-999.00;"
)


def test_parse_fields_handles_the_trailing_separator():
    fields = parse_fields(SAMPLE)
    assert fields["bat"] == "87"
    assert fields["agz"] == "-999.00"
    assert "" not in fields


def test_parse_fields_ignores_junk_between_separators():
    fields = parse_fields("a:1;;garbage;b:2;")
    assert fields == {"a": "1", "b": "2"}


def test_decode_converts_to_si():
    state = decode_state(SAMPLE, timestamp=10.0)
    assert state.valid
    assert state.battery == pytest.approx(0.87)
    assert state.height == pytest.approx(1.20)
    assert state.temperature == pytest.approx(70.0)
    assert np.rad2deg(state.attitude[0]) == pytest.approx(-14.0)
    assert np.rad2deg(state.attitude[1]) == pytest.approx(2.0)
    assert np.rad2deg(state.attitude[2]) == pytest.approx(-1.0)


def test_decode_flips_the_lateral_and_vertical_axes():
    """The aircraft reports y right and z down; this stack is y left and z up.

    Two sign flips, in one place. Getting either wrong produces a system that
    flies correctly forwards and diverges sideways.
    """
    state = decode_state(SAMPLE, timestamp=10.0)
    assert state.velocity_body[0] == pytest.approx(0.3)
    assert state.velocity_body[1] == pytest.approx(0.2)
    assert state.velocity_body[2] == pytest.approx(-0.1)
    assert state.accel[0] == pytest.approx(-0.03)
    assert state.accel[1] == pytest.approx(0.04)
    assert state.accel[2] == pytest.approx(9.99)


def test_decode_gives_gravity_upwards_when_level():
    """A level, stationary aircraft measures a specific force of +g on body z.

    The accelerometer feels the normal force, not gravity, so a Tello sitting on
    a table reports roughly +9.8 on the axis pointing up. If this came out
    negative the filter would double gravity instead of removing it and the
    estimate would accelerate into the floor at 2 g.
    """
    state = decode_state(SAMPLE, timestamp=10.0)
    assert state.accel[2] > 9.0


def test_gyro_rate_is_differentiated_from_yaw():
    first = decode_state(SAMPLE, timestamp=10.0)
    second = decode_state(SAMPLE.replace("yaw:-14", "yaw:-4"), timestamp=10.1, previous=first)
    assert np.rad2deg(second.gyro_z) == pytest.approx(100.0, rel=0.01)


def test_gyro_rate_wraps_the_short_way():
    """Crossing the wrap point must not read as a 355 degree turn."""
    first = decode_state(SAMPLE.replace("yaw:-14", "yaw:179"), timestamp=10.0)
    second = decode_state(SAMPLE.replace("yaw:-14", "yaw:-179"), timestamp=10.1, previous=first)
    assert np.rad2deg(second.gyro_z) == pytest.approx(20.0, rel=0.01)


def test_gyro_rate_is_zero_without_a_previous_packet():
    assert decode_state(SAMPLE, timestamp=10.0).gyro_z == 0.0


def test_gyro_rate_ignores_an_implausible_interval():
    first = decode_state(SAMPLE, timestamp=10.0)
    stale = decode_state(SAMPLE.replace("yaw:-14", "yaw:100"), timestamp=99.0, previous=first)
    assert stale.gyro_z == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "garbage",
        "pitch:0;roll:0;",
        "pitch:x;roll:0;yaw:0;vgx:0;vgy:0;vgz:0;bat:0;agx:0;agy:0;agz:0;",
    ],
)
def test_malformed_packets_are_invalid_not_exceptions(payload):
    """A malformed packet is an ordinary event on a real link.

    Raising in the receive thread would take the link down; returning an invalid
    state lets the filter ignore it and the link count it.
    """
    state = decode_state(payload, timestamp=1.0)
    assert isinstance(state, TelloState)
    assert not state.valid


def test_truncated_packet_is_invalid():
    assert not decode_state(SAMPLE[: len(SAMPLE) // 2], timestamp=1.0).valid


def test_barometer_ambiguity_is_handled_both_ways():
    """The SDK document has been published with this field in both units."""
    centimetres = decode_state(SAMPLE.replace("baro:-30.61", "baro:-3061.0"), timestamp=1.0)
    metres = decode_state(SAMPLE.replace("baro:-30.61", "baro:-3.5"), timestamp=1.0)
    assert centimetres.barometer == pytest.approx(-30.61)
    assert metres.barometer == pytest.approx(-3.5)


def test_encode_decode_round_trip():
    """The simulator emits real Tello strings and parses them with this decoder.

    That is what keeps the unit conversions and the sign flips exercised by every
    simulated flight, rather than being dead code until an aircraft arrives.
    """
    original = decode_state(SAMPLE, timestamp=10.0)
    round_tripped = decode_state(encode_state(original), timestamp=10.0)
    assert round_tripped.valid
    assert np.allclose(round_tripped.velocity_body, original.velocity_body, atol=0.06)
    assert np.allclose(round_tripped.accel, original.accel, atol=0.02)
    assert round_tripped.battery == pytest.approx(original.battery, abs=0.01)
    assert round_tripped.height == pytest.approx(original.height, abs=0.01)
    assert np.allclose(round_tripped.attitude, original.attitude, atol=np.deg2rad(1.0))
