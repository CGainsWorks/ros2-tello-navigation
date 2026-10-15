"""Decoding the Tello's telemetry string into SI units and REP-103 axes.

The aircraft broadcasts a semicolon-separated key:value string on UDP 8890 at
roughly 10 Hz::

    pitch:0;roll:0;yaw:-14;vgx:0;vgy:0;vgz:0;templ:69;temph:71;tof:10;h:0;
    bat:87;baro:-30.61;time:0;agx:-3.00;agy:-4.00;agz:-999.00;

This module is the only place in the repository that knows any of that. Above it
everything is metres, metres per second, radians and REP-103 and the conversion
happens exactly once.

Units and what is actually known about them
--------------------------------------------
DJI's SDK 2.0 document names the fields and gives units for some of them. The
values below are what this implementation assumes:

===========  ========================  ==========================================
Field        Assumed unit              Confidence
===========  ========================  ==========================================
pitch/roll/  degrees                   Documented.
yaw
vgx/vgy/vgz  decimetres per second     Widely used and consistent with observed
                                       ranges. Not stated in the document.
tof, h       centimetres               Documented.
baro         centimetres               Documented as metres in some revisions of
                                       the document and observed as centimetres.
                                       Treated as centimetres, with a range check
                                       that falls back if the value is
                                       implausible.
bat          per cent                  Documented.
templ/temph  degrees Celsius           Documented.
agx/agy/agz  centimetres per second^2  This is the one that matters most and is
                                       the least certain. Some sources give
                                       thousandths of g. The two differ by a
                                       factor of 0.98, which is inside the noise
                                       for a hovering aircraft and is not
                                       something the simulator can settle.
===========  ========================  ==========================================

The accelerometer scaling being uncertain to two per cent is tolerable because
the filter estimates an accelerometer bias and a two per cent scale error on a
signal dominated by the constant 9.81 looks very like a bias. It is still a real
gap and it is in ``docs/HARDWARE.md``.

Axes
----
The Tello reports vgx forward, vgy to the **right** and vgz **down**, an
aerospace-style frame, while everything above this module uses REP-103 with y to
the left and z up. The conversion is a sign flip on y and z and the same applies
to the accelerometer. Getting this wrong produces a system that flies beautifully
in one axis and diverges in another, which is a great deal harder to debug than
one that fails outright, so the sign flips are here, once, with a test that pins
them.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..geometry import wrap_angle
from ..types import TelloState

# Conversions from the aircraft's units to SI.
DECIMETRES_PER_SECOND = 0.1
CENTIMETRES = 0.01
CM_PER_S2 = 0.01

REQUIRED_FIELDS = ("pitch", "roll", "yaw", "vgx", "vgy", "vgz", "bat", "agx", "agy", "agz")


def parse_fields(payload: str) -> Dict[str, str]:
    """Split the raw broadcast into a dictionary of strings.

    Tolerant of a trailing separator, of whitespace and of a truncated final
    field, all of which happen on a lossy link. Not tolerant of a field it cannot
    split, which is returned as absent so that the caller's required-field check
    catches it.
    """
    fields: Dict[str, str] = {}
    for chunk in payload.replace("\r", "").replace("\n", "").split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, _, value = chunk.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def decode_state(
    payload: str,
    timestamp: float,
    previous: Optional[TelloState] = None,
) -> TelloState:
    """Decode one packet. Never raises.

    A malformed packet comes back as a :class:`~tello_core.types.TelloState` with
    ``valid=False`` rather than as an exception, because on a real link a
    malformed packet is an ordinary event and an exception in the receive thread
    is not. The filter ignores invalid packets and the link counts them.

    ``previous`` is used only to differentiate yaw into a yaw rate, since the
    aircraft does not publish one.
    """
    fields = parse_fields(payload)
    if any(key not in fields for key in REQUIRED_FIELDS):
        return _invalid(timestamp)

    try:
        pitch = np.deg2rad(float(fields["pitch"]))
        roll = np.deg2rad(float(fields["roll"]))
        yaw = np.deg2rad(float(fields["yaw"]))

        # Body velocity: aircraft frame is x forward, y right, z down.
        velocity_body = np.array(
            [
                float(fields["vgx"]) * DECIMETRES_PER_SECOND,
                -float(fields["vgy"]) * DECIMETRES_PER_SECOND,
                -float(fields["vgz"]) * DECIMETRES_PER_SECOND,
            ]
        )

        accel = np.array(
            [
                float(fields["agx"]) * CM_PER_S2,
                -float(fields["agy"]) * CM_PER_S2,
                -float(fields["agz"]) * CM_PER_S2,
            ]
        )

        battery = float(fields["bat"]) / 100.0
        height = float(fields.get("tof", fields.get("h", 0.0))) * CENTIMETRES
        barometer = _decode_barometer(fields.get("baro"))
        temperature = _decode_temperature(fields)
    except (TypeError, ValueError):
        return _invalid(timestamp)

    gyro_z = 0.0
    if previous is not None and previous.valid:
        dt = timestamp - previous.timestamp
        if 1e-3 < dt < 1.0:
            gyro_z = wrap_angle(yaw - float(previous.attitude[0])) / dt

    return TelloState(
        timestamp=float(timestamp),
        accel=accel,
        gyro_z=float(gyro_z),
        attitude=np.array([yaw, pitch, roll]),
        velocity_body=velocity_body,
        height=float(height),
        barometer=float(barometer),
        battery=float(np.clip(battery, 0.0, 1.0)),
        temperature=float(temperature),
        valid=True,
    )


def _decode_barometer(raw: Optional[str]) -> float:
    """Barometric altitude in metres, coping with the documented ambiguity.

    Some revisions of the SDK document give this field in metres and the aircraft
    reports centimetres. Values beyond a few hundred are certainly centimetres;
    values under about 20 could be either and are treated as metres, because an
    indoor test area is not 20 m tall and is not 2000 cm tall either.
    """
    if raw is None:
        return float("nan")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value * CENTIMETRES if abs(value) > 20.0 else value


def _decode_temperature(fields: Dict[str, str]) -> float:
    values = []
    for key in ("templ", "temph"):
        try:
            values.append(float(fields[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else float("nan")


def _invalid(timestamp: float) -> TelloState:
    return TelloState(
        timestamp=float(timestamp),
        accel=np.zeros(3),
        gyro_z=0.0,
        attitude=np.zeros(3),
        velocity_body=np.zeros(3),
        height=0.0,
        barometer=float("nan"),
        battery=0.0,
        temperature=float("nan"),
        valid=False,
    )


def encode_state(state: TelloState) -> str:
    """The inverse of :func:`decode_state`, for the simulator and for tests.

    The simulator's link produces genuine Tello-format strings and puts them
    through the same decoder the real driver uses, rather than handing the filter
    a ready-made :class:`~tello_core.types.TelloState`. That means the unit
    conversions, the sign flips and the malformed-packet handling are all
    exercised by every simulated flight, which is where they would otherwise
    never be tested until the day an aircraft arrived.
    """
    yaw, pitch, roll = [float(v) for v in np.asarray(state.attitude).ravel()]
    return (
        f"pitch:{int(round(np.rad2deg(pitch)))};"
        f"roll:{int(round(np.rad2deg(roll)))};"
        f"yaw:{int(round(np.rad2deg(yaw)))};"
        f"vgx:{int(round(state.velocity_body[0] / DECIMETRES_PER_SECOND))};"
        f"vgy:{int(round(-state.velocity_body[1] / DECIMETRES_PER_SECOND))};"
        f"vgz:{int(round(-state.velocity_body[2] / DECIMETRES_PER_SECOND))};"
        f"templ:{int(round(state.temperature - 1))};"
        f"temph:{int(round(state.temperature + 1))};"
        f"tof:{int(round(state.height / CENTIMETRES))};"
        f"h:{int(round(state.height / CENTIMETRES))};"
        f"bat:{int(round(state.battery * 100))};"
        f"baro:{state.barometer / CENTIMETRES:.2f};"
        f"time:{int(state.timestamp)};"
        f"agx:{state.accel[0] / CM_PER_S2:.2f};"
        f"agy:{-state.accel[1] / CM_PER_S2:.2f};"
        f"agz:{-state.accel[2] / CM_PER_S2:.2f};"
    )
