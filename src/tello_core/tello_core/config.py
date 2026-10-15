"""Everything about the aircraft that is a number rather than an algorithm.

Loaded from YAML so that a change of airframe, camera mount or speed setting is a
configuration change rather than a code change and so that the value a result
was produced with is recorded alongside it. ``src/tello_bringup/config`` holds
the shipped files; the ROS 2 nodes read the same ones through their parameter
files.

Every constant here is either taken from DJI's published SDK documentation or
derived from it and each is marked with which. Where a figure could not be
verified without an aircraft, it says so and ``docs/HARDWARE.md`` lists the
whole set in one place. That matters because several of them, the camera
extrinsics and the rc-channel scaling in particular, are the kind of thing that
is nearly right by construction and exactly right only after somebody has
measured it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .geometry import R_BC, euler_to_matrix, make_transform


@dataclass(frozen=True)
class CameraMount:
    """Where the forward camera sits relative to the body origin.

    The body origin is taken at the centre of the airframe, level with the
    propeller plane. The Tello's forward camera is in the nose, so the offset is
    a few centimetres forward and slightly below.

    UNVERIFIED. These are read off DJI's published dimensions, not measured on an
    aircraft. A 1 cm error in the offset is a 1 cm bias in every position
    estimate, which is comparable with the whole error budget, so this is among
    the first things to measure when hardware arrives. ``docs/HARDWARE.md``
    describes the procedure.
    """

    x: float = 0.032
    """Metres forward of the body origin."""

    y: float = 0.0
    z: float = -0.012
    """Metres above the body origin, so negative is below."""

    pitch: float = 0.0
    """Radians. Positive tilts the optical axis upwards. The Tello's forward
    camera is nominally level with the airframe but the airframe pitches
    forward to translate, which is handled in the estimator rather than here."""

    yaw: float = 0.0
    roll: float = 0.0

    def transform(self) -> np.ndarray:
        """4x4 body-to-camera transform, ``T_bc``.

        Composes the fixed body-to-camera axis permutation from
        :mod:`tello_core.geometry` with the mount's own small misalignment, so
        that a mount rotation is expressed in the sane frame (body axes) rather
        than in OpenCV's.
        """
        misalignment = euler_to_matrix(self.yaw, self.pitch, self.roll)
        return make_transform(misalignment @ R_BC, np.array([self.x, self.y, self.z]))


@dataclass(frozen=True)
class RcScaling:
    """Converting a velocity in m/s into the SDK's ``rc`` channel units.

    The Tello's ``rc a b c d`` takes four integers in [-100, 100]. What 100 means
    depends on the aircraft's speed setting, which the SDK's ``speed`` command
    sets between 10 and 100 cm/s. The figures below are for the default setting
    and are the published maxima, not measured ones.

    UNVERIFIED and the one place where being wrong is least dangerous and most
    annoying: an error here shows up as a proportional-gain error, so the
    controller is detuned rather than unstable and retuning would hide it.
    ``docs/HARDWARE.md`` has the procedure: fly a fixed stick input past a marker
    pair and solve for the scaling from the resulting velocity.
    """

    horizontal_max: float = 1.0
    """m/s at full stick, forward-back and left-right."""

    vertical_max: float = 0.8
    """m/s at full stick, up-down."""

    yaw_rate_max: float = np.deg2rad(100.0)
    """rad/s at full stick."""

    deadband: int = 3
    """Channel counts below which the aircraft does nothing. Commands inside the
    deadband are snapped to zero rather than sent, so that a small steady error
    does not sit there heating the motors with a command that has no effect."""


@dataclass(frozen=True)
class BatteryPolicy:
    """When to give up and land. See ``docs/FAILSAFE.md`` for the reasoning."""

    low: float = 0.25
    """Fraction at which the mission is abandoned and the aircraft returns."""

    critical: float = 0.15
    """Fraction at which it lands where it is, immediately."""

    reserve_seconds: float = 25.0
    """Flight time the return leg is assumed to need. Used to decide whether
    returning is even possible: below this the correct action is to land here
    rather than to start a journey that ends in a fall."""


@dataclass(frozen=True)
class Geofence:
    """The volume the aircraft is allowed to be in."""

    min_corner: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.35, 0.40]))
    max_corner: np.ndarray = field(default_factory=lambda: np.array([3.65, 2.65, 1.90]))

    def contains(self, position) -> bool:
        p = np.asarray(position, dtype=float)
        return bool(np.all(p >= self.min_corner) and np.all(p <= self.max_corner))

    def breach_distance(self, position, ignore_floor: bool = False) -> float:
        """How far outside the fence a point is, metres. Zero inside.

        ``ignore_floor`` drops the lower z bound. That bound is a minimum height
        above the floor and an aircraft taking off or landing is below it by
        definition: it starts and ends on the ground. Applying it during those
        phases means the geofence trips half a second into every take-off, which
        is exactly what it did the first time this was flown. The horizontal
        bounds and the ceiling still apply throughout, because there is no phase
        of flight in which drifting into a wall is expected.
        """
        p = np.asarray(position, dtype=float)
        low = self.min_corner.copy()
        if ignore_floor:
            low = low.copy()
            low[2] = -np.inf
        outside = np.maximum(np.maximum(low - p, p - self.max_corner), 0.0)
        return float(np.linalg.norm(outside))


@dataclass(frozen=True)
class AircraftConfig:
    """The whole aircraft configuration."""

    mount: CameraMount = field(default_factory=CameraMount)
    rc: RcScaling = field(default_factory=RcScaling)
    battery: BatteryPolicy = field(default_factory=BatteryPolicy)
    geofence: Geofence = field(default_factory=Geofence)

    mass: float = 0.080
    """kg, with battery. Published."""

    max_tilt: float = np.deg2rad(20.0)
    """Radians. The aircraft's own limit; the controller never commands beyond
    this because it cannot and the estimator uses it to bound how far the
    accelerometer's gravity component can be wrong."""

    telemetry_rate: float = 10.0
    """Hz, published for the state broadcast on UDP 8890."""

    video_rate: float = 30.0
    """Hz, published for the 960x720 stream on UDP 11111."""

    source: str = "defaults"

    @staticmethod
    def from_yaml(path) -> "AircraftConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return AircraftConfig.from_dict(data, source=str(path))

    @staticmethod
    def from_dict(data: dict, source: str = "dict") -> "AircraftConfig":
        mount_raw = data.get("camera_mount", {}) or {}
        rc_raw = data.get("rc_scaling", {}) or {}
        battery_raw = data.get("battery", {}) or {}
        fence_raw = data.get("geofence", {}) or {}
        return AircraftConfig(
            mount=CameraMount(
                x=float(mount_raw.get("x_m", CameraMount.x)),
                y=float(mount_raw.get("y_m", CameraMount.y)),
                z=float(mount_raw.get("z_m", CameraMount.z)),
                pitch=float(np.deg2rad(float(mount_raw.get("pitch_deg", 0.0)))),
                yaw=float(np.deg2rad(float(mount_raw.get("yaw_deg", 0.0)))),
                roll=float(np.deg2rad(float(mount_raw.get("roll_deg", 0.0)))),
            ),
            rc=RcScaling(
                horizontal_max=float(rc_raw.get("horizontal_max_mps", RcScaling.horizontal_max)),
                vertical_max=float(rc_raw.get("vertical_max_mps", RcScaling.vertical_max)),
                yaw_rate_max=float(
                    np.deg2rad(float(rc_raw.get("yaw_rate_max_dps", 100.0)))
                ),
                deadband=int(rc_raw.get("deadband_counts", RcScaling.deadband)),
            ),
            battery=BatteryPolicy(
                low=float(battery_raw.get("low_fraction", BatteryPolicy.low)),
                critical=float(battery_raw.get("critical_fraction", BatteryPolicy.critical)),
                reserve_seconds=float(
                    battery_raw.get("reserve_seconds", BatteryPolicy.reserve_seconds)
                ),
            ),
            geofence=Geofence(
                min_corner=np.asarray(
                    fence_raw.get("min_corner", [0.35, 0.35, 0.40]), dtype=float
                ),
                max_corner=np.asarray(
                    fence_raw.get("max_corner", [3.65, 2.65, 1.90]), dtype=float
                ),
            ),
            mass=float(data.get("mass_kg", 0.080)),
            max_tilt=float(np.deg2rad(float(data.get("max_tilt_deg", 20.0)))),
            telemetry_rate=float(data.get("telemetry_rate_hz", 10.0)),
            video_rate=float(data.get("video_rate_hz", 30.0)),
            source=source,
        )


def default_config(path: Optional[str] = None) -> AircraftConfig:
    """The shipped configuration, or the defaults if the file is missing.

    The file is looked up relative to the repository root so that scripts run
    from anywhere get the same answer.
    """
    if path is not None:
        return AircraftConfig.from_yaml(path)
    candidate = (
        Path(__file__).resolve().parents[2] / "tello_bringup" / "config" / "aircraft.yaml"
    )
    if candidate.is_file():
        return AircraftConfig.from_yaml(candidate)
    return AircraftConfig()
