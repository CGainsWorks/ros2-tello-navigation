"""ROS 2 node: camera frames in, marker fixes out.

A wrapper and nothing else. Every line of detection, pose solving and covariance
estimation lives in :mod:`tello_core.localisation`, which has no ROS dependency
and is what the test suite exercises. This file subscribes, converts, calls and
publishes.

That split is deliberate and it is the reason this repository's results can be
reproduced on a laptop with nothing but Python and OpenCV installed. It also means
the interesting code is testable at all: a node is awkward to unit-test and a
function that takes an image and returns a pose is not.

Topics
------
``~/image`` (``sensor_msgs/Image``)
    The forward camera. BGR8 or MONO8.
``~/fix`` (``tello_msgs/MarkerFix``)
    One message per frame that produced a usable fix. Frames that produced none
    publish nothing, rather than publishing an invalid fix: a subscriber that has
    to check a validity flag will eventually forget to.
``~/nav_state`` (``tello_msgs/NavState``), subscribed
    Used only for the prior that resolves the planar pose ambiguity. Optional; the
    node works without it and rejects more single-marker fixes as a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The core package is a sibling directory rather than an installed distribution,
# so that a colcon build and a bare `python -m pytest` see the same source. The
# installed layout puts it on the path already, in which case this is a no-op.
_CORE = Path(__file__).resolve().parents[2] / "tello_core"
if _CORE.is_dir() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.localisation.markers import MarkerMap
from tello_core.localisation.multi import estimate_pose
from tello_core.localisation.qr import QrDetector, filter_known
from tello_core.types import Frame
from tello_msgs.msg import MarkerFix, NavState

# Sensor data, not commands: best effort with a depth of one. A detector that
# falls behind should drop frames rather than queue them, because a fix computed
# from a frame three deep in a queue is a fix about where the aircraft was.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class MarkerNode(Node):
    """Detects QR markers and publishes absolute fixes."""

    def __init__(self) -> None:
        super().__init__("tello_marker_localisation")

        self.declare_parameter("marker_map", "")
        self.declare_parameter("aircraft_config", "")
        self.declare_parameter("camera_calibration", "")
        self.declare_parameter("detect_every", 1)
        self.declare_parameter("publish_unknown_payloads", True)

        map_path = self.get_parameter("marker_map").value
        if not map_path:
            raise RuntimeError(
                "the 'marker_map' parameter is required. Without a map a detection "
                "cannot be turned into a position and defaulting to an empty map "
                "would mean flying with an estimator that silently never fixes."
            )
        self.marker_map = MarkerMap.from_yaml(map_path)

        config_path = self.get_parameter("aircraft_config").value
        self.config = (
            AircraftConfig.from_yaml(config_path) if config_path else AircraftConfig()
        )
        calibration = self.get_parameter("camera_calibration").value
        self.camera = (
            CameraModel.from_yaml(calibration) if calibration else CameraModel.tello_nominal()
        )
        self.detect_every = max(1, int(self.get_parameter("detect_every").value))

        self.detector = QrDetector()
        self._counter = 0
        self._prior = None
        self._prior_sigma = 0.5
        self._unknown: dict = {}

        self.create_subscription(Image, "image", self._on_image, SENSOR_QOS)
        self.create_subscription(NavState, "nav_state", self._on_nav_state, 10)
        self.fix_publisher = self.create_publisher(MarkerFix, "fix", 10)

        self.get_logger().info(
            f"marker localisation up: {len(self.marker_map)} markers from {map_path}, "
            f"detector backend {self.detector.backend}, camera {self.camera.source}"
        )
        if self.marker_map.survey.method == "designed":
            self.get_logger().warning(
                "the marker map's survey method is 'designed', not measured. Every "
                "position this node publishes is relative to a layout that has not "
                "been surveyed."
            )

    # ------------------------------------------------------------------ callbacks

    def _on_nav_state(self, message: NavState) -> None:
        self._prior = np.array(
            [message.position.x, message.position.y, message.position.z]
        )
        self._prior_sigma = max(float(message.position_sigma), 0.05)

    def _on_image(self, message: Image) -> None:
        self._counter += 1
        if self._counter % self.detect_every:
            return

        image = _image_to_array(message)
        if image is None:
            self.get_logger().warning(
                f"unsupported image encoding {message.encoding!r}", throttle_duration_sec=10.0
            )
            return

        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        observations = self.detector.detect(Frame(timestamp=stamp, image=image))
        known, unknown = filter_known(observations, self.marker_map)
        for payload in unknown:
            self._unknown[payload] = self._unknown.get(payload, 0) + 1
            if self._unknown[payload] == 1:
                self.get_logger().info(
                    f"saw a QR code that is not in the map: {payload!r}. Harmless if it "
                    "is a laptop sticker; a typo in the map if it is not."
                )

        fix = estimate_pose(
            known,
            self.marker_map,
            self.camera,
            self.config.mount.transform(),
            timestamp=stamp,
            prior_position=self._prior,
            prior_sigma=self._prior_sigma,
        )
        if fix is None:
            return

        self.fix_publisher.publish(_fix_to_message(fix, message.header.frame_id))


def _image_to_array(message: Image):
    """Convert without cv_bridge.

    cv_bridge is a heavy dependency that has historically been awkward to install
    outside a full ROS desktop and this conversion is six lines. The encodings
    handled are the two the Tello driver ever produces.
    """
    height, width = message.height, message.width
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    if message.encoding in ("bgr8", "rgb8"):
        image = buffer.reshape(height, width, 3)
        return image[:, :, ::-1] if message.encoding == "rgb8" else image
    if message.encoding == "mono8":
        return buffer.reshape(height, width)
    return None


def _fix_to_message(fix, frame_id: str) -> MarkerFix:
    message = MarkerFix()
    message.header.frame_id = frame_id or "map"
    message.header.stamp.sec = int(fix.timestamp)
    message.header.stamp.nanosec = int((fix.timestamp % 1.0) * 1e9)
    message.position.x = float(fix.position[0])
    message.position.y = float(fix.position[1])
    message.position.z = float(fix.position[2])
    message.yaw = float(fix.yaw)
    message.marker_ids = list(fix.marker_ids)
    message.n_points = int(fix.n_points)
    message.reprojection_rmse = float(fix.reprojection_rmse)
    message.position_covariance = [
        float(v) for v in np.asarray(fix.position_covariance, dtype=float).ravel()
    ]
    message.yaw_variance = float(fix.yaw_variance)
    return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MarkerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # launch sends SIGINT and then, if a node is slow to go, SIGTERM; a
        # second interrupt arriving inside destroy_node must not turn a clean
        # shutdown into a traceback and a non-zero exit code.
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
