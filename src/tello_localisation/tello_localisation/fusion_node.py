"""ROS 2 node: telemetry and marker fixes in, fused navigation state out.

As with the marker node, everything of substance is in
:mod:`tello_core.localisation.fusion` and this file only plumbs. It also
publishes the ``map`` to ``base_link`` transform, so that RViz shows the aircraft
where the estimator thinks it is rather than where a separate and inevitably
divergent copy of the maths thinks it is.

A note on time
--------------
The filter is driven off the timestamps in the messages, not off the node's own
clock. That is what lets the same code run against a bag replayed at half speed
and it is what makes the out-of-order handling work: a fix arrives after the
telemetry it should have been applied before and the filter rewinds. Reading the
wall clock here instead would quietly convert every one of those into a
measurement applied at the wrong instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parents[2] / "tello_core"
if _CORE.is_dir() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import TransformBroadcaster

from tello_core.localisation.fusion import Ekf, FusionSettings
from tello_core.types import MarkerFix as CoreFix
from tello_core.types import TelloState
from tello_msgs.msg import MarkerFix, NavState
from tello_msgs.msg import TelloState as TelloStateMessage

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class FusionNode(Node):
    """Runs the EKF and publishes the navigation state."""

    def __init__(self) -> None:
        super().__init__("tello_fusion")

        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("body_frame", "base_link")
        self.declare_parameter("fix_timeout", 2.0)
        self.declare_parameter("max_position_sigma", 0.45)

        settings = FusionSettings(
            fix_timeout=float(self.get_parameter("fix_timeout").value),
            max_position_sigma=float(self.get_parameter("max_position_sigma").value),
        )
        self.filter = Ekf(settings)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.body_frame = str(self.get_parameter("body_frame").value)

        self.create_subscription(TelloStateMessage, "tello_state", self._on_state, SENSOR_QOS)
        self.create_subscription(MarkerFix, "fix", self._on_fix, 10)
        self.publisher = self.create_publisher(NavState, "nav_state", 10)
        self.broadcaster = TransformBroadcaster(self)

        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / max(rate, 1e-3), self._publish)
        self.get_logger().info(f"fusion up, publishing nav_state at {rate:g} Hz")

    # ------------------------------------------------------------------ callbacks

    def _on_state(self, message: TelloStateMessage) -> None:
        """One decoded state packet, exactly as the offline runner feeds it.

        This used to arrive as a ``sensor_msgs/Imu`` and be rebuilt into a state
        with the height and the body velocity set to zero, because an Imu has
        nowhere to put them. The filter then applied a zero body velocity as a
        measurement on every packet and its height update never ran.
        ``tello_msgs/TelloState`` carries every field the decoder produces, so
        the ROS graph and the offline runner now give the filter the same inputs.
        """
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        state = TelloState(
            timestamp=stamp,
            accel=np.array([message.accel.x, message.accel.y, message.accel.z]),
            gyro_z=float(message.gyro_z),
            attitude=np.array([message.yaw, message.pitch, message.roll]),
            velocity_body=np.array(
                [message.velocity_body.x, message.velocity_body.y, message.velocity_body.z]
            ),
            height=float(message.height),
            barometer=float(message.barometer),
            battery=float(message.battery),
            temperature=float(message.temperature),
            valid=True,
        )
        self.filter.predict(state)

    def _on_fix(self, message: MarkerFix) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        fix = CoreFix(
            timestamp=stamp,
            position=np.array(
                [message.position.x, message.position.y, message.position.z]
            ),
            yaw=float(message.yaw),
            marker_ids=tuple(message.marker_ids),
            reprojection_rmse=float(message.reprojection_rmse),
            position_covariance=np.asarray(
                message.position_covariance, dtype=float
            ).reshape(3, 3),
            yaw_variance=float(message.yaw_variance),
            n_points=int(message.n_points),
        )
        if not self.filter.update_marker_fix(fix):
            self.get_logger().debug(
                f"fix from {list(message.marker_ids)} rejected", throttle_duration_sec=2.0
            )

    # ------------------------------------------------------------------- output

    def _publish(self) -> None:
        if self.filter.timestamp is None:
            return
        state = self.filter.nav_state()
        now = self.get_clock().now().to_msg()

        message = NavState()
        message.header.stamp = now
        message.header.frame_id = self.map_frame
        message.position.x = float(state.position[0])
        message.position.y = float(state.position[1])
        message.position.z = float(state.position[2])
        message.velocity.x = float(state.velocity[0])
        message.velocity.y = float(state.velocity[1])
        message.velocity.z = float(state.velocity[2])
        message.yaw = float(state.yaw)
        message.accel_bias.x = float(state.accel_bias[0])
        message.accel_bias.y = float(state.accel_bias[1])
        message.accel_bias.z = float(state.accel_bias[2])
        message.gyro_bias = float(state.gyro_bias)
        message.position_covariance = [
            float(v) for v in np.asarray(state.covariance)[:3, :3].ravel()
        ]
        message.position_sigma = state.position_sigma()
        message.localised = bool(state.localised)
        message.time_since_fix = float(min(state.time_since_fix, 1e6))
        self.publisher.publish(message)

        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.body_frame
        transform.transform.translation.x = float(state.position[0])
        transform.transform.translation.y = float(state.position[1])
        transform.transform.translation.z = float(state.position[2])
        half = 0.5 * float(state.yaw)
        transform.transform.rotation.z = float(np.sin(half))
        transform.transform.rotation.w = float(np.cos(half))
        self.broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusionNode()
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
