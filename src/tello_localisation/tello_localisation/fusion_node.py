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
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from tello_core.localisation.fusion import Ekf, FusionSettings
from tello_core.types import MarkerFix as CoreFix
from tello_core.types import TelloState
from tello_msgs.msg import MarkerFix, NavState

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

        self.create_subscription(Imu, "imu", self._on_imu, SENSOR_QOS)
        self.create_subscription(MarkerFix, "fix", self._on_fix, 10)
        self.publisher = self.create_publisher(NavState, "nav_state", 10)
        self.broadcaster = TransformBroadcaster(self)

        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / max(rate, 1e-3), self._publish)
        self.get_logger().info(f"fusion up, publishing nav_state at {rate:g} Hz")

    # ------------------------------------------------------------------ callbacks

    def _on_imu(self, message: Imu) -> None:
        """The driver republishes the Tello state packet as an Imu.

        Only the fields the aircraft actually measures are populated: linear
        acceleration, the yaw rate and the orientation from its own attitude
        filter. The rest are left at zero with a negative covariance, which is
        the ROS convention for "not measured" and this node ignores them.
        """
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        yaw, pitch, roll = _quaternion_to_euler(
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
        )
        state = TelloState(
            timestamp=stamp,
            accel=np.array(
                [
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                ]
            ),
            gyro_z=float(message.angular_velocity.z),
            attitude=np.array([yaw, pitch, roll]),
            velocity_body=np.zeros(3),
            height=0.0,
            barometer=float("nan"),
            battery=1.0,
            temperature=float("nan"),
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


def _quaternion_to_euler(x: float, y: float, z: float, w: float):
    """Quaternion to intrinsic Z-Y-X angles, without tf_transformations.

    Twenty lines rather than a dependency that is packaged differently on every
    ROS distribution and is the single most common reason a node fails to start
    on somebody else's machine.
    """
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sin_roll, cos_roll))

    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = float(np.clip(sin_pitch, -1.0, 1.0))
    pitch = float(np.arcsin(sin_pitch))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(sin_yaw, cos_yaw))
    return yaw, pitch, roll


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
