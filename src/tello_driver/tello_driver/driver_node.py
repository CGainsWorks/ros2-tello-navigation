"""ROS 2 node: owns the link and nothing else.

Publishes what the aircraft says and sends what it is told. The only node in the
stack that talks to hardware, which is what makes it the only node that has to be
swapped to fly the simulator instead: point ``link_type`` at ``sim`` and the rest
of the graph cannot tell the difference.

That substitution is not a testing convenience bolted on afterwards. It is the
same :class:`~tello_core.link.base.TelloLink` interface the offline mission runner
uses, so the simulator, the ROS graph and the test suite all drive one
implementation of the flight logic.

Publishes
---------
``image`` (``sensor_msgs/Image``), ``imu`` (``sensor_msgs/Imu``),
``link_status`` (``tello_msgs/LinkStatus``), ``battery`` (``sensor_msgs/BatteryState``)

Subscribes
----------
``rc`` (``tello_msgs/RcCommand``)

Services
--------
``takeoff``, ``land``, ``emergency`` (``std_srvs/Trigger``)

A note on the emergency service
-------------------------------
``emergency`` cuts the motors. It is exposed because the failsafe machine needs
it and it is a ``Trigger`` like the others because dressing it up in a
confirmation protocol would add latency to the one command whose entire value is
that it is immediate. It is not something to call by hand from the command line
while the aircraft is above head height.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parents[2] / "tello_core"
if _CORE.is_dir() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import BatteryState, Image, Imu
from std_srvs.srv import Trigger

from tello_core.config import AircraftConfig
from tello_core.link.udp import UdpLink
from tello_core.types import RcCommand
from tello_msgs.msg import LinkStatus
from tello_msgs.msg import RcCommand as RcMessage

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class DriverNode(Node):
    """Bridges a :class:`~tello_core.link.base.TelloLink` onto ROS topics."""

    def __init__(self) -> None:
        super().__init__("tello_driver")

        self.declare_parameter("link_type", "udp")
        self.declare_parameter("aircraft_config", "")
        self.declare_parameter("marker_map", "")
        self.declare_parameter("poll_rate", 60.0)
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("sim_seed", 1)
        self.declare_parameter("sim_start", [1.0, 0.9, 0.0])

        config_path = self.get_parameter("aircraft_config").value
        self.config = AircraftConfig.from_yaml(config_path) if config_path else AircraftConfig()
        self.frame_id = str(self.get_parameter("frame_id").value)

        link_type = str(self.get_parameter("link_type").value).lower()
        self.link = self._build_link(link_type)
        if not self.link.connect():
            raise RuntimeError(
                f"could not connect over the '{link_type}' link: {self.link.status().detail}"
            )
        self.link.start_video()

        self.image_publisher = self.create_publisher(Image, "image", SENSOR_QOS)
        self.imu_publisher = self.create_publisher(Imu, "imu", SENSOR_QOS)
        self.status_publisher = self.create_publisher(LinkStatus, "link_status", 10)
        self.battery_publisher = self.create_publisher(BatteryState, "battery", 10)
        self.create_subscription(RcMessage, "rc", self._on_rc, 10)

        self.create_service(Trigger, "takeoff", self._on_takeoff)
        self.create_service(Trigger, "land", self._on_land)
        self.create_service(Trigger, "emergency", self._on_emergency)

        rate = float(self.get_parameter("poll_rate").value)
        self.create_timer(1.0 / max(rate, 1e-3), self._poll)
        self.create_timer(0.2, self._publish_status)
        self.get_logger().info(f"driver up on the '{link_type}' link")

    def _build_link(self, link_type: str):
        if link_type == "udp":
            self.get_logger().warning(
                "the UDP driver has never been run against a real Tello. See "
                "docs/HARDWARE.md before flying anything with it."
            )
            return UdpLink()
        if link_type == "sim":
            from tello_core.camera import CameraModel
            from tello_core.localisation.markers import MarkerMap
            from tello_core.sim.link import SimSettings
            from tello_core.sim.world import Room

            map_path = self.get_parameter("marker_map").value
            if not map_path:
                raise RuntimeError("the simulated link needs a 'marker_map' parameter")
            marker_map = MarkerMap.from_yaml(map_path)
            start = [float(v) for v in self.get_parameter("sim_start").value]
            return _WallClockSimLink(
                Room.from_marker_map(marker_map),
                self.config,
                CameraModel.tello_nominal(),
                SimSettings(seed=int(self.get_parameter("sim_seed").value)),
                start_position=np.array(start),
                clock=lambda: self.get_clock().now().nanoseconds * 1e-9,
            )
        raise RuntimeError(f"unknown link_type {link_type!r}, expected 'udp' or 'sim'")

    # ------------------------------------------------------------------- polling

    def _poll(self) -> None:
        state = self.link.poll_state()
        if state is not None:
            self.imu_publisher.publish(self._imu_message(state))
            self.battery_publisher.publish(self._battery_message(state))

        frame = self.link.poll_frame()
        if frame is not None:
            self.image_publisher.publish(self._image_message(frame))

    def _imu_message(self, state) -> Imu:
        message = Imu()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.linear_acceleration.x = float(state.accel[0])
        message.linear_acceleration.y = float(state.accel[1])
        message.linear_acceleration.z = float(state.accel[2])
        message.angular_velocity.z = float(state.gyro_z)

        yaw, pitch, roll = [float(v) for v in np.asarray(state.attitude).ravel()]
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        message.orientation.w = float(cr * cp * cy + sr * sp * sy)
        message.orientation.x = float(sr * cp * cy - cr * sp * sy)
        message.orientation.y = float(cr * sp * cy + sr * cp * sy)
        message.orientation.z = float(cr * cp * sy - sr * sp * cy)

        # ROS convention for "this field is not measured". The Tello publishes no
        # covariance at all and inventing one would be an assertion about an
        # aircraft nobody here has.
        message.angular_velocity_covariance[0] = -1.0
        return message

    def _battery_message(self, state) -> BatteryState:
        message = BatteryState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.percentage = float(state.battery)
        message.present = True
        message.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        return message

    def _image_message(self, frame) -> Image:
        image = frame.image
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.height, message.width = int(image.shape[0]), int(image.shape[1])
        if image.ndim == 2:
            message.encoding = "mono8"
            message.step = message.width
        else:
            message.encoding = "bgr8"
            message.step = message.width * 3
        message.is_bigendian = 0
        message.data = np.ascontiguousarray(image).tobytes()
        return message

    def _publish_status(self) -> None:
        status = self.link.status()
        message = LinkStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.connected = bool(status.connected)
        message.last_state_age = float(min(status.last_state_age, 1e6))
        message.last_frame_age = float(min(status.last_frame_age, 1e6))
        message.packets_received = int(status.packets_received)
        message.packets_dropped = int(status.packets_dropped)
        message.commands_sent = int(status.commands_sent)
        message.detail = str(status.detail)
        self.status_publisher.publish(message)

    # ------------------------------------------------------------------ commands

    def _on_rc(self, message: RcMessage) -> None:
        self.link.send_rc(
            RcCommand(
                left_right=int(message.left_right),
                forward_back=int(message.forward_back),
                up_down=int(message.up_down),
                yaw_rate=int(message.yaw_rate),
            )
        )

    def _on_takeoff(self, request, response):
        del request
        self.link.takeoff()
        response.success = True
        response.message = "takeoff sent"
        return response

    def _on_land(self, request, response):
        del request
        self.link.land()
        response.success = True
        response.message = "land sent"
        return response

    def _on_emergency(self, request, response):
        del request
        self.link.emergency()
        self.get_logger().error("emergency: motors cut")
        response.success = True
        response.message = "motors cut"
        return response

    def destroy_node(self) -> bool:
        """Close the link before the node goes away.

        Without this the receive threads outlive the node and keep a UDP socket
        bound, so the next launch fails to bind and reports something unrelated
        about an address already in use.
        """
        try:
            self.link.close()
        except Exception as error:  # noqa: BLE001 - shutdown must not raise
            self.get_logger().warning(f"link did not close cleanly: {error}")
        return super().destroy_node()


class _WallClockSimLink:
    """The simulator driven by a ROS clock instead of by its own.

    A thin adapter rather than a second simulator. The offline runner advances
    simulated time explicitly, which is what makes an offline flight run faster
    than real time and reproducibly. Inside a ROS graph nothing is going to call
    ``advance``, so this wrapper steps the simulation by however much wall clock
    has actually passed each time it is polled.

    The consequence is that a simulated flight inside ROS runs in real time and
    is not bit-for-bit reproducible, because the step sizes depend on scheduling.
    That is the correct trade for a graph whose point is to exercise the ROS
    plumbing; the reproducible numbers come from the offline runner.
    """

    def __init__(self, room, config, camera, settings, start_position, clock) -> None:
        from tello_core.sim.link import SimLink

        self._link = SimLink(
            room, config, camera, settings, start_position=start_position
        )
        self._clock = clock
        self._last = None

    def _catch_up(self) -> None:
        now = self._clock()
        if self._last is None:
            self._last = now
            return
        delta = now - self._last
        self._last = now
        if delta > 0:
            self._link.advance(min(delta, 0.5))

    def connect(self) -> bool:
        return self._link.connect()

    def close(self) -> None:
        self._link.close()

    def start_video(self) -> bool:
        return self._link.start_video()

    def poll_state(self):
        self._catch_up()
        return self._link.poll_state()

    def poll_frame(self):
        return self._link.poll_frame()

    def send_rc(self, command) -> None:
        self._link.send_rc(command)

    def takeoff(self) -> None:
        self._link.takeoff()

    def land(self) -> None:
        self._link.land()

    def emergency(self) -> None:
        self._link.emergency()

    def status(self):
        return self._link.status()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
