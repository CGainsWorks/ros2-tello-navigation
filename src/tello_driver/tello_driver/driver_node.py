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
``tello_state`` (``tello_msgs/TelloState``), every decoded state packet, which is
what the fusion filter and the failsafe read; ``image`` (``sensor_msgs/Image``),
decoded from the H.264 stream; ``imu`` (``sensor_msgs/Imu``) and ``battery``
(``sensor_msgs/BatteryState``) for standard tools; ``link_status``
(``tello_msgs/LinkStatus``).

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
import time
from pathlib import Path

_CORE = Path(__file__).resolve().parents[2] / "tello_core"
if _CORE.is_dir() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import BatteryState, Image, Imu
from std_srvs.srv import Trigger

from tello_core.config import AircraftConfig
from tello_core.link.udp import UdpLink, UdpSettings
from tello_core.types import RcCommand
from tello_msgs.msg import LinkStatus
from tello_msgs.msg import RcCommand as RcMessage
from tello_msgs.msg import TelloState as TelloStateMessage

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
        # Where the aircraft is. The defaults are the Tello's own: its access
        # point puts it at 192.168.10.1 and it answers on 8889 to whichever port
        # the command came from. The emulator launch changes only the address
        # and lets the operating system pick the local command port, because the
        # emulator already holds 8889 on the same machine.
        self.declare_parameter("tello_address", "192.168.10.1")
        self.declare_parameter("command_port", 8889)
        self.declare_parameter("local_command_port", 8889)
        self.declare_parameter("state_port", 8890)
        self.declare_parameter("video_port", 11111)
        # bgr8 is what the decoder produces. mono8 is a third of the bytes per
        # frame on the wire between nodes and the marker detector converts to
        # grey first anyway.
        self.declare_parameter("image_encoding", "bgr8")
        # Seconds to keep trying to enter SDK mode. A Tello that has only just
        # been switched on, or a laptop still joining its access point, does not
        # answer the first "command"; neither does an emulator still building
        # its room.
        self.declare_parameter("connect_timeout", 15.0)

        config_path = self.get_parameter("aircraft_config").value
        self.config = AircraftConfig.from_yaml(config_path) if config_path else AircraftConfig()
        self.frame_id = str(self.get_parameter("frame_id").value)

        link_type = str(self.get_parameter("link_type").value).lower()
        self.link = self._build_link(link_type)
        self._connect(link_type, float(self.get_parameter("connect_timeout").value))
        if not self.link.start_video():
            self.get_logger().error(f"no video: {self.link.status().detail}")
        self.image_encoding = str(self.get_parameter("image_encoding").value)
        if self.image_encoding not in ("bgr8", "mono8"):
            raise RuntimeError(f"image_encoding must be bgr8 or mono8, not {self.image_encoding!r}")

        self.state_publisher = self.create_publisher(TelloStateMessage, "tello_state", 10)
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
                "the UDP driver has never been run against a real Tello, only against "
                "the emulator. See docs/HARDWARE.md before flying anything with it."
            )
            settings = UdpSettings(
                address=(
                    str(self.get_parameter("tello_address").value),
                    int(self.get_parameter("command_port").value),
                ),
                local_command_port=int(self.get_parameter("local_command_port").value),
                state_port=int(self.get_parameter("state_port").value),
                video_port=int(self.get_parameter("video_port").value),
            )
            return UdpLink(settings, video_decoder=self._make_decoder())
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

    def _connect(self, link_type: str, timeout: float) -> None:
        deadline = time.monotonic() + max(timeout, 0.0)
        while not self.link.connect():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"could not connect over the '{link_type}' link: "
                    f"{self.link.status().detail}"
                )
            self.get_logger().info(
                f"no answer yet: {self.link.status().detail}", throttle_duration_sec=2.0
            )
            time.sleep(0.5)

    def _make_decoder(self):
        try:
            from tello_core.link.video import H264Decoder

            return H264Decoder()
        except RuntimeError as error:
            self.get_logger().error(f"{error}. Running without video.")
            return None

    # ------------------------------------------------------------------- polling

    def _poll(self) -> None:
        state = self.link.poll_state()
        if state is not None:
            self.state_publisher.publish(self._state_message(state))
            self.imu_publisher.publish(self._imu_message(state))
            self.battery_publisher.publish(self._battery_message(state))

        frame = self.link.poll_frame()
        if frame is not None:
            self.image_publisher.publish(self._image_message(frame))

    def _state_message(self, state) -> TelloStateMessage:
        message = TelloStateMessage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.accel.x, message.accel.y, message.accel.z = (
            float(v) for v in np.asarray(state.accel).ravel()
        )
        message.gyro_z = float(state.gyro_z)
        message.yaw, message.pitch, message.roll = (
            float(v) for v in np.asarray(state.attitude).ravel()
        )
        message.velocity_body.x, message.velocity_body.y, message.velocity_body.z = (
            float(v) for v in np.asarray(state.velocity_body).ravel()
        )
        message.height = float(state.height)
        message.barometer = float(state.barometer)
        message.battery = float(state.battery)
        message.temperature = float(state.temperature)
        return message

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
        if self.image_encoding == "mono8" and image.ndim == 3:
            image = image[:, :, 0] if _is_grey(image) else _to_grey(image)
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


def _is_grey(image) -> bool:
    """Cheap check on a few rows, so a colour camera is never silently mangled."""
    rows = image[:: max(1, image.shape[0] // 8)]
    return bool(np.array_equal(rows[:, :, 0], rows[:, :, 1]))


def _to_grey(image):
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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
