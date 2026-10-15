"""ROS 2 node: the waypoint controller and the failsafe machine, together.

Together deliberately. The obvious ROS decomposition puts the failsafe in its own
node publishing a mode and the controller in another subscribing to it and it
is the wrong decomposition for this particular job. The failsafe's whole purpose
is to be able to veto a command and a veto that travels over a topic arrives at
best one queue later. On a link-loss trip that delay is spent transmitting stick
commands the failsafe has already decided must stop, which is precisely the
behaviour ``docs/FAILSAFE.md`` explains must not happen.

So the machine and the controller share a process and a function call and the
node publishes the failsafe's decisions for observation rather than for control.
Anything that wants to watch, log, or display them can; nothing has to be
listening for the aircraft to be safe.

Subscribes
----------
``nav_state`` (``tello_msgs/NavState``), ``link_status`` (``tello_msgs/LinkStatus``)

Publishes
---------
``rc`` (``tello_msgs/RcCommand``), ``failsafe_event``, ``waypoint_event``

Services
--------
``arm`` (``std_srvs/Trigger``) requests take-off; ``abort`` requests an immediate
controlled landing.
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
from std_srvs.srv import Trigger

from tello_core.camera import CameraModel
from tello_core.config import AircraftConfig
from tello_core.control.failsafe import (
    DescentProfile,
    FailsafeInputs,
    FailsafeMachine,
    FailsafeSettings,
)
from tello_core.control.mission import Mission
from tello_core.control.waypoint import WaypointController, heading_for_markers
from tello_core.link.base import LinkStatus as CoreLinkStatus
from tello_core.localisation.markers import MarkerMap
from tello_core.types import FlightState, RcCommand
from tello_core.types import NavState as CoreNavState
from tello_msgs.msg import FailsafeEvent, LinkStatus, NavState, WaypointEvent
from tello_msgs.msg import RcCommand as RcMessage


class MissionNode(Node):
    """Flies a mission over ROS topics."""

    def __init__(self) -> None:
        super().__init__("tello_mission")

        self.declare_parameter("mission", "")
        self.declare_parameter("marker_map", "")
        self.declare_parameter("aircraft_config", "")
        self.declare_parameter("camera_calibration", "")
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("auto_arm", False)

        mission_path = self.get_parameter("mission").value
        map_path = self.get_parameter("marker_map").value
        if not mission_path or not map_path:
            raise RuntimeError("the 'mission' and 'marker_map' parameters are both required")

        self.mission = Mission.from_yaml(mission_path)
        self.marker_map = MarkerMap.from_yaml(map_path)
        config_path = self.get_parameter("aircraft_config").value
        self.config = AircraftConfig.from_yaml(config_path) if config_path else AircraftConfig()
        calibration = self.get_parameter("camera_calibration").value
        self.camera = (
            CameraModel.from_yaml(calibration) if calibration else CameraModel.tello_nominal()
        )

        problems = self.mission.validate(self.config, self.marker_map)
        if problems:
            raise RuntimeError("mission is not flyable:\n  " + "\n  ".join(problems))

        self.controller = WaypointController(self.config)
        self.failsafe = FailsafeMachine(self.config, FailsafeSettings())
        self.descent = DescentProfile(FailsafeSettings())

        self._nav: CoreNavState = None
        self._link = CoreLinkStatus(connected=False, last_state_age=float("inf"),
                                    last_frame_age=float("inf"))
        self._battery = 1.0
        self._temperature = float("nan")
        self._height = 0.0
        self._index = 0
        self._loop = 0
        self._waypoint_started = None
        self._start_time = None
        self._arm_requested = bool(self.get_parameter("auto_arm").value)
        self._abort = False
        self._published_events = 0
        self._resolved_yaw: dict = {}

        self.create_subscription(NavState, "nav_state", self._on_nav_state, 10)
        self.create_subscription(LinkStatus, "link_status", self._on_link_status, 10)
        self.rc_publisher = self.create_publisher(RcMessage, "rc", 10)
        self.failsafe_publisher = self.create_publisher(FailsafeEvent, "failsafe_event", 10)
        self.waypoint_publisher = self.create_publisher(WaypointEvent, "waypoint_event", 10)

        self.create_service(Trigger, "arm", self._on_arm)
        self.create_service(Trigger, "abort", self._on_abort)

        rate = float(self.get_parameter("control_rate").value)
        self._period = 1.0 / max(rate, 1e-3)
        self._last_tick = None
        self.create_timer(self._period, self._tick)
        self.get_logger().info(
            f"mission '{self.mission.name}' loaded with {len(self.mission.waypoints)} "
            f"waypoints, control at {rate:g} Hz. Waiting for the arm service."
        )

    # ------------------------------------------------------------------ services

    def _on_arm(self, request, response):
        del request
        self._arm_requested = True
        response.success = True
        response.message = "take-off requested"
        return response

    def _on_abort(self, request, response):
        del request
        self._abort = True
        response.success = True
        response.message = "abort requested, landing"
        return response

    # ----------------------------------------------------------------- callbacks

    def _on_nav_state(self, message: NavState) -> None:
        self._nav = CoreNavState(
            timestamp=message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            position=np.array([message.position.x, message.position.y, message.position.z]),
            velocity=np.array([message.velocity.x, message.velocity.y, message.velocity.z]),
            yaw=float(message.yaw),
            accel_bias=np.array(
                [message.accel_bias.x, message.accel_bias.y, message.accel_bias.z]
            ),
            gyro_bias=float(message.gyro_bias),
            covariance=_expand_covariance(message.position_covariance),
            localised=bool(message.localised),
            time_since_fix=float(message.time_since_fix),
        )

    def _on_link_status(self, message: LinkStatus) -> None:
        self._link = CoreLinkStatus(
            connected=bool(message.connected),
            last_state_age=float(message.last_state_age),
            last_frame_age=float(message.last_frame_age),
            packets_received=int(message.packets_received),
            packets_dropped=int(message.packets_dropped),
            commands_sent=int(message.commands_sent),
            detail=str(message.detail),
        )

    # ---------------------------------------------------------------------- loop

    def _tick(self) -> None:
        if self._nav is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._start_time is None:
            self._start_time = now
        dt = self._period if self._last_tick is None else max(now - self._last_tick, 1e-3)
        self._last_tick = now

        inputs = FailsafeInputs(
            now=now,
            link=self._link,
            estimate=self._nav,
            battery=self._battery,
            temperature=self._temperature,
            height=self._height,
            flight_time=now - self._start_time,
            estimator_diverged=self._nav.position_sigma() > 2.0,
            operator_abort=self._abort,
            mission_complete=self._index >= len(self.mission.waypoints),
            takeoff_requested=self._arm_requested,
            airborne=self._height > 0.12,
        )
        self._arm_requested = False
        decision = self.failsafe.update(inputs)
        self._publish_new_events()

        command = self._command_for(decision.state, now, dt)
        if decision.send_rc:
            self.rc_publisher.publish(_rc_to_message(command, self.get_clock().now().to_msg()))

    def _command_for(self, state, now: float, dt: float) -> RcCommand:
        if state in (FlightState.IDLE, FlightState.LANDED):
            self.controller.reset()
            return RcCommand.zero()
        if state in (FlightState.LAND, FlightState.EMERGENCY_LAND):
            self.controller.reset()
            return self.controller.to_rc(np.array([0.0, 0.0, self.descent.rate(state)]), 0.0)
        if state is FlightState.TAKEOFF:
            target = np.array(
                [self._nav.position[0], self._nav.position[1], self.mission.takeoff_height]
            )
            return self.controller.update(target, self._nav.yaw, self._nav, dt)
        if state is FlightState.HOLD:
            return self.controller.update(
                self._nav.position.copy(), self._nav.yaw, self._nav, dt
            )
        if state is FlightState.RETURN:
            target = (
                np.asarray(self.mission.home, dtype=float)
                if self.mission.home is not None
                else self._nav.position
            )
            return self.controller.update(target, self._nav.yaw, self._nav, dt)

        if self._index >= len(self.mission.waypoints):
            return RcCommand.zero()
        waypoint = self.mission.waypoints[self._index]
        if self._waypoint_started is None:
            self._waypoint_started = now
            self.controller.start_waypoint()
        yaw = waypoint.yaw
        if yaw is None:
            yaw = self._resolved_yaw.get(self._index)
            if yaw is None:
                yaw = heading_for_markers(
                    waypoint.position,
                    self.marker_map,
                    self.config,
                    self.camera,
                    prefer=self._nav.yaw,
                )
                self._resolved_yaw[self._index] = yaw
        command = self.controller.update(waypoint.position, yaw, self._nav, dt)

        if self.controller.check_arrival(waypoint, self._nav, now):
            self._finish_waypoint(waypoint, now, "arrived")
        elif now - self._waypoint_started > waypoint.timeout:
            self._finish_waypoint(waypoint, now, "timeout")
        return command

    def _finish_waypoint(self, waypoint, now: float, kind: str) -> None:
        message = WaypointEvent()
        message.header.stamp = self.get_clock().now().to_msg()
        message.index = int(self._index)
        message.name = waypoint.name
        message.kind = kind
        message.position_error = float(
            np.linalg.norm(self._nav.position - waypoint.position)
        )
        self.waypoint_publisher.publish(message)
        self.get_logger().info(
            f"waypoint {waypoint.name} {kind} with {message.position_error:.2f} m of error"
        )
        self._index += 1
        self._waypoint_started = None

    def _publish_new_events(self) -> None:
        while self._published_events < len(self.failsafe.events):
            event = self.failsafe.events[self._published_events]
            self._published_events += 1
            message = FailsafeEvent()
            message.header.stamp = self.get_clock().now().to_msg()
            message.from_state = event.from_state.value
            message.to_state = event.to_state.value
            message.cause = event.cause.value
            message.detail = event.detail
            self.failsafe_publisher.publish(message)
            self.get_logger().warning(
                f"failsafe {event.from_state.value} -> {event.to_state.value} "
                f"({event.cause.value}): {event.detail}"
            )


def _expand_covariance(flat) -> np.ndarray:
    """Put a 3x3 position covariance into the filter's full-state layout.

    The controller and the failsafe only ever read the position block, so the
    other diagonals are filled with a value large enough to be obviously
    uninformative rather than with zeros, which would read as certainty.
    """
    covariance = np.eye(11) * 1e3
    covariance[:3, :3] = np.asarray(flat, dtype=float).reshape(3, 3)
    return covariance


def _rc_to_message(command: RcCommand, stamp) -> RcMessage:
    message = RcMessage()
    message.header.stamp = stamp
    message.left_right = int(command.left_right)
    message.forward_back = int(command.forward_back)
    message.up_down = int(command.up_down)
    message.yaw_rate = int(command.yaw_rate)
    return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
