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

The cycle mirrors MissionRunner
-------------------------------
The offline :class:`tello_core.control.mission.MissionRunner` owns its link and
its filter; this node gets both over topics. Everything between is the same
sequence: wait on the ground for a fix and live telemetry, take off, latch the
home point on the first localised estimate, feed the failsafe the battery,
height and temperature from the state packet, command ``takeoff``, ``land`` and
``emergency`` when it says so and fall silent on ``rc`` when it says so. The
first version of this node did not command take-off or landing at all and fed
the failsafe a battery that never ran down. Nothing noticed because nothing
launched it; the launch tests against the emulator in ``tello_bringup/test`` are
what now do.

Subscribes
----------
``nav_state`` (``tello_msgs/NavState``), ``link_status`` (``tello_msgs/LinkStatus``),
``tello_state`` (``tello_msgs/TelloState``)

Publishes
---------
``rc`` (``tello_msgs/RcCommand``), ``failsafe_event``, ``waypoint_event``

Calls
-----
``takeoff``, ``land``, ``emergency`` (``std_srvs/Trigger``) on the driver.

Services
--------
``~/arm`` (``std_srvs/Trigger``) requests take-off; ``~/abort`` requests an
immediate controlled landing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parents[2] / "tello_core"
if _CORE.is_dir() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
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
from tello_core.control.pid import ControlGains
from tello_core.control.waypoint import (
    WaypointController,
    heading_for_markers,
    heading_is_adequate,
)
from tello_core.geometry import wrap_angle
from tello_core.link.base import LinkStatus as CoreLinkStatus
from tello_core.localisation.markers import MarkerMap
from tello_core.types import FailsafeCause, FlightState, RcCommand
from tello_core.types import NavState as CoreNavState
from tello_msgs.msg import FailsafeEvent, LinkStatus, NavState, TelloState, WaypointEvent
from tello_msgs.msg import RcCommand as RcMessage

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class MissionNode(Node):
    """Flies a mission over ROS topics."""

    def __init__(self) -> None:
        super().__init__("tello_mission")

        self.declare_parameter("mission", "")
        self.declare_parameter("marker_map", "")
        self.declare_parameter("aircraft_config", "")
        self.declare_parameter("control_config", "")
        self.declare_parameter("camera_calibration", "")
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("auto_arm", False)
        self.declare_parameter("preflight_timeout", 12.0)
        self.declare_parameter("takeoff_settle", 2.0)
        self.declare_parameter("command_resend", 1.0)
        self.declare_parameter("yaw_replan_interval", 0.4)

        mission_path = self.get_parameter("mission").value
        map_path = self.get_parameter("marker_map").value
        if not mission_path or not map_path:
            raise RuntimeError("the 'mission' and 'marker_map' parameters are both required")

        self.mission = Mission.from_yaml(mission_path)
        self.marker_map = MarkerMap.from_yaml(map_path)
        config_path = self.get_parameter("aircraft_config").value
        self.config = AircraftConfig.from_yaml(config_path) if config_path else AircraftConfig()
        control_path = self.get_parameter("control_config").value
        gains = ControlGains.from_yaml(control_path) if control_path else ControlGains()
        calibration = self.get_parameter("camera_calibration").value
        self.camera = (
            CameraModel.from_yaml(calibration) if calibration else CameraModel.tello_nominal()
        )

        problems = self.mission.validate(self.config, self.marker_map)
        if problems:
            raise RuntimeError("mission is not flyable:\n  " + "\n  ".join(problems))

        self.settings = FailsafeSettings()
        self.controller = WaypointController(self.config, gains)
        self.failsafe = FailsafeMachine(self.config, self.settings)
        self.descent = DescentProfile(self.settings)
        self.preflight_timeout = float(self.get_parameter("preflight_timeout").value)
        self.takeoff_settle = float(self.get_parameter("takeoff_settle").value)
        self.command_resend = float(self.get_parameter("command_resend").value)
        self.yaw_replan_interval = float(self.get_parameter("yaw_replan_interval").value)

        self._nav: CoreNavState = None
        self._link = CoreLinkStatus(
            connected=False, last_state_age=float("inf"), last_frame_age=float("inf")
        )
        self._state = None
        self._index = 0
        self._loop = 0
        self._waypoint_started = None
        self._start_time = None
        self._arm_requested = bool(self.get_parameter("auto_arm").value)
        self._armed_at = None
        self._abort = False
        self._published_events = 0
        self._resolved_yaw: dict = {}
        self._yaw_replanned_at = float("-inf")
        self._home = None
        self._flight_state = None
        self._takeoff_at = None
        self._hold_position = None
        self._hold_yaw = None
        self._last_sent = {
            "takeoff": float("-inf"),
            "land": float("-inf"),
            "emergency": float("-inf"),
        }

        self.create_subscription(NavState, "nav_state", self._on_nav_state, 10)
        self.create_subscription(LinkStatus, "link_status", self._on_link_status, 10)
        self.create_subscription(TelloState, "tello_state", self._on_tello_state, SENSOR_QOS)
        self.rc_publisher = self.create_publisher(RcMessage, "rc", 10)
        self.failsafe_publisher = self.create_publisher(FailsafeEvent, "failsafe_event", 10)
        self.waypoint_publisher = self.create_publisher(WaypointEvent, "waypoint_event", 10)
        self._driver_clients = {
            name: self.create_client(Trigger, name) for name in ("takeoff", "land", "emergency")
        }

        self.create_service(Trigger, "~/arm", self._on_arm)
        self.create_service(Trigger, "~/abort", self._on_abort)

        rate = float(self.get_parameter("control_rate").value)
        self._period = 1.0 / max(rate, 1e-3)
        self._last_tick = None
        self.create_timer(self._period, self._tick)
        self.get_logger().info(
            f"mission '{self.mission.name}' loaded with {len(self.mission.waypoints)} "
            f"waypoints, control at {rate:g} Hz. "
            + ("Arming automatically." if self._arm_requested else "Waiting for ~/arm.")
        )

    # ------------------------------------------------------------------ services

    def _on_arm(self, request, response):
        del request
        if self.failsafe.state is not FlightState.IDLE:
            response.success = False
            response.message = f"not idle: {self.failsafe.state.value}"
            return response
        self._arm_requested = True
        self._armed_at = None
        response.success = True
        response.message = "take-off requested once localised"
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

    def _on_tello_state(self, message: TelloState) -> None:
        self._state = message

    # ---------------------------------------------------------------------- loop

    def _tick(self) -> None:
        if self._nav is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._start_time is None:
            self._start_time = now
        dt = self._period if self._last_tick is None else max(now - self._last_tick, 1e-3)
        self._last_tick = now
        nav = self._nav

        takeoff_requested = self._preflight(now)

        if self._home is None and nav.localised:
            if self.mission.home is not None:
                self._home = np.asarray(self.mission.home, dtype=float)
            else:
                # The launch point at cruise height, for the reason given in
                # MissionRunner.run: the launch point itself is below the fence.
                self._home = np.array(
                    [
                        nav.position[0],
                        nav.position[1],
                        max(self.mission.takeoff_height, self.config.geofence.min_corner[2]),
                    ]
                )

        state = self._state
        height = float(state.height) if state is not None else 0.0
        inputs = FailsafeInputs(
            now=now,
            link=self._link,
            estimate=nav,
            battery=float(state.battery) if state is not None else 1.0,
            temperature=float(state.temperature) if state is not None else float("nan"),
            height=height,
            flight_time=now - self._start_time,
            estimator_diverged=nav.position_sigma() > 2.0,
            operator_abort=self._abort,
            mission_complete=self._mission_complete(),
            takeoff_requested=takeoff_requested,
            airborne=state is not None and height > self.settings.landed_height,
            home_reached=self._home_reached(nav),
        )
        decision = self.failsafe.update(inputs)

        if decision.state is not self._flight_state:
            # Every transition resets the controller, as in the offline runner.
            self.controller.reset()
            self._hold_position = None
            self._hold_yaw = None
            if decision.state is FlightState.TAKEOFF:
                self._takeoff_at = now
            self._flight_state = decision.state
        self._publish_new_events()

        command = self._command_for(decision.state, now, dt)

        if decision.state is FlightState.TAKEOFF and not inputs.airborne:
            # Once, and again only if the aircraft has evidently not heard it. A
            # second takeoff during the climb is answered with an error.
            self._call("takeoff", now, every=3.0)
        if decision.command_land:
            self._call("land", now, every=self.command_resend)
        if decision.command_emergency:
            self._call("emergency", now, every=self.command_resend)
        if decision.send_rc:
            self.rc_publisher.publish(_rc_to_message(command, self.get_clock().now().to_msg()))

    def _preflight(self, now: float) -> bool:
        """Whether this cycle is the one that asks the failsafe for a take-off.

        The same gate as ``MissionRunner._preflight``: a fix and live telemetry,
        both, or no take-off. An aircraft that leaves the ground not knowing
        where it is has to find out while moving.
        """
        if not self._arm_requested or self.failsafe.state is not FlightState.IDLE:
            return False
        if self._armed_at is None:
            self._armed_at = now
        fresh = (
            self._link.packets_received > 0
            and self._link.last_state_age <= self.settings.link_timeout
            and self._state is not None
        )
        if self._nav.localised and fresh:
            self._arm_requested = False
            self.get_logger().info(
                f"localised after {now - self._armed_at:.1f} s on the ground, taking off"
            )
            return True
        if now - self._armed_at > self.preflight_timeout:
            self._arm_requested = False
            self.get_logger().error(
                f"no marker fix and live telemetry within {self.preflight_timeout:.0f} s; "
                "not taking off"
            )
        return False

    def _call(self, name: str, now: float, every: float) -> None:
        if now - self._last_sent[name] < every:
            return
        client = self._driver_clients[name]
        if not client.service_is_ready():
            self.get_logger().warning(
                f"driver service '{name}' is not available", throttle_duration_sec=2.0
            )
            return
        self._last_sent[name] = now
        client.call_async(Trigger.Request())

    def _command_for(self, state, now: float, dt: float) -> RcCommand:
        nav = self._nav
        if state in (FlightState.IDLE, FlightState.LANDED):
            self.controller.reset()
            return RcCommand.zero()
        if state in (FlightState.LAND, FlightState.EMERGENCY_LAND):
            # Vertical only, so the aircraft does not touch down moving sideways.
            self.controller.reset()
            return self.controller.to_rc(np.array([0.0, 0.0, self.descent.rate(state)]), 0.0)
        if state is FlightState.TAKEOFF:
            if self._takeoff_at is not None and now - self._takeoff_at < self.takeoff_settle:
                return RcCommand.zero()
            target = np.array([nav.position[0], nav.position[1], self.mission.takeoff_height])
            return self.controller.update(target, nav.yaw, nav, dt)
        if state is FlightState.HOLD:
            # Latched and accumulated, for the reasons in MissionRunner._command_for.
            if self._hold_position is None:
                self._hold_position = nav.position.copy()
                self._hold_yaw = float(nav.yaw)
            self._hold_yaw = wrap_angle(self._hold_yaw + self.settings.hold_search_rate * dt)
            return self.controller.update(self._hold_position, self._hold_yaw, nav, dt)
        if state is FlightState.RETURN:
            target = self._home if self._home is not None else nav.position
            return self.controller.update(target, self._auto_yaw(target, nav.yaw), nav, dt)

        if self._index >= len(self.mission.waypoints):
            self.controller.reset()
            return RcCommand.zero()
        waypoint = self.mission.waypoints[self._index]
        if self._waypoint_started is None:
            self._waypoint_started = now
            self.controller.start_waypoint()
        yaw = waypoint.yaw
        if yaw is None:
            yaw = self._resolved_yaw.get(self._index)
            due = now - self._yaw_replanned_at >= self.yaw_replan_interval
            if yaw is None or (
                due
                and not heading_is_adequate(
                    nav.position, yaw, self.marker_map, self.config, self.camera
                )
            ):
                yaw = self._auto_yaw(nav.position, nav.yaw)
                self._resolved_yaw[self._index] = yaw
            if due:
                self._yaw_replanned_at = now
        command = self.controller.update(waypoint.position, yaw, nav, dt)

        if self.controller.check_arrival(waypoint, nav, now):
            self._finish_waypoint(waypoint, now, "arrived")
        elif now - self._waypoint_started > waypoint.timeout:
            self._finish_waypoint(waypoint, now, "timeout")
        return command

    def _auto_yaw(self, position, current_yaw: float) -> float:
        return heading_for_markers(
            position, self.marker_map, self.config, self.camera, prefer=current_yaw
        )

    def _mission_complete(self) -> bool:
        return self._index >= len(self.mission.waypoints) and self._loop >= self.mission.loops

    def _home_reached(self, nav) -> bool:
        if self._home is None:
            return False
        delta = np.asarray(nav.position, dtype=float) - self._home
        return bool(np.linalg.norm(delta[:2]) < 0.5 and abs(delta[2]) < 0.25)

    def _finish_waypoint(self, waypoint, now: float, kind: str) -> None:
        message = WaypointEvent()
        message.header.stamp = self.get_clock().now().to_msg()
        message.index = int(self._index)
        message.name = waypoint.name
        message.kind = kind
        message.position_error = float(np.linalg.norm(self._nav.position - waypoint.position))
        self.waypoint_publisher.publish(message)
        self.get_logger().info(
            f"waypoint {waypoint.name} {kind} with {message.position_error:.2f} m of "
            "estimated error"
        )
        self._index += 1
        self._waypoint_started = None
        if self._index >= len(self.mission.waypoints):
            self._loop += 1
            if self._loop < self.mission.loops:
                self._index = 0
                self._resolved_yaw.clear()

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
            text = (
                f"failsafe {event.from_state.value} -> {event.to_state.value} "
                f"({event.cause.value}): {event.detail}"
            )
            # Two call sites, not one call through a chosen method: rclpy refuses
            # a logging call site whose severity changes between calls and the
            # first failsafe warning after an info took the node down.
            if event.cause is FailsafeCause.NONE:
                self.get_logger().info(text)
            else:
                self.get_logger().warning(text)


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
