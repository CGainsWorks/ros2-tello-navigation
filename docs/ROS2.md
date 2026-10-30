# The ROS 2 side

Nothing in this repository needs ROS 2. That is the arrangement, not an apology:
every algorithm lives in `src/tello_core`, which is plain Python and the ROS
packages are wrappers that subscribe, convert, call into it and publish. See
`docs/ARCHITECTURE.md` for why.

If you have a ROS 2 distribution, this is how to build and run the graph. If you
do not, `make demo` runs the same flight logic with the same configuration and you
are not missing any of the substance.

Built and launch-tested against **Humble** on Ubuntu 22.04, in
`docker/ros2/Dockerfile`, on every push. Jazzy is untested.

---

## Build

```bash
# from the repository root, which is already a colcon workspace
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
pip install -r docker/ros2/requirements.txt   # PyAV and an OpenCV that suits numpy 1.x
colcon build --symlink-install
source install/setup.bash
```

Or build exactly what CI builds:

```bash
docker build -f docker/ros2/Dockerfile -t tello-ros2 .
docker run --rm --network none tello-ros2          # the launch tests
```

`--symlink-install` is worth it here: the Python packages are pure Python, so a
code change needs no rebuild.

---

## Run, against the emulator

```bash
ros2 launch tello_bringup emulator.launch.py auto_arm:=true
```

The Tello emulator (a simulated aircraft speaking SDK 2.0 on 127.0.0.1:8889, see
[EMULATOR.md](EMULATOR.md)) plus `hardware.launch.py`, included unchanged with the
driver pointed at loopback. Every packet between the graph and the aircraft is a
real UDP datagram in the Tello's format and the video is real H.264. This is the
graph the launch tests fly.

## Run, against the in-process simulator

```bash
ros2 launch tello_bringup sim.launch.py
```

That brings up four nodes:

| Node | Package | Does |
| --- | --- | --- |
| `tello_driver` | `tello_driver` | Owns the link. Publishes `tello_state`, `image`, `imu`, `battery`, `link_status`; subscribes `rc`; serves `takeoff`, `land`, `emergency`. |
| `tello_marker_localisation` | `tello_localisation` | `image` in, `fix` out. |
| `tello_fusion` | `tello_localisation` | `tello_state` and `fix` in, `nav_state` and the `map` to `base_link` transform out. |
| `tello_mission` | `tello_control` | `nav_state`, `link_status` and `tello_state` in, `rc` out, calls the driver's `takeoff`/`land`/`emergency`, plus the failsafe. |

Then arm it:

```bash
ros2 service call /tello_mission/arm std_srvs/srv/Trigger
```

and watch:

```bash
ros2 topic echo /nav_state
ros2 topic echo /failsafe_event
ros2 run rqt_image_view rqt_image_view /image
```

To abort:

```bash
ros2 service call /tello_mission/abort std_srvs/srv/Trigger
```

### Against a real aircraft

```bash
ros2 launch tello_bringup hardware.launch.py
```

which is the same graph with `link_type:=udp`. **Read `docs/HARDWARE.md` first.**
The UDP driver has never exchanged a packet with a Tello, only with the emulator,
and the node logs a warning saying so every time it starts.

---

## The simulated link inside ROS is not reproducible

Worth knowing before trusting a number from it.

The offline mission runner advances simulated time explicitly, which is what lets
a flight run faster than real time and identically every time. Inside a ROS graph
nothing calls `advance`, so the driver node wraps the simulator in an adapter that
steps it by however much wall clock has actually passed since the last poll. Step
sizes then depend on scheduling and two runs are not identical.

That is the right trade for a graph whose purpose is to exercise the ROS plumbing.
The reproducible numbers come from the offline runner and `docs/RESULTS.md` says
which is which.

---

## Topics and messages

```
                        ┌──────────────┐
                        │ tello_driver │
                        └──┬────┬───┬──┘
          image (Image)    │    │   │  link_status (LinkStatus)
         ┌──────────────────┘    │   └──────────────────┐
         │   tello_state         │                      │
         ▼   (TelloState)        ▼                      │
┌─────────────────┐     ┌────────────────┐              │
│ marker_node     │     │ fusion_node    │              │
│                 ├────►│                │              │
└─────────────────┘ fix └───────┬────────┘              │
     ▲      (MarkerFix)         │ nav_state (NavState)  │
     │ nav_state                │  + tf map->base_link  │
     └──────────────────────────┤                       │
                                ▼                       ▼
                        ┌───────────────────────────────────┐
                        │ mission_node                      │
                        │ waypoint controller + failsafe    │
                        └───────┬──────────┬────────────┬───┘
                      rc        │          │            │
                   (RcCommand)  │  failsafe_event   waypoint_event
                                ▼
                          back to the driver
```

`nav_state` goes to the marker node as well as to the mission node and only for
one purpose: it supplies the prior that resolves the planar pose ambiguity for a
single-marker view. It is optional. Without it the marker node rejects more
single-marker fixes and publishes fewer.

The seven messages are in `src/tello_msgs/msg`. `TelloState` carries every field
the state decoder produces; it replaced a `sensor_msgs/Imu` path that had nowhere
to put the height or the body velocity and fed the filter zeros for both. `imu`
and `battery` are still published for standard tools. `MarkerFix` and `NavState` both
carry a real covariance, because everything downstream of them needs to know how
much to believe them.

---

## Parameters worth knowing

Set in `src/tello_bringup/config/nodes.yaml` and overridable at launch.

| Node | Parameter | Default | Note |
| --- | --- | --- | --- |
| `tello_driver` | `link_type` | `sim` in `sim.launch.py` | `sim` or `udp`. |
| | `tello_address` | `192.168.10.1` | `127.0.0.1` in `emulator.launch.py`. |
| | `local_command_port` | 8889 | `0` against the emulator, which already holds 8889 on the same machine. |
| | `image_encoding` | `bgr8` | `mono8` against the emulator, whose camera is monochrome. |
| | `poll_rate` | 60 Hz | How often the driver drains the link. Above the video rate so a frame is never held. |
| `marker_node` | `marker_map` | the shipped map | Required. The node refuses to start without it: defaulting to an empty map would mean flying with an estimator that silently never fixes. |
| | `detect_every` | 1 | Detect on one frame in N. The first thing to raise on a machine that cannot keep up and it costs localisation rate directly. |
| `fusion_node` | `fix_timeout` | 2.0 s | How long `localised` stays true without a fix. |
| | `max_position_sigma` | 0.45 m | Above this the filter reports itself unlocalised regardless. |
| `mission_node` | `mission` | required | |
| | `auto_arm` | false | A launch argument. True takes off once localised with live telemetry. Convenient in simulation, a bad idea with an aircraft on the bench. It is deliberately absent from `nodes.yaml`: a node-specific entry there would override the launch argument. |
| | `control_rate` | 20 Hz | |

---

## What is checked and what is not

`tests/test_ros_packages.py` runs on every push with no ROS installed. It parses
every `package.xml`, checks each names the package its directory does and declares
the right build type, that each node package has the `__init__.py` without which
`find_packages()` installs nothing, that every console script points at a module
and a function that exist, that every node parses and defines `main` and that every
message file is listed in `CMakeLists.txt`.

The `ROS 2 graph against the emulator` CI job then builds the workspace with
colcon in `docker/ros2/Dockerfile` and runs the three launch tests in
`src/tello_bringup/test/`: a waypoint mission, a link loss and a low battery,
flown by the full graph against the emulator and asserted against its simulated
ground truth. [EMULATOR.md](EMULATOR.md) lists what each asserts.

The first launches found six faults the static checks could not: node packages
that installed no code, a mission node that never commanded a take-off or a
landing, a fusion node feeding the filter zeros, an attribute that shadowed an
rclpy internal, a logging call site rclpy refused because its severity changed
and a launch argument overridden by the parameter file. All six are fixed and the
first is now also a static check.

What none of this checks is a Tello. See [HARDWARE.md](HARDWARE.md).
