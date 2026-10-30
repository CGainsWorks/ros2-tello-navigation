# The Tello emulator

`src/tello_core/tello_core/emulator` is a simulated DJI Tello that speaks the
Tello SDK 2.0 UDP protocol. The real driver (`UdpLink` with the PyAV decoder,
inside the real ROS 2 driver node) talks to it exactly as it would talk to an
aircraft: the same three ports, the same text commands and the same state strings,
plus an H.264 stream it has to decode.

**Everything it produces is simulated.** It is a substitute for the aircraft
built from the same assumptions as the driver, so it can show that the driver,
the decoder and the ROS graph work together. It cannot show that any of them work
with a Tello. [HARDWARE.md](HARDWARE.md) lists what only an aircraft can settle.

```
   ROS 2 graph (unchanged hardware.launch.py)            emulator process
  +----------------------------------------+    +-------------------------------+
  | driver_node: UdpLink + H264Decoder     |    | command socket   127.0.0.1:8889|
  |   "command", "takeoff", "rc a b c d" --+--->|   SDK parser, replies          |
  |   <-- "ok" / "error" / values ---------+----|                                |
  |   state socket :8890  <----------------+----| state strings, 10 Hz           |
  |   video socket :11111 <----------------+----| H.264 (libx264 via PyAV),      |
  |                                        |    |   <= 1460-byte datagrams       |
  | marker_node, fusion_node, mission_node |    | TelloDynamics + ray caster     |
  +----------------------------------------+    | ground truth -> CSV, not UDP   |
                                                +-------------------------------+
```

## Run it

```bash
python -m tello_core.emulator --map maps/marker_map.yaml --mission missions/hop.yaml
# or, in a colcon workspace
ros2 run tello_core tello_emulator --map maps/marker_map.yaml --mission missions/hop.yaml
# the whole graph against it
ros2 launch tello_bringup emulator.launch.py auto_arm:=true
```

`--mission` only chooses the launch pose, as `scripts/run_mission.py` does: the
first waypoint's x and y, facing the markers visible from there.

## What it answers

| Command | Reply |
| --- | --- |
| anything before `command` | nothing |
| `command` | `ok`; state starts flowing to port 8890 on the sender's host |
| `takeoff` | `ok` when the climb has finished, `error` if already flying |
| `land` | `ok` on touchdown, `error` if not flying |
| `emergency`, `stop` | `ok` at once |
| `streamon`, `streamoff` | `ok`; H.264 to port 11111 on the sender's host |
| `rc a b c d` | never answered; ignored if malformed or out of range |
| `forward`/`back`/`left`/`right`/`up`/`down` 20 to 500 | `ok` when the move is done |
| `cw`/`ccw` 1 to 360 | `ok` when the turn is done |
| `speed 10` to `100` | `ok` |
| `battery?` `speed?` `time?` `height?` `tof?` `temp?` `baro?` `attitude?` `acceleration?` | values from the simulated state |
| `wifi?` `sdk?` `sn?` | `error Not supported by the emulator` (no made-up values) |
| anything else | `unknown command: <word>` |

Replies come from port 8889 to whichever address and port sent the command. The
aircraft lands itself after fifteen seconds without a command. The exact error
strings are the emulator's; the SDK document does not fix them all.

## Video

Frames are rendered by the simulator's ray caster (`sim/render.py`) with motion
blur, vignetting and sensor noise but without its JPEG stage, because H.264 is now
the compression: libx264 through PyAV, `ultrafast`/`zerolatency`, no B-frames, a
keyframe every 30 frames with the parameter sets repeated on each so a decoder
can join mid-stream. The bitrate and keyframe interval are emulator settings, not
figures from an aircraft. Each encoded frame is cut into datagrams of at most 1460
bytes, the size widely described for the Tello; a shorter datagram ends a frame.

The decoder (`link/video.py`) runs every datagram through FFmpeg's H.264 parser
and, by default, flushes the parser on a short datagram so a frame is handed over
the moment its last datagram arrives. Without that hint the parser holds each
frame until the next begins, a frame of latency on every marker fix.

The numpy ray caster is slow, a handful of 960x720 frames per second on one core,
well short of the Tello's 30. Two things follow and both are stated here rather
than hidden:

* **Its render time is compensated.** Each frame is rendered at the true pose
  extrapolated to when it is expected to be sent, so the stream carries the
  latency of the link and the decoder, not of the renderer. Without that the
  launch tests saw marker fixes describe where the aircraft had been a few
  hundred milliseconds earlier and the failsafe hold for lost localisation. A
  real camera's own capture-to-send latency is therefore **not** modelled.
* **The achieved frame rate is lower than a Tello's.** The emulator reports it
  (`video_fps_achieved` in its stop message) and `--render-workers N` renders in a
  process pool on a machine with cores to spare. On a four-core CI runner the
  render pool competes with the marker detector for the same cores, so the
  launch tests render in-thread.

## Faults

All times are seconds after the first accepted `takeoff`, because time on the
ground waiting for a first fix varies from run to run in real time.

* `--outage START DURATION`: nothing gets through in either direction. Commands
  are logged as dropped. The encoder keeps running, so the decoder meets the gap
  as it would on a real link: missing reference frames until the next keyframe.
* `--video-outage START DURATION`: video only.
* `--battery`, `--battery-drain`: starting charge and an in-flight drain
  multiplier, the same meaning as in `scripts/run_mission.py`.
* `--state-loss`, `--frame-loss`: random loss probabilities.

## Ground truth

`--truth PATH` writes the simulated true pose, velocity, battery and link state at
20 Hz to a CSV. `--events PATH` writes every datagram received or dropped, every
reply and every automatic action as JSON lines. Neither goes near the Tello ports,
so nothing in the graph can read the truth; the launch tests read the files after
the flight.

## The launch tests

`src/tello_bringup/test/` holds three `launch_testing` tests, run by `colcon test`
inside `docker/ros2/Dockerfile` (ROS 2 Humble) locally and in CI:

| Test | Fault | Asserts |
| --- | --- | --- |
| `test_emulator_nominal_launch.py` | none | both waypoints arrived, LANDED with no cause, true arrival error under 0.40 m, estimate RMS against truth under 0.25 m, the graph (not the dead-man timer) landed it |
| `test_emulator_link_loss_launch.py` | outage 4 s after take-off for 11 s | LINK_LOSS to LAND, LANDED, no `rc` reached the aircraft between the link returning and touchdown |
| `test_emulator_low_battery_launch.py` | 30 per cent, drain x8 | LOW_BATTERY, never CRITICAL_BATTERY, LANDED within 0.8 m of the launch point |

The bounds are acceptance criteria, not results. Each test writes
`results/ros2/emulator/<scenario>/summary.json` with what it measured, labelled
simulated. CI uploads the directory as an artifact.

```bash
docker build -f docker/ros2/Dockerfile -t tello-ros2 .
docker run --rm --network none tello-ros2
```

`--network none` is worth keeping on a machine running other ROS containers: the
image already confines DDS discovery to loopback, which is what stopped another
project's `/rc` and `/nav_state` arriving in this graph during development.
