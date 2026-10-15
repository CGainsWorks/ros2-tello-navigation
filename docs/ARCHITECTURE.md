# Architecture

## The shape of it

```
                    ┌──────────────────────────────────────────┐
                    │              maps/marker_map.yaml        │
                    │   18 surveyed markers, 4.0 x 3.0 x 2.4 m │
                    └───────────────────┬──────────────────────┘
                                        │ the survey
                                        ▼
  ┌──────────┐  960x720  ┌───────────────────────┐   MarkerFix    ┌──────────────┐
  │          │  30 Hz    │  QR detect + solvePnP │  p, yaw, 3x3   │              │
  │  camera  ├──────────►│  aruco, classic       ├───────────────►│              │
  │          │           │  fallback, sub-pixel  │   covariance   │     EKF      │
  └──────────┘           └───────────┬───────────┘                │              │
                                     │ frames                     │  p v yaw     │
                                     ▼                            │  accel bias  │
                         ┌───────────────────────┐  VoEstimate    │  gyro bias   │
                         │  visual odometry      ├───────────────►│              │
                         │  de-rotated LK flow   │  body velocity │   11 states  │
                         └───────────────────────┘                │              │
  ┌──────────┐  state    ┌───────────────────────┐                │              │
  │ telemetry├──────────►│  decode: cm, dm/s,    ├───────────────►│              │
  │  10 Hz   │  string   │  deg -> SI, REP-103   │  accel, att,   └──────┬───────┘
  └──────────┘           └───────────────────────┘  v_body, tof          │
                                                                         │ NavState
                                                    ┌────────────────────▼───────┐
                                                    │   failsafe state machine   │
                                                    │   8 states, 11 causes      │
                                                    │   decides: may it fly?     │
                                                    └────────────┬───────────────┘
                                                                 │ permission
                                                    ┌────────────▼───────────────┐
                              waypoints ───────────►│   waypoint controller      │
                              + heading planner     │   PID -> world velocity    │
                                                    │   -> body -> rc counts     │
                                                    └────────────┬───────────────┘
                                                                 │ rc a b c d
                                                                 ▼
                                                            the aircraft
```

Two things about that diagram are the design rather than the drawing.

**The markers are the only absolute measurement.** Everything else, the
accelerometer and the visual odometry alike, says how the state *changed*. Only a
marker fix says where the aircraft *is*. That is why the fix carries a real
covariance rather than a tuned constant, why the heading planner exists at all
and why losing sight of the markers is a failsafe condition rather than an
inconvenience.

**The failsafe sits between the estimate and the command**, not beside them. It
reads the estimate, decides whether the aircraft may be commanded and only then
is a command computed. Reversing those two steps is the tempting simplification
and it is how a system flies one more cycle on an estimate it has already decided
not to trust.

---

## Why there is a `tello_core`

The brief asks for four ROS 2 packages. There are six directories under `src/`
and the extra one is the point.

Every algorithm lives in `src/tello_core`, which is plain Python and imports no
ROS. The ROS 2 packages are wrappers: they subscribe, convert, call into
`tello_core` and publish. Nothing of substance is in them.

That split buys three things:

1. **The test suite runs anywhere.** Over three hundred tests in about half a
   minute, on any machine with Python and OpenCV. A reviewer can check this
   repository without installing a ROS distribution and CI costs a pip install
   rather than a container.
2. **The interesting code is testable at all.** A node is awkward to unit-test. A
   function that takes an image and returns a pose with a covariance is not.
3. **One implementation of the flight logic.** The offline mission runner, the
   ROS graph and the test suite all drive the same failsafe machine and the same
   controller. There is no simulated copy to drift out of step.

The cost is one extra package to build and a `sys.path` line at the top of each
node. `tests/test_ros_packages.py` checks that line is before the import, since
the whole arrangement depends on it.

### The packages

| Directory | Build type | What is in it |
| --- | --- | --- |
| `src/tello_core` | ament_python | Everything: links, detection, pose, fusion, control, failsafe, simulator, evaluation. No ROS. |
| `src/tello_msgs` | ament_cmake | Six message definitions. |
| `src/tello_driver` | ament_python | The one node that talks to hardware. Switches to the simulator with one parameter. |
| `src/tello_localisation` | ament_python | `marker_node`, `fusion_node`. |
| `src/tello_control` | ament_python | `mission_node`: the controller and the failsafe, in one process. |
| `src/tello_bringup` | ament_python | Launch files and all configuration. |

### Why the controller and the failsafe share a process

The obvious ROS decomposition puts the failsafe in its own node publishing a
mode and the controller in another subscribing to it. It is the wrong
decomposition here.

The failsafe's whole purpose is to veto a command and a veto that travels over a
topic arrives at best one queue later. On a link-loss trip that delay is spent
transmitting stick commands the failsafe has already decided must stop, which is
precisely the behaviour `docs/FAILSAFE.md` explains must not happen. So the
machine and the controller share a process and a function call and the node
publishes the failsafe's decisions for observation rather than for control.

---

## The estimator

An extended Kalman filter over eleven states:

```
[ p(3)  v(3)  yaw  accel_bias(3)  gyro_bias ]
```

**Roll and pitch are inputs, not states.** The Tello runs its own attitude filter
and publishes roll and pitch that are well observed, because gravity is a large
constant vector a stationary accelerometer sees directly. Re-estimating them would
mean fighting a filter with better data.

**Yaw is a state**, because nothing on this aircraft observes absolute heading.
The magnetometer is not exposed over the SDK and would be useless indoors anyway,
so the Tello's yaw is an integrated rate that drifts and the markers are the only
thing that pins it down.

**The accelerometer bias is a state because it has to be.** Ten-hertz
accelerometer data integrated twice turns a 0.05 m/s² bias into 25 cm of position
error in ten seconds, which is longer than this aircraft routinely goes without
seeing a marker.

Three details that are not optional:

* **Joseph-form covariance update.** The short `(I - KH)P` form is algebraically
  identical and numerically is not: over a whole mission at 10 Hz it loses
  symmetry and eventually positive definiteness.
* **Out-of-order measurements.** A marker fix is only available after detection
  and the pose solve have run, tens of milliseconds after the frame arrived, by
  which time more telemetry has been processed. A fix whose timestamp is in the
  past is the normal case, not the exception and the filter rewinds to it and
  replays rather than pretending it is fresh. At 0.8 m/s and 30 ms, pretending
  costs 2.4 cm, which is comparable with the fix's own accuracy.
* **A covariance floor and a rejection counter.** Both are in
  `src/tello_core/tello_core/localisation/fusion.py` with the full reasoning;
  between them they are the difference between a filter that works and one that
  deadlocks. See `docs/RESULTS.md`.

---

## The detection path

Frame in, `MarkerFix` out, in four steps.

1. **Detect.** `cv2.QRCodeDetectorAruco` first, `cv2.QRCodeDetector` as well on
   any frame where the first found a symbol it could not read. The two have
   opposite strengths and the repository needs both; see the module docstring of
   `tello_core/localisation/qr.py`.
2. **Refine.** `cv2.cornerSubPix` on every corner. A pixel of corner error at 2 m
   is about 6 mm of position error and this is the dominant term in the whole
   pipeline.
3. **Solve.** One marker goes through `SOLVEPNP_IPPE_SQUARE` via
   `solvePnPGeneric`, so both branches of the planar ambiguity are available; two
   or more go through a single `SOLVEPNP_SQPNP` over all the corners at once.
   Deliberately not "solve each marker and average the poses": averaging weights a
   marker at 3 m the same as one at 1 m.
4. **Covariance.** The projection Jacobian from `cv2.projectPoints`, the measured
   reprojection residual as the pixel noise, `sigma^2 (J^T J)^-1`, pushed through
   to world position and yaw by finite differences, plus the markers' own survey
   uncertainty. The result grows when the aircraft is far from the markers, when
   it sees them obliquely and when it sees fewer of them, which is what the filter
   and the failsafe need in order to do anything sensible.

---

## The simulator

`src/tello_core/tello_core/sim` is the substitute for hardware and it is built to
be wrong in the ways a Tello is wrong rather than realistic in general.

* **`world.py`** builds the room as six textured rectangles and prints the real
  QR symbols onto them at their surveyed positions.
* **`render.py`** is an exact nearest-hit ray caster, so occlusion is correct,
  followed by motion blur, vignetting, signal-dependent sensor noise and a JPEG
  round trip, in the order a camera applies them.
* **`dynamics.py`** is a first-order velocity lag with a rate limit, an
  Ornstein-Uhlenbeck disturbance, attitude coupling, accelerometer bias and noise,
  battery drain and ground contact.
* **`link.py`** produces telemetry at 10 Hz and video at 30 Hz on a virtual
  clock, serialises the telemetry into genuine Tello state strings and parses them
  back with the real decoder, drops and corrupts packets, takes the link down on a
  schedule and lands the aircraft itself after fifteen seconds without a command.

Frames are rendered lazily, on poll rather than on emission, because a 20 Hz
control loop against a 30 Hz camera discards a third of them and drawing those was
a third of the cost of a simulated flight.

The one thing the simulator never does is tell the flight software where the
aircraft is. Ground truth reaches the log through the recorder, not the estimator.

---

## Where every number lives

| Thing | File |
| --- | --- |
| Marker positions, sizes, survey provenance | `maps/marker_map.yaml` |
| Airframe, camera mount, rc scaling, battery policy, geofence | `src/tello_bringup/config/aircraft.yaml` |
| Controller gains and limits | `src/tello_bringup/config/control.yaml` |
| Waypoints | `missions/*.yaml` |
| Filter noise model and gating | `FusionSettings` in `fusion.py` |
| Failsafe thresholds | `FailsafeSettings` in `failsafe.py` |
| Camera intrinsics | derived in `camera.py`, overridable from YAML |

Nothing is tuned in two places. Every results file records which configuration
produced it, including the camera's `source` field and the map's `survey.method`,
so an accuracy figure cannot be read without knowing what it was measured
against.
