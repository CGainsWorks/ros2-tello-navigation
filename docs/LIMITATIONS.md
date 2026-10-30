# Limitations

Everything this repository cannot do, in one place, written down deliberately
rather than discovered.

The first and largest is in its own document: **there is no aircraft**. See
`docs/HARDWARE.md` for the full list of what is asserted without evidence. The
rest of this file is about the design and it would still apply with a Tello on
the bench.

---

## It is a room-scale system and nothing more

The whole design assumes a small, surveyed, static indoor space.

* **Range.** A 300 mm marker through the Tello's camera is readable to about
  3.8 m and reliably readable to about 3.0 m
  (`results/detection_range.csv`). A larger space needs larger markers,
  proportionally: a 10 m hall needs roughly 800 mm symbols, which is a plotter
  job rather than a printer one.
* **Coverage.** The system has absolute position only when a marker is in view
  and `scripts/validate_map.py` exists because a layout with a hole in it
  localises beautifully for most of a mission and loses the aircraft in one
  corner. Between fixes it dead-reckons on a 10 Hz accelerometer, which is worth
  a few seconds and no more.
* **Static world.** The marker map is read once. A marker that moves, falls or is
  covered is not detected as wrong: if it moves 20 cm the estimator becomes
  confidently 20 cm wrong, with healthy residuals and nothing anywhere to
  indicate a problem.

## No obstacle sensing of any kind

The Tello has a downward time-of-flight sensor and nothing forward. The geofence
keeps the aircraft inside a box chosen to be empty; it does nothing about
anything inside that box. A person walking through the test area is both an
occlusion, which is handled and an obstacle, which is not.

## On link loss the aircraft may fly on for up to fifteen seconds

On link loss the controller falls silent so the aircraft's own dead-man timer can
land it, which is argued in `docs/FAILSAFE.md`. Until that timer fires nothing on
this side of the link can steer it.

This section used to say the aircraft hovers meanwhile, drifting under a metre.
It does not, in this repository's own models: the simulator and the emulator hold
the last `rc` command through an outage. The committed offline `link_loss` case
orbited the room for the whole outage; the emulator launch test, whose last
command was a cruise leg, left the room by several metres (its summary reports
the path flown). The geofence cannot help, because enforcing it needs the link.
If a real Tello behaves the same, the response is only safe in a space large
enough for fifteen seconds at cruise speed, or with a lower cruise speed. Which
behaviour the aircraft has is a hardware check (`docs/HARDWARE.md`, item 6) and
the obvious mitigation, commanding a hover as soon as telemetry starts to go
stale and before declaring the link lost, is not built.

## The visual odometry has no scale of its own

Monocular flow gives velocity over depth. The depth comes from the marker map and
the filter's own position, so the velocity estimate is only as good as that
depth and `VoEstimate.scale_source` records where each one came from.

It also assumes the tracked features lie on the plane it was told about. Fly at a
wall with a table in front of it and the estimate is biased by roughly the ratio
of the two depths. The inlier ratio catches gross violations, because a second
depth layer produces flow a single-plane model cannot fit and the estimate is
down-weighted accordingly. That is not the same as knowing.

## The time-of-flight sensor measures the floor beneath the aircraft

Which is world z only over a flat floor. The room this targets has one. A room
with a table in it does not and the filter would take the table top as the floor
and put the aircraft 70 cm too low.

## Single-marker fixes are much worse than pairs

A single square marker leaves its own tilt almost unobservable when viewed head
on, so half a pixel of corner noise swings the solved pose by several degrees,
which at 1.5 m is about 15 cm of lateral error. The two IPPE branches are
genuinely indistinguishable in that geometry.

This is handled honestly rather than hidden: the fix is published at the midpoint
of the two branches with a covariance widened to span them, so the filter is told
exactly which direction not to trust. But the right answer is to see two markers,
which is why the shipped map is built from pairs 45 cm apart and why
`scripts/validate_map.py` reports two-marker coverage separately.

## The results are simulation results

Every accuracy figure in `docs/RESULTS.md` that needs ground truth is a statement
about the error model in `src/tello_core/sim` and is labelled as one. The figures
that would carry over unchanged to a real flight are the ones that need no truth:
localisation availability, fix rate, reprojection residual and the static survey
check. `src/tello_core/tello_core/eval/metrics.py` separates the two kinds
explicitly and `evaluate()` omits the ground-truth metrics entirely rather than
reporting them as NaN, so a results file from a real flight cannot be read as if
it contained an absolute trajectory error.

## Performance

The simulator renders about ten frames a second on one core of a laptop, which is
roughly three times slower than real time including detection. That is fine for
the offline runner, which controls its own clock and it is why the ROS 2 graph's
simulated link is explicitly not reproducible: inside a graph the step sizes
depend on scheduling.

A real flight has the opposite problem. Detection is about 50 ms a frame at
960x720, so a laptop sustains roughly 15 to 20 Hz, not the camera's 30. The
`detect_every` setting exists for that and its value is recorded in every results
file, because raising it trades localisation rate for throughput and a reader
needs to know which was used.

## CI runs one Python version

`.github/workflows/ci.yml` tests Python 3.11 only. A matrix over three versions
would triple the bill on a free account to catch a class of problem this code does
not have: it is plain Python with numpy and OpenCV and the behaviour that matters
is pinned by the test suite. Development happens on 3.9 and 3.11 and the suite
passes on both.

## The ROS 2 side is thin and tested against an emulator, not an aircraft

The nodes in `src/tello_localisation`, `src/tello_control` and `src/tello_driver`
subscribe, convert, call into `tello_core` and publish. That is deliberate: it is
what lets everything of substance be tested without a ROS installation.

Until the emulator existed the node layer was checked only structurally. The
first launch showed what that missed: the node packages installed no code, the
mission node never commanded a take-off, the fusion node fed the filter a zero
body velocity and a launch argument was silently overridden by a parameter file.
`docs/EMULATOR.md` describes the launch tests that now build the workspace with
colcon and fly the graph in CI.

What remains true:

* **The ROS graph runs in real time and is not reproducible.** Scheduling sets
  the step sizes, so its numbers vary from run to run. The reproducible numbers
  come from the offline runner and the launch tests assert bounds rather than
  compare with committed values.
* **The ROS mission node mirrors `MissionRunner` rather than sharing it.** The
  offline runner owns a link and a filter; the node gets both over topics. The
  launch tests keep them honest; even so a change to one has to be made to the
  other.
* **The ROS graph has no visual odometry.** The offline runner fuses VO; the
  graph fuses marker fixes with the state packet only.
* **The emulated camera is slow.** The numpy ray caster delivers a handful of
  frames per second on a CI runner, against the Tello's 30. Its render time
  is compensated so real camera latency is not modelled. See `docs/EMULATOR.md`.
* **The launch tests are simulated end to end.** Aircraft, room, camera and link
  are the emulator's. They show that the graph, the driver and the decoder work
  together; they say nothing about a Tello.

## Things deliberately not built

* **A map-building mode.** Estimating marker positions from flight data instead of
  surveying them is a reasonable thing to want and it is a different project. The
  survey is the thing the accuracy is measured against and estimating it from the
  same data would make the measurement circular.
* **A second camera or a downward marker view.** Would remove the heading
  constraint entirely, which is the single biggest limitation of the design: on
  this aircraft, where the camera points decides whether it knows where it is.
* **Recorded-video replay.** The brief asks for testing against recorded video
  before flying. The simulator covers that need better for a project with no
  aircraft, because it can generate any geometry on demand and there is no
  recorded video to replay. `ReplayLink` is the obvious addition the moment real
  footage exists and the `TelloLink` interface is shaped for it.
