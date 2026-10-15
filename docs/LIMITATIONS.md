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

## The link-loss response costs up to fifteen seconds of drift

On link loss the controller falls silent so the aircraft's own dead-man timer can
land it, which is the right call and is argued in `docs/FAILSAFE.md`. The cost is
that the aircraft hovers, drifting, until that timer fires.

At the measured 5 cm/s of drift that is under a metre, which is acceptable in a
4 x 3 m room with the geofence inset 0.35 m from the walls. In a larger space, or
one with any air movement, it would not be. There is no way to shorten it from
this side of the link.

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

## The ROS 2 side is thin and largely untested

The nodes in `src/tello_localisation`, `src/tello_control` and `src/tello_driver`
subscribe, convert, call into `tello_core` and publish. That is deliberate: it is
what lets everything of substance be tested without a ROS installation and it is
what makes the offline runner and the ROS graph exercise one implementation of the
flight logic.

The consequence is that the node layer itself is checked structurally rather than
behaviourally. `tests/test_ros_packages.py` parses every `package.xml`, every
`setup.py` entry point and every message definition and verifies that each node
module parses and defines `main`, without importing `rclpy`. What it does not do
is launch a graph and watch messages flow, because that needs a ROS distribution
that CI does not install. `docs/ROS2.md` has the colcon build for anyone who has
one.

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
