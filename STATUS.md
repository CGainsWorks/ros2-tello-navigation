# Status

## What state this is in

A complete GPS-denied indoor navigation stack that flies autonomous waypoint
missions and lands itself when anything goes wrong, **tested entirely in
simulation**. There is no aircraft. See
[docs/HARDWARE.md](docs/HARDWARE.md).

Everything the brief asked for is built and measured. The one thing it asked for
that is not here is a flight.

---

## Build checklist, from the brief

- [x] **Connect to the Tello and stream video and telemetry.** The `TelloLink`
      interface with three concerns separated: a Tello SDK 2.0 UDP driver, the
      simulator and the state-string decoder they share. The driver is written
      and has never been run against an aircraft, which is stated wherever it
      appears, including in a warning the node logs at startup.
- [x] **QR detection giving a pose from a single marker.** Both OpenCV detectors,
      used together because they have opposite failure modes, sub-pixel corner
      refinement, `SOLVEPNP_IPPE_SQUARE` through `solvePnPGeneric` so both
      branches of the planar ambiguity are available and a covariance from the
      projection Jacobian rather than a tuned constant.
- [x] **Marker map file format and multi-marker localisation.** A strict loader
      that refuses any map that is not fully determined, survey provenance carried
      into every results file and a single `SOLVEPNP_SQPNP` over every corner of
      every visible marker rather than an average of per-marker poses.
- [x] **IMU and visual-odometry fusion.** An eleven-state EKF with an
      accelerometer bias, Joseph-form updates, chi-square gating, an out-of-order
      measurement path, a covariance floor and a rejection counter. The
      visual-odometry front end is de-rotated Lucas-Kanade flow with the scale
      taken from the marker map and recorded.
- [x] **PID waypoint controller.** Derivative on measurement, back-calculation
      anti-windup, explicit `dt` and a yaw limit set from a measured motion-blur
      curve rather than chosen as a control number.
- [x] **Failsafe state machine: link loss, low battery, lost localisation.** Eight
      states, eleven cause codes, a documented priority order, monotone severity
      and the counter-intuitive link-loss response of falling silent so the
      aircraft's own dead-man timer can act. Every case flown and the evidence
      committed.
- [x] **Full autonomous waypoint mission.** `make demo`. Four corners of the room
      with automatic heading, then a landing.
- [x] **Position error evaluation against surveyed markers.** `make survey`,
      which is the one accuracy measurement here that would mean the same thing on
      hardware.

Beyond the checklist, because the work needed them:

- [x] A simulator that stands in for the aircraft: an exact ray caster over a
      textured room with the real QR symbols printed on it, a lossy link and
      flight dynamics with lag, drift and bias.
- [x] Measured detector curves against range, turn rate and viewing angle. Three
      of the configuration's most important numbers come from them.
- [x] A marker-map coverage check that fails the build if any part of the flyable
      area cannot see a marker.
- [x] Printable marker sheets with crop marks at the symbol's own corners.
- [x] A tuning sensitivity sweep, because the plant the gains were tuned against
      has never been measured.
- [x] 326 tests, about a minute, no ROS needed.

---

## Definition of done, from the brief

- [x] **The whole thing runs from one command on a clean machine.**
      `make demo`, or `docker compose up` with no Python installed.
- [x] **The README opens with a diagram, not a wall of text.**
- [x] **Every number quoted in the README is reproducible from a script.**
      Stronger than that: no number in the README or in `docs/RESULTS.md` is typed
      in by hand. `scripts/make_results.py` generates both from files under
      `results/` and `scripts/check_reproducible.py` compares a fresh flight with
      the committed metrics inside stated tolerances.
- [x] **Limitations are stated honestly in their own section.**
      [docs/LIMITATIONS.md](docs/LIMITATIONS.md), plus
      [docs/HARDWARE.md](docs/HARDWARE.md) for the specific case of there being no
      aircraft.
- [x] **A licence is declared.** MIT.
- [x] **Large binaries live in Releases or LFS, not in the Git history.**
      `.gitattributes` routes bags, video and archives to LFS. The committed
      results are CSV and JSON, a few tens of kilobytes; the figures are
      regenerated from them by `make figures`.
- [ ] **The repository URL is added to the matching CV bullet point.** Not done
      here; it is a change to another document.

---

## What is next, in order

1. **Fly it.** Everything on the list in
   [docs/HARDWARE.md](docs/HARDWARE.md), starting with reading the state packet on
   a table, which takes ten minutes and settles the two assumptions that would
   silently corrupt the estimator.
2. **Survey a room.** An afternoon and the work that turns every number in
   `docs/RESULTS.md` from a statement about a file into a statement about a room.
3. **A video decoder for the UDP driver.** PyAV or an `ffmpeg` subprocess. Nothing
   else about the hardware path can be checked until frames arrive.
4. **Replay from recorded flights.** The `TelloLink` interface is shaped for it and
   it is not built, because there is no recording to replay. The moment there is
   real footage this becomes the most valuable thing in the repository: a
   regression suite made of real frames.
5. **Re-measure the detector curves on real frames.** The simulated curves set the
   marker size, the yaw limit and the planner's incidence limit. The real numbers
   will be worse and those three choices depend on them.

---

## Known weaknesses

Not a list of excuses, a list of the things that would be worth a reviewer's
attention. The full version is [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

* **Every ground-truth accuracy figure is a simulation result** and the error
  model it is measured against was written by the same person who wrote the
  estimator. The figures that do not need truth are the honest ones and they are
  labelled.
* **The filter is still somewhat overconfident.** The consistency ratio, error over
  reported sigma, is above one. That is the direction that matters, because the
  failsafe reads that number. It is reported rather than tuned away and the
  covariance floor is the mitigation.
* **Trajectory error is an order of magnitude worse than fix accuracy.** Fixes are
  good to about a centimetre; the flown trajectory is good to about fifteen. The
  gap is the fix rate, the covariance floor and controller lag, all of which are
  explained in `docs/RESULTS.md` and the arrival error is the figure to read for
  "can it fly to a point".
* **No obstacle sensing.** The geofence keeps the aircraft inside a box chosen to
  be empty.
* **The ROS 2 node layer is checked structurally, not behaviourally.** No test
  launches a graph, because that needs a ROS distribution CI does not install.

---

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml), built for a free account.

Every push runs lint, 315 fast tests, the marker-map coverage check, every mission
file, the ROS packaging check, a twenty-second smoke flight and a link-loss
failsafe flight: about four minutes. Runs are cancelled when superseded,
documentation-only pushes are skipped and every job has a timeout.

The weekly schedule and manual dispatch run the slow half: the end-to-end flight
tests, the full reference mission, every failsafe case, the reproducibility check
and the container build. Those re-derive numbers that are already committed and
cannot regress without one of the fast checks also failing, so they are not worth
twenty minutes per commit.

**The repository has to be public for any of it to run.** Actions on a private
repository under a free account fails in seconds with no runner assigned and no
useful message.
