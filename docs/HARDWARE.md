# Hardware: what has not been done

**There is no Tello.** This repository was built and tested entirely against a
simulator. No line of it has exchanged a packet with an aircraft, no marker has
been printed and no room has been surveyed.

That is the single most important thing to know before reading any number in this
repository, so it is stated here, in the README, in the output of every script
that reports an accuracy and in the metadata of every results file. The rest of
this document is the specific list: what is asserted without evidence, what would
have to be measured and in what order.

---

## What the simulator does and does not substitute for

The simulator is not a convenience. It is a deliberate substitute and it is
honest about which half of the problem it covers.

**It does substitute for the navigation problem.** A real QR symbol is generated
by OpenCV's encoder, printed into a textured room at its surveyed position,
rendered through a pinhole camera with perspective, motion blur, vignetting,
signal-dependent sensor noise and JPEG compression and then found by the same
`cv2.QRCodeDetectorAruco` that would run on hardware. When detection fails in
simulation it fails for the reasons it would fail in a room: too far, too oblique,
too blurred. The telemetry is serialised into genuine Tello state strings and
parsed back by the same decoder the real driver uses, so the unit conversions and
the axis flips are exercised by every simulated flight. The link drops packets,
corrupts packets, goes down entirely for seconds at a time and the aircraft lands
itself after fifteen seconds without a command exactly as the SDK says it does.

**It does not substitute for the aircraft.** The flight dynamics are a
first-order velocity lag with a disturbance, not a rigid-body model and its time
constants are estimates. The accelerometer noise and bias are plausible figures,
not measured ones. Print quality, lens distortion, rolling shutter, H.264
artefacts, Wi-Fi behaviour in a building with other networks in it, propeller
wash off a nearby wall and a battery whose voltage sags under load are all
absent.

So: a result from this repository is a statement about the algorithms and it is
evidence that they are correctly implemented and internally consistent. It is not
evidence about a DJI Tello.

---

## Every unverified assertion and what it would cost to check

Ordered by how much a wrong value would matter.

### 1. Camera intrinsics

`CameraModel.tello_nominal()` derives the focal length from DJI's published 82.6
degree diagonal field of view at 960x720:

```
f = (sqrt(960^2 + 720^2) / 2) / tan(82.6 deg / 2) = 683.0 px
```

Distortion is set to zero because there is no published figure and inventing one
would be worse than admitting there is none. The Tello's lens does have visible
barrel distortion at the frame edges.

**Why it matters most.** The focal length scales every range estimate linearly. A
five per cent error in `f` is a five per cent error in every distance, which at
2 m is 10 cm, which is larger than the entire error budget. Uncorrected
distortion adds a position error that grows towards the frame edges, which is
exactly where a marker is when the aircraft is about to lose sight of it.

**How to check.** `scripts/calibrate_camera.py` takes a recording of a printed
chessboard and writes the same YAML format. Twenty frames at varied angles,
fifteen minutes.

### 2. Camera mount offset

`aircraft.yaml` puts the camera 32 mm forward of and 12 mm below the body origin,
read off DJI's published 98 x 92.5 x 41 mm dimensions.

**Why it matters.** A one centimetre error here is a one centimetre bias in every
position estimate, in a direction that rotates with the aircraft. It is not
removable by any amount of filtering because it is not noise.

**How to check.** Hold the aircraft against a marker at a measured distance,
record the fix and solve for the offset that makes the fix agree with the tape
measure. Repeat at two headings 90 degrees apart to separate the forward and
lateral components. Half an hour.

### 3. The `rc` channel scaling

`rc_scaling` says full stick is 1.0 m/s horizontally, 0.8 m/s vertically and 100
deg/s in yaw, at the default speed setting. The maxima are published. The
linearity between zero and full stick is assumed and is certainly not exact.

**Why it matters.** An error here is a proportional-gain error, so the controller
is detuned rather than unstable. That makes it easy to hide: anyone would retune
the gains and never notice the scaling was wrong.

**How to check.** Hover in front of a marker pair, command a fixed stick input on
one channel for three seconds and read the velocity the estimator reports. The
scaling is that velocity divided by the stick fraction. Repeat at two or three
stick values to check the linearity rather than assuming it. Ten minutes per
channel and it wants the geofence tightened first.

### 4. Accelerometer units

The state packet's `agx`, `agy`, `agz` are documented inconsistently. This
implementation treats them as centimetres per second squared, following
`djitellopy`. Some sources give thousandths of g, which differs by a factor of
0.98.

**Why it tolerates being wrong.** Two per cent of a signal dominated by the
constant 9.81 looks very like a bias and the filter estimates an accelerometer
bias. It is still a real gap.

**How to check.** Sit the aircraft on a table and read the packet. A level,
stationary aircraft reports one g on the vertical axis and which number appears
settles the units immediately. Two minutes and the first thing to do.

### 5. The axis conventions in the state packet

`tello_core.link.state` assumes the aircraft reports `vgy` to the right and `vgz`
down, aerospace style and flips both to reach the REP-103 frame the rest of the
stack uses.

**Why it matters.** A wrong sign here produces a system that flies correctly in
one axis and diverges in another, which is far harder to diagnose than one that
fails outright.

**How to check.** Push the aircraft gently by hand in each of three directions
and watch the reported velocity. Five minutes and the second thing to do.

### 6. The UDP driver itself

`tello_core/link/udp.py` speaks Tello SDK 2.0 from the published document. It has
never sent a packet to an aircraft. It is exercised against a loopback socket in
the test suite, which proves it parses and serialises correctly and proves nothing
about whether a Tello likes it.

It also ships without a video decoder, deliberately. The Tello streams raw H.264
in UDP datagrams with no container, so a decoder is needed; the options are PyAV
or an `ffmpeg` subprocess and this repository does not assert a dependency it
cannot test. `UdpLink` takes a decoder as an argument.

### 7. Flight dynamics

Every number in `DynamicsSettings` is an estimate. The velocity time constants of
0.35 s horizontal and 0.45 s vertical are chosen so that the closed-loop
behaviour resembles what is reported for the Tello, which is a much weaker
statement than "measured".

**Consequence.** The controller gains in `control.yaml` are tuned against this
plant. `docs/PID_TUNING.md` includes a sensitivity sweep showing how much of the
behaviour survives a plausible range of time constants, which is the honest way
to say how far they can be trusted.

### 8. The marker survey

`maps/marker_map.yaml` says `survey: designed`. The layout describes a
4.0 x 3.0 x 2.4 m room, it is internally consistent and nothing in it has been
measured. Every accuracy figure in `docs/RESULTS.md` is therefore a statement
about agreement with a file.

**How to check.** A laser distance meter and a plumb line get a wall-mounted
marker to about 5 mm, which is the `stated_accuracy_m` the file already claims.
An afternoon for eighteen markers and it is the work that turns every number
here into a measurement.

### 9. Print quality

Simulated markers are geometrically perfect. A printed one has bled edges, a
crease, a reflection off the paper under a ceiling light and tape over a corner.
All of those cost detection range and none of them are modelled.

---

## The order to do it in

1. **Read the state packet on a table.** Settles the accelerometer units and the
   axis signs, which are the two assumptions that would silently corrupt the
   estimator. Ten minutes.
2. **Calibrate the camera.** The largest single error source. Fifteen minutes.
3. **Get video decoding.** Nothing else can be checked until frames arrive.
4. **Survey the room.** An afternoon and the thing that makes the numbers mean
   something.
5. **Measure the mount offset.** Half an hour, against the survey.
6. **Calibrate the rc scaling.** Ten minutes per channel, tethered or in a
   corner.
7. **Re-measure the detection curves on real frames.** `make detection`
   regenerates them against the simulator; the hardware equivalent is a marker on
   a wall and a tape measure. The simulated curve says detection collapses beyond
   3.8 m and beyond 50 degrees of incidence; the real numbers will be worse and
   the marker layout depends on them.
8. **Then and only then, fly a mission.** Tethered first, with the geofence
   tightened and the hold timeout shortened.

Until step 8, the correct description of this repository is "a navigation stack
that works in simulation" and that is how it is described.

---

## What would change in the code

Very little, which is the point of the `TelloLink` interface. Flying a real
aircraft is a different `link_type` parameter on the driver node, or a different
object passed to `MissionRunner`. The failsafe machine, the controller, the
estimator and the detector are unchanged and they are the parts that have been
tested.

What would change is the configuration: `aircraft.yaml` would get measured
numbers, `camera_tello.yaml` would get a real calibration, `marker_map.yaml`
would get a real survey and its `survey.method` would stop saying `designed`.
