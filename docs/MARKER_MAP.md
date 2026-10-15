# The marker map

The map is the survey. Everything this system claims about where it is rests on
`maps/marker_map.yaml` being right, so this document covers the format, the frame,
how the shipped layout was designed and how to survey a real room.

Current status: `survey: designed`. The layout describes a 4.0 x 3.0 x 2.4 m room,
it is internally consistent and **nothing in it has been measured**. See
`docs/HARDWARE.md`.

---

## The frame

Right-handed, z up, metres. The origin is the floor at one corner of the room, x
runs along the 4.0 m wall and y along the 3.0 m one. This is REP-103 ENU with a
local origin rather than a geodetic one, which is the correct choice for a
GPS-denied indoor frame.

`normal_yaw_deg` is the yaw of the direction the printed face looks, that is, the
bearing of the marker's outward normal. A marker on the wall at x = 0 looking
across the room has `normal_yaw_deg: 0`. That is the natural way to describe a
marker on a wall and it is why the format does not take Euler angles about the
world axes.

`tilt_deg` rotates the marker about its own x axis, for a marker that leans.
`roll_deg` rotates about its own z, for one hung out of square. Both default to
zero and both exist because a real survey never comes out perfectly and forcing
the surveyor to pretend it did pushes the same error into the localisation
instead.

### The marker's own frame

OpenCV's, the one `SOLVEPNP_IPPE_SQUARE` documents: x right as printed, y
**upwards** as printed, z out of the paper towards the viewer. Corners are
ordered top-left, top-right, bottom-right, bottom-left, which is also the order
`cv2.QRCodeDetector` returns them in, so the image-to-object correspondence is
fixed and no orientation search is needed. That is the main practical reason this
project uses QR symbols rather than a marker family whose four-fold symmetry has
to be resolved.

It is worth being explicit about why y is up. "x right, y down, z out of the page"
is the combination that looks natural if you are used to image coordinates and it
is not a right-handed frame: with x right and y down, z points *into* the page.
Using it anyway renders every marker mirrored and a mirrored marker still
decodes, still solves and gives a pose that is wrong in a way no residual
reveals, because the mirrored geometry is self-consistent. This project got it
wrong once and it cost an afternoon.

---

## The format

```yaml
frame_id: map

survey:
  method: designed          # how it was measured. "designed" means it was not.
  instrument: none
  date: unsurveyed
  stated_accuracy_m: 0.005  # one sigma, used as the survey term in every fix covariance
  notes: >
    Free text, carried into the results metadata.

default_size_m: 0.30           # symbol side, excluding the quiet zone
default_survey_sigma_m: 0.005

markers:
  - id: M01                 # whatever the survey sheet calls it
    payload: TLM01          # the string encoded in the symbol. ASCII, short.
    position: [0.010, 0.900, 1.400]   # centre of the symbol, metres
    normal_yaw_deg: 0.0
    # optional: size_m, tilt_deg, roll_deg, survey_sigma_m
```

The loader is strict and each refusal is a failure mode worth stopping for:

* **duplicate `id`**, which makes the log ambiguous;
* **duplicate `payload`**, because two markers that decode to the same string cannot be
  told apart in flight and the pose solve would place the aircraft somewhere
  between them;
* **non-ASCII `payload`**, which encodes as byte mode and decodes inconsistently
  across readers;
* **no size and no file default**, because marker size scales every position estimate
  by exactly the same factor, so there is no safe default to invent;
* **a position that is not three finite numbers.**

`scripts/validate_map.py` also reports warnings that are not errors, because a map
can be geometrically awkward and still be a correct description of a real room:
markers closer than 5 cm, a layout nearly confined to one plane, a marker below
the floor.

### Size means the symbol, not the sheet

`default_size_m` is the side of the QR symbol, measured across the outer edge of
the finder-pattern square, **not** including the white quiet zone. A 300 mm symbol
needs a four-module quiet zone of 57 mm on each side, so the printed sheet is
414 mm square.

This is the single easiest number to get wrong and the hardest to notice. A marker
printed 5 per cent small makes every position estimate 5 per cent too far away,
with a perfectly healthy reprojection residual and nothing anywhere to suggest a
problem. `scripts/print_markers.py` puts crop marks at the symbol's own corners so
the size can be checked against the paper with a ruler.

---

## How the shipped layout was designed

Eighteen markers, all 300 mm, in nine pairs. Three decisions, each from a
measurement in `results/`.

### Marker size: 300 mm

From `results/detection_range.csv`. At 300 mm the decode rate is 100 per cent out
to 1.6 m, above 80 per cent to 3.8 m and collapses at 4.0 m. At 250 mm it
collapses before 2 m, which does not cover a 4 x 3 m room.

300 mm is also the largest symbol whose sheet fits A2, which is the largest paper
an ordinary printer handles.

### Markers come in pairs, 45 cm apart

This is the decision that mattered most and it was not in the original design.

A single square marker leaves its own tilt almost unobservable when viewed head
on. The two solutions `SOLVEPNP_IPPE_SQUARE` returns reproject almost equally
well, so half a pixel of corner noise swings the answer by several degrees, which
at 1.5 m is about **15 cm of lateral position error**. Two markers in the same
view turn a 300 mm object into a 750 mm one, the tilt becomes well determined and
the measured fix error drops to **under 3 cm**.

A factor of five, for the cost of printing twice as many sheets.

They do not need to be on different walls, which is fortunate: two markers on
*perpendicular* walls near a corner subtend more than 100 degrees from anywhere
close to that corner and the camera has 70 degrees of horizontal field. The
original ten-marker layout had exactly that problem and the four corners of the
reference mission could see only one marker from any heading.

Coplanar pairs are also much easier to survey, because both measurements are
against the same wall.

### Vertical spread

`M03` is deliberately low at 0.60 m and `M06` deliberately high at 1.85 m. With
every marker at chest height the vertical geometry is weak and z is the
worst-observed axis whenever a single marker is in view.

### Checking the coverage

```bash
python scripts/validate_map.py
```

flies a grid of virtual poses through the geofence and reports, for each, the most
markers any heading could see. The shipped layout gives:

```
  sees a marker  100.0 per cent
  sees two       100.0 per cent
```

The two-marker figure is the one that matters, for the reason above and it is why
the script reports it separately. CI fails the build if single-marker coverage
drops below 95 per cent, because a map edit that opens a hole in the coverage is
exactly the kind of change that looks harmless in review.

---

## Surveying a real room

1. **Choose the frame.** Pick a floor corner as the origin and a wall as the x
   axis. Mark both on the floor with tape, because every later measurement refers
   to them and a frame you cannot find again is not a frame.
2. **Print the markers.** `python scripts/print_markers.py --out build/markers`,
   at 100 per cent scale with fit-to-page off. Measure between the crop marks
   before putting anything up.
3. **Put them up flat.** A crease through the timing pattern costs detection
   range; a reflection off a ceiling light costs more. Pairs 45 cm apart, centre to
   centre, on the same wall.
4. **Measure the centre of each symbol**, not of the paper. A laser distance meter
   and a plumb line get a wall-mounted marker to about 5 mm, which is what
   `stated_accuracy_m: 0.005` in the shipped file claims.
5. **Write it down** and set `survey.method`, `survey.instrument`,
   `survey.date` and `survey.stated_accuracy_m` to what you actually did. Those
   fields are carried into every results file, so an accuracy figure can never be
   read without knowing the survey behind it.
6. **Validate.** `python scripts/validate_map.py` and
   `python scripts/validate_missions.py`.

Expect an afternoon for eighteen markers and expect it to be the work that turns
every number in `docs/RESULTS.md` from a statement about a file into a statement
about a room.
