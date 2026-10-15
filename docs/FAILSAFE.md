# Failsafe

The brief for this project says failsafe logic first, autonomy second and do not
fly anything that cannot land itself. This is that instruction written down.

Everything else in the repository decides where the aircraft goes. This decides
whether it is allowed to go anywhere and what happens when it is not.

The machine is `src/tello_core/tello_core/control/failsafe.py`, it is about three
hundred lines and it has the largest test file in the repository
(`tests/test_failsafe.py`) because every branch of it is a thing that has to work
on the one occasion it is needed.

---

## States

| State | Meaning |
| --- | --- |
| `IDLE` | On the ground, not commanded. |
| `TAKEOFF` | Climbing to the mission's height. |
| `NAVIGATE` | Flying the mission. |
| `HOLD` | Stationary, turning slowly, waiting for something to come back. |
| `RETURN` | Flying to the launch point. |
| `LAND` | Descending under control. |
| `EMERGENCY_LAND` | Descending without a usable estimate. |
| `LANDED` | On the ground, terminal. |

Two structural properties and they matter more than the list:

**Severity is monotone.** Once the machine has entered `RETURN` it will not go
back to `NAVIGATE` and once it has entered `LAND` it will not go back to
anything. A machine that can be talked out of a landing decision can be talked
out of it by a sensor that flickers and the entire value of a failsafe is that it
cannot be.

**`HOLD` is the only recoverable state**, because lost localisation is the only
recoverable condition. A battery does not refill and a link that has been down for
two seconds is not evidence of anything improving.

---

## Priorities

Several conditions can be true at once. They are evaluated in a fixed order and
the first match wins. The order is by **how little time is left**, not by how bad
the condition sounds:

1. `ESTIMATOR_DIVERGED`, `CRITICAL_BATTERY`. Seconds left. Land immediately, wherever
   the aircraft is.
2. *Return complete*. If the aircraft is already going home and has arrived,
   land. Checked above the conditions that cause a return, because those
   conditions are all still true when it gets there: the battery is still low.
   Testing it further down leaves the aircraft hovering over the launch point
   until something else lands it, which is what the first version did.
3. `LINK_LOSS`. The aircraft is about to make its own decision. See below.
4. `GEOFENCE_BREACH`. It is leaving the room.
5. `OVERHEAT`. It will land itself soon.
6. `LOW_BATTERY`, `MISSION_TIMEOUT`. Tens of seconds left. Return, then land.
7. `OPERATOR_ABORT`. Whatever was asked for. Below the automatic conditions,
   because an operator asking for a landing while the battery is critical should
   still get the critical-battery response, which is the more urgent one.
8. `LOCALISATION_LOST`. Hold, look for a marker, land if it does not come back.

---

## Link loss, which is the interesting one

A Tello lands itself fifteen seconds after the last command it hears. That is not
a failsafe this software can override and it should not try.

So when the link drops, this machine does two things and both are slightly
counter-intuitive. It commands `land` and it **stops sending rc commands
entirely**.

Continuing to send them is the instinctive response and it is actively harmful. On
a link that is degraded rather than dead, an rc command that gets through resets
the aircraft's own dead-man timer. A controller faithfully transmitting into a
half-broken link can therefore keep an aircraft it has no control over airborne
indefinitely. Falling silent hands the aircraft to its own failsafe, which is the
only thing still able to act.

`FailsafeDecision.send_rc` is the field that carries this, `tests/test_failsafe.py`
pins it and `scripts/check_flight.py --expect-silent` checks it on every CI run,
because it is the kind of behaviour that an otherwise sensible refactor would
remove.

The cost is that the aircraft hovers, drifting, for up to fifteen seconds before
its own timer fires. In a 4 x 3 m room at the measured 5 cm/s of drift that is
well inside the walls. In a larger or windier space it would not be and that is
in `docs/LIMITATIONS.md` rather than hidden here.

---

## Thresholds and where each comes from

| Setting | Value | Why |
| --- | --- | --- |
| `link_timeout` | 1.0 s | Ten expected packets at 10 Hz. Long enough that ordinary loss on a 2.4 GHz link in a building does not trip it; short enough to leave most of the aircraft's own fifteen-second window for the response. |
| `frame_timeout` | 2.5 s | Longer than the telemetry timeout, because video is bulkier and drops out more readily and because losing video costs localisation rather than control. |
| `localisation_timeout` | 2.0 s | How long the filter will vouch for its own absolute position without a fix. |
| `hold_timeout` | 6.0 s | The aircraft dead-reckons through this on a 10 Hz accelerometer. The measured drift says the estimate is worth little beyond about eight seconds, so six is a margin rather than a limit. |
| `hold_search_rate` | 35 deg/s | Slow enough that motion blur does not defeat the detector. Measured: see `results/detection_yaw_rate.csv`, where the decode rate falls off a cliff at about 8 px of blur, which is 40 deg/s. |
| `geofence_grace` | 0.15 m | Stops a noisy estimate sitting on the boundary from tripping it. |
| `geofence_hard` | 0.50 m | Beyond this, the estimate saying "outside" is probably also wrong about which way is back, so returning is abandoned and the aircraft lands where it is. |
| `low_battery` | 25 per cent | Return. |
| `critical_battery` | 15 per cent | Land here, now. |
| `battery reserve` | 25 s | Below this much estimated flight time, returning is not attempted: do not start a journey that ends in a fall. |
| `overheat_temperature` | 88 C | The aircraft throttles and eventually lands itself above about 90. |
| `landed_height` / `landed_dwell` | 0.12 m / 1.0 s | Both halves needed. Without the dwell a bounce reads as a landing. |

Two of these are perception constraints masquerading as control settings and
both were set from measurement after the first autonomous flight failed:
`hold_search_rate` and the controller's own yaw limit in `control.yaml`. See
`docs/RESULTS.md`.

---

## The geofence floor

The fence's lower z bound is a minimum height above the floor and an aircraft
taking off or landing is below it by definition: it starts and ends on the ground.

The first version of this machine checked the bound in every state. It landed an
aircraft that had not taken off, for leaving a volume it had never been in, half a
second into the first flight. The floor is now exempt in `TAKEOFF`, `LAND` and
`EMERGENCY_LAND`; the horizontal bounds and the ceiling apply throughout, because
there is no phase of flight in which drifting into a wall is expected.

The fence is also only checked when the estimate is localised. An unlocalised
estimate saying the aircraft is outside is not evidence that it is.

---

## Refusing to take off

Two separate refusals, for two different reasons.

**No fix.** `MissionRunner` will not leave the ground until the filter has an
absolute fix and gives up after `preflight_timeout` (12 s) if it cannot get one.

An aircraft that takes off not knowing where it is has to find out while moving,
using a camera that is now being shaken. Refusing to fly is the cheap failure and
the log says so: the metadata carries `aborted: no marker fix before take-off`
rather than an empty flight that looks like a crash.

This is also why the simulated aircraft has a 3 cm resting height. The camera is
mounted below the body origin, so with the origin at z = 0 the camera would be
below the floor plane and the pre-flight check could never pass.

**A condition that would land it.** Anything in the priority list that would cut a
flight short is instead a reason not to start one and on the ground the aircraft
is already where a landing would put it, so the refusal is terminal rather than a
descent. A battery below the return threshold, a diverged estimator, a dead link,
an overheating airframe and an operator abort all refuse the take-off and the log
records which.

Getting this wrong was instructive. A battery below the return threshold used to
send the machine straight to `RETURN` while the aircraft sat on the floor with its
motors off; `RETURN` does not command a take-off, so nothing moved, nothing
landed and the flight ran until the wall-clock limit expired ten minutes later.
`tests/test_failsafe.py` now parametrises the refusal over every condition that
should cause one and asserts that lost localisation is *not* one of them, because
on the ground that is a take-off that has not happened rather than a refusal.

---

## Evidence

`make failsafe` flies every case and writes a full trajectory and event log for
each into `results/failsafe/`. `scripts/summarise_failsafe.py` turns those into
`results/failsafe_log.csv` and asserts the expected outcome for each case, so this
document cannot quietly stop being true.

| Case | What is wrong | Expected |
| --- | --- | --- |
| `nominal` | nothing | mission completes, nothing ends the flight |
| `link_loss` | link drops at 18 s, never returns | `LINK_LOSS`, lands, controller silent |
| `video_loss` | video dies at 12 s, telemetry fine | `LOCALISATION_LOST`, holds, lands |
| `low_battery` | crosses the return threshold in flight | `LOW_BATTERY`, returns, lands |
| `critical_battery` | crosses the critical threshold in flight | `CRITICAL_BATTERY`, lands where it is |
| `refused_battery` | already below the return threshold on the ground | refuses to take off |
| `lossy_link` | 20 per cent of telemetry and 25 per cent of video lost, plus malformed packets | mission completes, nothing ends the flight |

"Nothing ends the flight" rather than "nothing trips", because both clean cases
*do* trip. Each loses sight of the markers for about two seconds while turning,
holds, reacquires and carries on. That is the machine working rather than a
fault. The check that matters is `final_cause`: the flight must not be *ended* by
a failsafe. `scripts/summarise_failsafe.py` makes the distinction explicitly with
three kinds of expectation, `requires`, `allows` and `ends_clean`, because the
alternative is either calling correct behaviour a regression or loosening
`localisation_timeout` until the failsafe protects nothing.

`scripts/check_flight.py`, which CI runs on every push, draws the same line and
adds a `--max-triggers` bound. A flight that recovers once is healthy. A flight
that recovers eleven times has a perception problem, even though every hold ends.

Two of those cases need a word about how they are produced.

The battery cases **multiply the drain rate** so that a threshold is crossed
inside a thirty-second mission rather than after eight minutes of flight. The
multiplier is recorded in each results file, so a battery figure from one of these
flights can never be read as if it came from a real discharge curve. What is being
tested is the machine's response to crossing a threshold and that does not depend
on how long the crossing took.

The video case loses video **mid-flight** rather than from the start. Losing it
from the start is also correct behaviour and tests nothing: the aircraft simply
refuses to take off, because it never gets a fix. The interesting case is losing
sight of the markers while airborne, with the telemetry still arriving, so the
aircraft is fully controllable and completely unable to localise.

The last case is the one that stops the thresholds being set too tight. A failsafe
that fires on ordinary wireless attrition is a failsafe that gets turned off.

### A nominal flight is allowed to hold

`scripts/summarise_failsafe.py` does not require the nominal flight to be free of
failsafe activity and that is deliberate rather than a loosened standard.

The reference flight loses sight of its markers for a couple of seconds four
times, once per turn, enters `HOLD`, reacquires within half a second and carries
on to complete the mission. That is the machine working. Forbidding it would leave
two options, both worse: call correct behaviour a regression, or lengthen the
localisation timeout until the failsafe stopped protecting anything.

So the check asks the three questions separately. Did the cause that was supposed
to appear appear? Did anything appear that should not have? And did the flight end
*because* of a failsafe? A mission that completed and then landed normally is a
success however many times it paused on the way and `final_cause` in every
results file is the field that says which.

The measured numbers for each case are in `docs/RESULTS.md` and the figure is
`analysis/figures/failsafe.png`.

---

## What is not handled

* **Obstacles.** There is no obstacle sensing of any kind. The Tello has a
  downward time-of-flight sensor and nothing forward. The geofence keeps the
  aircraft inside a box that was chosen to be empty; it does not keep it away from
  anything inside that box.
* **A person walking through the test area.** They will occlude markers, which
  reads as lost localisation and is handled and they are also an obstacle, which
  is not.
* **Propeller damage or a motor failure.** Nothing is observable about either
  through the SDK until the aircraft is already falling.
* **A wrong marker map.** If a marker is 20 cm from where the file says, the
  estimator will be confidently 20 cm wrong and every residual will look healthy.
  `scripts/validate_map.py` catches an internally inconsistent map; nothing
  catches a consistently wrong one except measuring the room.
