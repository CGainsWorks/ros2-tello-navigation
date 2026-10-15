# Controller and tuning

## The loop

One loop, not two. The obvious architecture for a position controller is a
cascade: position to velocity, velocity to attitude. The Tello already closes its
own velocity loop inside the airframe and the SDK's `rc a b c d` command *is* a
velocity request, so a second velocity controller on top of the aircraft's own
means two integrators fighting over one actuator across a wireless link with 30 to
60 ms of delay. That is a reliable way to produce a slow oscillation nobody can
tune out.

So the position loop produces a world-frame velocity directly, rotates it into the
body frame and scales it to stick counts. The damping comes from the derivative
term acting on the estimated position, which for a fixed setpoint is the negative
of the estimated velocity: exactly the velocity feedback a cascade would provide,
without the second integrator.

```
waypoint ──► PID per world axis ──► clamp ──► rotate to body ──► scale to counts
                   ▲                                                     │
                   └──────────── estimated position ◄────── EKF          ▼
                                                                    rc a b c d
```

---

## The three details that matter on a real vehicle

### Derivative on measurement, not on error

A step change in the setpoint makes the error jump and a derivative term acting
on that jump produces an impulse. On this aircraft that is a full-scale stick
deflection for one control cycle every time a waypoint is reached. The derivative
term exists to damp the *vehicle's* motion and the vehicle's motion is the
measurement, so differentiating the measurement and negating gives the damping
without the kick. `tests/test_pid.py::test_no_derivative_kick_on_a_setpoint_step`
pins it.

### Anti-windup by back-calculation

Clamping the output without telling the integrator is the classic mistake. The
integrator keeps accumulating while the actuator is saturated and when the error
finally reverses the controller spends seconds unwinding before it responds at
all.

This matters more on a Tello than it might, because full stick is only 1 m/s and a
controller asked to cross a 4 m room saturates immediately. Back-calculation feeds
the difference between the requested and the delivered output back into the
integrator, so it stops growing exactly when the actuator stops responding.
`tests/test_pid.py::test_back_calculation_stops_windup_while_saturated` measures
the difference in cycles to reverse.

### Explicit `dt`

The controller never reads a clock. Every call takes the interval since the last
one, which makes it testable, makes a simulated run reproducible and makes it
behave correctly when the loop rate wobbles, which on a wireless link it always
does.

---

## Yaw, which is not a control decision

The yaw rate limit in `control.yaml` is 30 deg/s and it is a perception
constraint.

The camera's 1/60 s exposure turns a yaw rate into image blur of
`fx * omega * exposure` pixels. `results/detection_yaw_rate.csv` is the measured
curve at 1.5 m range, twenty samples per point:

| yaw rate | blur | decode rate |
| --- | --- | --- |
| 0 to 35 deg/s | 0 to 7 px | 85 to 100 per cent |
| 40 deg/s | 7.9 px | 35 per cent |
| 50 deg/s and above | 9.9 px and above | 0 per cent |

The cliff is at about eight pixels of blur and it is very steep. 30 deg/s sits at
six pixels with a measured 85 per cent, which leaves margin for a marker further
away or more oblique than the one measured. It costs about three seconds on a 90
degree turn, which is the price of being able to see during it.

This was not the original value. The first autonomous flight in this repository was
allowed 90 deg/s, turned to face its first waypoint's heading, blinded itself and
landed nine seconds after take-off with the failsafe correctly reporting lost
localisation. The estimator was never the problem.

`tests/test_pid.py::test_shipped_yaw_limit_keeps_motion_blur_under_control`
asserts the blur stays under eight pixels, so raising the limit for a snappier
turn cannot quietly reintroduce the failure.

---

## The gains

From `src/tello_bringup/config/control.yaml`:

| | kp | ki | kd | output limit | integral limit |
| --- | --- | --- | --- | --- | --- |
| horizontal | 1.10 | 0.22 | 0.42 | 0.75 m/s | 0.30 m/s |
| vertical | 1.30 | 0.30 | 0.30 | 0.55 m/s | 0.30 m/s |
| yaw | 1.60 | 0 | 0.18 | 30 deg/s | none |

Why each is what it is:

* **`kp` of about 1** means an error of one metre asks for one metre per second,
  which saturates the output limit at about 0.7 m of error. That is the intended
  behaviour: cross the room at the speed limit, then let the proportional term
  take over for the last 70 cm.
* **The output limit is 0.75 m/s against the aircraft's 1.0**, so there is stick
  left over for the disturbance rejection and the yaw coupling. A position loop
  allowed to ask for full stick has nothing in reserve when the air moves.
* **The integral term exists for the air.** The room has currents and the airframe
  has trim errors and without an integrator the aircraft settles a few
  centimetres downwind of every waypoint and stays there. It is limited to 0.30
  m/s of authority, which is enough to cancel a steady disturbance and not enough
  to fly anywhere on its own.
* **`kd` is larger horizontally than vertically** because the horizontal plant is
  faster and less damped: the aircraft accelerates sideways by tilting, which is a
  second-order response, where it accelerates vertically by changing thrust.
* **Vertical gains are stiffer** because the sensor behind them is better. The
  downward time-of-flight sensor is good to about 3 cm; horizontal position comes
  from markers alone.
* **Yaw has no integral term.** Yaw has no steady disturbance to reject and an
  integrator on an axis whose measurement occasionally jumps by a marker fix is a
  slow oscillation waiting to happen.
* **`derivative_tau` of 0.10 s** filters the derivative. The signal being
  differentiated is a Kalman filter output at 10 Hz and differentiating it raw
  puts the estimator's noise straight onto the sticks.

---

## How much of this survives being wrong about the aircraft

The gains were tuned against the simulated plant in
`src/tello_core/sim/dynamics.py`, whose velocity time constants are estimates, not
measurements. That is a real weakness and the honest way to report it is a number
rather than a disclaimer.

`scripts/tuning_sweep.py` flies the same mission with the same gains against a
range of plants, varying the horizontal time constant, the vertical one and the
disturbance strength one at a time. `results/tuning_sweep.csv` holds the output and
the script prints a verdict.

What to look for: if every plant completes the mission and the error grows
smoothly, the gains are not perched on one particular plant. If completion fails
somewhere inside the range, they are and they would not survive a real aircraft
without retuning.

Run it with:

```bash
python scripts/tuning_sweep.py
```

Either way, the gains in this repository are a **starting point for a real
Tello, not a finished tuning.** The plant they were tuned against has never been
measured, so a sweep around it says how sensitive the tuning is, not whether it is
right.

---

## Retuning on hardware

In this order, tethered or with the geofence tightened:

1. **Fix the scaling first.** `rc_scaling` in `aircraft.yaml` says what full stick
   means in m/s and it is unverified. An error there is a proportional-gain error,
   so retuning the gains will hide it rather than fix it and every later step will
   be built on it. See `docs/HARDWARE.md`.
2. **Vertical axis, hovering.** Set the horizontal gains to zero, command a height
   change of 0.3 m and watch the step response. Raise `kp` until it overshoots,
   then add `kd` until it does not.
3. **Horizontal, one axis at a time**, with 0.5 m steps. The lag is longer than
   the vertical axis and the overshoot is slower to appear; give it four seconds
   before deciding.
4. **Add the integral terms last** and only if the aircraft settles off-target in
   a consistent direction. An integrator added to fix an oscillation makes it
   worse.
5. **Leave the yaw limit alone** unless you have re-measured the detection curve
   on real frames. It is a perception number and the aircraft's own motor response
   has nothing to do with it.

---

## The filter's noise model

Not controller gains but the same kind of question, so it is recorded here too.
`FusionSettings` in `fusion.py` carries the reasoning for each in its docstrings;
the two that are not sensor properties are:

* **`min_position_sigma` (3.5 cm)**, a floor on the filter's own reported
  uncertainty. A Kalman filter shrinks its covariance as the square root of the
  number of independent measurements and at 10 Hz it shrinks a long way. Those
  fixes are not independent: consecutive frames see the same markers at almost the
  same angle and share whatever error the survey, the calibration and the corner
  detection contribute and no amount of averaging removes a systematic error.
  Without the floor the filter reported 1.5 cm of uncertainty against a measured
  6 cm of error and the failsafe believed it.
* **`max_consecutive_rejections` (5)**, which is what stops the gate deadlocking.
  See `docs/RESULTS.md`; it is the most important line in that file.
