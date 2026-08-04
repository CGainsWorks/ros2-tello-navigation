# ros2-tello-navigation

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

Indoor autonomous flight without GPS. QR markers at known world positions provide absolute localisation, fused with IMU and visual odometry, driving a PID waypoint controller with a failsafe land state machine.

---

## What this covers

CV entries this repository is the evidence for:

- Autonomous TELLO Drone Navigation

---

## Requirements

### Hardware

- DJI Tello
- Printed QR markers at surveyed positions
- Host laptop
- Wi-Fi access point
- Indoor test area with measured marker layout

### Software

- ROS1 Noetic and tello_driver (a ROS2 port is a reasonable extension)
- OpenCV QR detection
- Visual odometry front end
- IMU fusion filter
- PID waypoint controller
- Failsafe land state machine

### Report and documentation deliverables

- README with system diagram
- Marker map layout with surveyed positions
- PID tuning notes
- Position error against measured marker positions
- Failsafe test log
- Flight video

---

## Inputs

- Tello camera stream
- Onboard IMU telemetry
- QR marker map with known world positions
- Waypoint list
- Battery and link status

## Outputs

- Estimated 6-DoF pose
- Velocity and yaw-rate commands
- Waypoint arrival events
- Trajectory logs
- Failsafe triggers with cause codes

---

## Intended structure

```
src/
  tello_bringup/          launch and config
  tello_localisation/     QR detection, VO, fusion
  tello_control/          PID waypoint controller, failsafe FSM
  tello_msgs/
maps/                     marker_map.yaml with surveyed positions
analysis/                 trajectory error notebooks
docs/
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Connect to the Tello and stream video and telemetry
- [ ] QR detection giving a pose from a single marker
- [ ] Marker map file format and multi-marker localisation
- [ ] IMU and visual-odometry fusion
- [ ] PID waypoint controller in simulation or tethered
- [ ] Failsafe state machine: link loss, low battery, lost localisation
- [ ] Full autonomous waypoint mission
- [ ] Position error evaluation against surveyed markers

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: ros2-tello-navigation
Indoor autonomous flight without GPS. QR markers at known world positions provide absolute localisation, fused with IMU and visual odometry, driving a PID waypoint controller with a failsafe land state machine.

TARGET STACK
- ROS1 Noetic and tello_driver (a ROS2 port is a reasonable extension)
- OpenCV QR detection
- Visual odometry front end
- IMU fusion filter
- PID waypoint controller
- Failsafe land state machine

HARDWARE CONTEXT
- DJI Tello
- Printed QR markers at surveyed positions
- Host laptop
- Wi-Fi access point
- Indoor test area with measured marker layout

THE SYSTEM MUST TAKE THESE INPUTS
- Tello camera stream
- Onboard IMU telemetry
- QR marker map with known world positions
- Waypoint list
- Battery and link status

AND PRODUCE THESE OUTPUTS
- Estimated 6-DoF pose
- Velocity and yaw-rate commands
- Waypoint arrival events
- Trajectory logs
- Failsafe triggers with cause codes

INTENDED REPOSITORY LAYOUT
src/
  tello_bringup/          launch and config
  tello_localisation/     QR detection, VO, fusion
  tello_control/          PID waypoint controller, failsafe FSM
  tello_msgs/
maps/                     marker_map.yaml with surveyed positions
analysis/                 trajectory error notebooks
docs/

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
