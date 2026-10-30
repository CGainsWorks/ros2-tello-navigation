#!/usr/bin/env bash
# Run the ROS 2 launch tests inside the image built from docker/ros2/Dockerfile.
#
# Results go to $TELLO_RESULTS (default /ws/results/ros2): the launch tests write
# a summary per scenario, the emulator's simulated ground truth and its event
# log there, and colcon writes the JUnit XML under it.
set -euo pipefail

cd /ws
# The ROS setup scripts read unset variables, so nounset is off while sourcing.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source install/setup.bash
set -u

export TELLO_RESULTS="${TELLO_RESULTS:-/ws/results/ros2}"
mkdir -p "$TELLO_RESULTS"

status=0
colcon test \
    --packages-select tello_bringup \
    --test-result-base "$TELLO_RESULTS/test_results" \
    --event-handlers console_direct+ \
    --pytest-args -v || status=$?
colcon test-result --test-result-base "$TELLO_RESULTS/test_results" --verbose || status=$?
exit "$status"
