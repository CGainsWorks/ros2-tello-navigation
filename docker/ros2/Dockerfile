# syntax=docker/dockerfile:1.7
#
# ROS 2 Humble with this workspace built by colcon: the image the launch tests
# run in, locally and in CI.
#
#   docker build -f docker/ros2/Dockerfile -t tello-ros2 .
#   docker run --rm tello-ros2
#
# The default command runs the launch tests, which fly the whole ROS graph
# against the Tello emulator (simulated) and assert on the outcome.
#
# The base image is pinned by digest as well as by tag, so a rebuild next month
# gets the same ROS packages as this one did.

FROM ros:humble-ros-base-jammy@sha256:860e53793ccf575e5369b5e9a19939de2bfcfb46a299954d06f54f2af54fed19

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-yaml \
        ros-humble-launch-testing \
        ros-humble-launch-testing-ros \
        ros-humble-launch-testing-ament-cmake \
        ros-humble-std-srvs \
        ros-humble-sensor-msgs \
        ros-humble-tf2-ros \
        ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Cyclone DDS, because the launch tests were developed and validated on it.
# Humble's default Fast DDS also carries the 691 KB frames once discovery is
# confined to loopback (below); switching is this one line.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# The whole graph and the emulator run in this one container, so discovery
# stays on loopback. Without it a test run on a machine with other ROS
# containers on the same Docker network receives their /rc and /nav_state, of
# different message definitions and every node logs deserialisation errors.
# That happened during development and at first looked like a DDS transport
# limit on large images; with discovery isolated the errors went away.
ENV ROS_LOCALHOST_ONLY=1

COPY docker/ros2/requirements.txt /tmp/ros2-requirements.txt
RUN python3 -m pip install -r /tmp/ros2-requirements.txt

WORKDIR /ws
COPY . /ws

# The ros images set ROS_DISTRO but nothing is on the path until setup.bash is
# sourced, so every step that needs ROS sources it explicitly.
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash \
    && colcon build --event-handlers console_direct- \
    && rm -rf build log

CMD ["bash", "docker/ros2/run_tests.sh"]
