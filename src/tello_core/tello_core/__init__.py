"""Pure-Python core: driver links, localisation, fusion, control and evaluation.

Nothing in this package imports ROS 2. That is deliberate and it is what lets the
test suite, the analysis scripts and CI run the real algorithms in seconds on any
machine with Python and OpenCV. The ROS 2 wrappers live in ``src/tello_localisation``
and ``src/tello_control`` and are thin by design.
"""

__version__ = "0.1.0"
