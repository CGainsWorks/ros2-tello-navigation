#!/usr/bin/env bash
# Install everything needed to run the offline half of this repository.
#
# The offline half is all of it except the ROS 2 nodes: the simulator, the
# detector, the estimator, the controller, the failsafe, the test suite and the
# analysis. No ROS installation is needed for any of that, which is deliberate
# and is what makes `make demo` work on a laptop with nothing but Python.
#
# ROS 2 itself is not installed here. Installing a ROS distribution is a decision
# about somebody's machine and this script does not get to make it; see
# docs/ROS2.md for the colcon build once you have one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pick an interpreter that actually runs, rather than one that merely exists.
# On Windows, `command -v python3` finds the Microsoft Store stub, which is on the
# path, is not Python and prints an advert when invoked.
pick_python() {
    local candidate
    for candidate in "${PYTHON:-}" python3 python py; do
        [ -n "$candidate" ] || continue
        if "$candidate" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    echo "no working Python interpreter found. Set PYTHON=/path/to/python" >&2
    return 1
}
PYTHON="$(pick_python)" || exit 1

echo "using $("$PYTHON" --version 2>&1) at $(command -v "$PYTHON")"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo
echo "checking the pieces that are easy to get wrong"
PYTHONPATH=src/tello_core "$PYTHON" - <<'PYTHON'
import sys

import cv2
import numpy as np

print(f"  python  {sys.version.split()[0]}")
print(f"  numpy   {np.__version__}")
print(f"  opencv  {cv2.__version__}")

if not hasattr(cv2, "QRCodeEncoder_create"):
    raise SystemExit(
        "  this OpenCV build has no QR encoder, so the simulator cannot draw markers"
    )
if not hasattr(cv2, "QRCodeDetectorAruco"):
    print(
        "  warning: no QRCodeDetectorAruco (OpenCV older than 4.7). The classic "
        "detector works but reads markers only to about 2 m instead of 3.8 m, so "
        "the shipped marker layout will not cover the whole room."
    )
else:
    print("  detector aruco and classic both available")

from tello_core.localisation.markers import MarkerMap

marker_map = MarkerMap.from_yaml("maps/marker_map.yaml")
print(f"  map     {len(marker_map)} markers, survey method '{marker_map.survey.method}'")
PYTHON

echo
echo "ready. try:"
echo "  make demo        fly the reference mission"
echo "  make test        run the test suite"
echo "  make validate    check the marker map and the missions"
