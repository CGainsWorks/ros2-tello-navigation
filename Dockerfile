# syntax=docker/dockerfile:1.7
#
# The syntax directive is load-bearing: the build-time sanity check below uses a
# here-document in a RUN instruction and that is a BuildKit feature. Without the
# directive an older builder treats each line of the heredoc as a separate
# instruction and the build fails with a parse error that says nothing useful.
#
# The zero-install path: everything in this repository except the ROS 2 nodes.
#
# That is not a reduced version. The simulator, the QR detection, the pose
# solver, the fusion filter, the controller, the failsafe machine, the test suite
# and the analysis all run here, because none of them imports ROS. The ROS 2
# wrappers are thin by design and building a ROS image to exercise them would
# multiply this image's size by about twenty for the sake of code that subscribes
# and publishes. docs/ROS2.md covers the colcon build for anyone who has a ROS
# distribution already.
#
# Slim rather than full, with the two OS packages OpenCV's headless wheel needs
# installed explicitly. `opencv-python-headless` still links libGL and libglib
# even though nothing here opens a window and the import fails at runtime
# without them in a way that reads as a Python problem.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="ros2-tello-navigation"
LABEL org.opencontainers.image.description="GPS-denied indoor drone navigation from QR markers"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src/tello_core \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first, so a code change does not re-resolve the dependencies. The
# OpenCV wheel is 40 MB and this saves it on every rebuild.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Fail the build rather than the first run if the image cannot do the one thing
# it exists for. A container that starts and then cannot encode a QR symbol is a
# much more confusing failure than a build that stops here and says so.
RUN python - <<'PYTHON'
import cv2

from tello_core.localisation.markers import MarkerMap
from tello_core.localisation.qr import QrDetector

if not hasattr(cv2, "QRCodeEncoder_create"):
    raise SystemExit("this OpenCV build cannot encode QR symbols, so the simulator cannot draw")
detector = QrDetector()
marker_map = MarkerMap.from_yaml("maps/marker_map.yaml")
print(f"opencv {cv2.__version__}, detector {detector.backend}, {len(marker_map)} markers")
PYTHON

CMD ["bash", "scripts/run_demo.sh"]
