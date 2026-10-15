"""Make the ROS 2 package sources importable without a colcon build.

Each ``src/<pkg>`` directory is an ament package whose importable module lives
one level down. Putting those directories on ``sys.path`` here means the test
suite, the analysis scripts and CI exercise exactly the code that gets
installed, with no copy step and no ROS 2 installation anywhere.

That matters more here than it usually would. The algorithms in
``src/tello_core`` are the whole substance of this repository and the ROS 2
nodes in ``src/tello_localisation`` and ``src/tello_control`` are thin wrappers
that subscribe, call into ``tello_core`` and publish. Testing the wrappers needs
ROS 2. Testing the substance does not and it should not have to.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

_PACKAGE_DIRS = [
    REPO_ROOT / "src" / "tello_core",
    REPO_ROOT / "src" / "tello_localisation",
    REPO_ROOT / "src" / "tello_control",
    REPO_ROOT / "src" / "tello_bringup",
    REPO_ROOT,
]

for _path in _PACKAGE_DIRS:
    entry = str(_path)
    if _path.is_dir() and entry not in sys.path:
        sys.path.insert(0, entry)
