#!/usr/bin/env bash
# The one-command demo.
#
# Flies the reference mission in the simulator, writes the flight log, prints the
# result and regenerates the figures. Everything it does is a script in this
# repository that can also be run on its own; this only fixes the order.

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
export PYTHONPATH=src/tello_core

echo "== validating the marker map and the mission =="
"$PYTHON" scripts/validate_map.py
"$PYTHON" scripts/validate_missions.py

echo
echo "== flying missions/perimeter.yaml =="
"$PYTHON" scripts/run_mission.py --mission missions/perimeter.yaml --out results/latest

echo
echo "== figures =="
"$PYTHON" analysis/make_figures.py --results results/latest --out analysis/figures
"$PYTHON" scripts/render_flight_frames.py --results results/latest

echo
echo "flight log   results/latest/"
echo "figures      analysis/figures/"
