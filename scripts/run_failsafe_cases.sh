#!/usr/bin/env bash
# Fly every failure the failsafe machine is supposed to handle and keep the logs.
#
# This is the evidence behind docs/FAILSAFE.md. Each case is a flight of the same
# mission with one thing wrong and each writes a full trajectory and event log
# into results/failsafe/, so a claim about what the aircraft does when the link
# drops can be checked rather than taken on trust.
#
# Takes about ten minutes. `make failsafe` runs it.

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

OUT="results/failsafe"
mkdir -p "$OUT"

run() {
    local name="$1"
    shift
    echo
    echo "=============================================================="
    echo "  $name"
    echo "=============================================================="
    "$PYTHON" scripts/run_mission.py \
        --mission missions/perimeter.yaml \
        --out "$OUT" --stem "$name" "$@"
}

# Nothing wrong. The control case: whatever the others do differently is caused
# by the fault and not by the mission.
run nominal --fault nominal

# The link drops eighteen seconds in and never comes back. The aircraft must
# land and the controller must stop transmitting so the aircraft's own
# fifteen-second dead-man timer is free to act.
run link_loss --fault link-loss

# Video only. Telemetry keeps arriving, so the aircraft is controllable but
# nothing can localise it. HOLD, then a landing.
run video_loss --fault video-loss

# A battery that starts healthy and crosses the return threshold in flight. The
# drain is multiplied so that happens inside a thirty-second mission rather than
# after eight minutes; the multiplier is recorded in the results metadata.
run low_battery --fault nominal --battery 0.30 --battery-drain 8

# The same, drained hard enough to cross the critical threshold: no return leg,
# land where it is.
run critical_battery --fault nominal --battery 0.30 --battery-drain 40

# A battery already below the return threshold before take-off. The correct
# response is to refuse the flight, not to attempt to fly home from the ground
# and getting that wrong left an aircraft sitting on the floor with its motors off
# until the wall-clock limit expired.
run refused_battery --fault nominal --battery 0.20

# A link that works badly rather than not at all. Twenty per cent of telemetry
# and a quarter of the video lost, plus malformed packets. Nothing should trip:
# this is the case that says the thresholds are not so tight that ordinary
# wireless attrition looks like a failure.
run lossy_link --fault lossy

echo
echo "=============================================================="
echo "  summary"
echo "=============================================================="
"$PYTHON" scripts/summarise_failsafe.py --results "$OUT"
