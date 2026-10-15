# One command per thing anyone would want to do.
#
# `make demo` is the one the README leads with and the one CI runs, so it cannot
# drift from what a reader is told to type.

PYTHON ?= python
export PYTHONPATH := src/tello_core

.DEFAULT_GOAL := help
.PHONY: help setup demo quick mission failsafe detection survey tuning noise \
        calibrate figures results reproduce test test-fast lint validate \
        visibility markers docker docker-demo clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

demo:  ## Fly the reference mission in the simulator and print the result
	$(PYTHON) scripts/run_mission.py --mission missions/perimeter.yaml --out results/latest

quick:  ## A short flight, for checking a change without waiting
	$(PYTHON) scripts/run_mission.py --mission missions/hop.yaml --out results/quick \
		--max-flight-time 30

mission: demo  ## Alias for `demo`

failsafe:  ## Fly every failure case and write the failsafe evidence
	bash scripts/run_failsafe_cases.sh

detection:  ## Measure the detector: range, yaw rate, incidence
	$(PYTHON) scripts/measure_detection.py --trials 20

survey:  ## Position error against surveyed check points
	$(PYTHON) scripts/survey_check.py

tuning:  ## Sweep the plant and report how much of the tuning survives
	$(PYTHON) scripts/tuning_sweep.py

noise:  ## Estimate the filter's noise parameters from the last flight log
	$(PYTHON) scripts/estimate_noise.py --results results/latest

calibrate:  ## Self-test the camera calibration against synthetic chessboards
	$(PYTHON) scripts/calibrate_camera.py --self-test

figures:  ## Regenerate every figure from the committed results
	$(PYTHON) analysis/make_figures.py
	$(PYTHON) scripts/render_flight_frames.py

results:  ## Regenerate docs/RESULTS.md from the committed results
	$(PYTHON) scripts/make_results.py

reproduce:  ## Check the committed results can be regenerated
	$(PYTHON) scripts/check_reproducible.py

test:  ## Run the whole test suite
	$(PYTHON) -m pytest

test-fast:  ## Run everything except the end-to-end flights
	$(PYTHON) -m pytest -m "not slow"

lint:  ## Ruff over everything
	ruff check src tests scripts analysis

validate:  ## Check the marker map and every mission
	$(PYTHON) scripts/validate_map.py
	$(PYTHON) scripts/validate_missions.py

visibility:  ## Print what each waypoint can see and at what heading
	$(PYTHON) scripts/validate_missions.py

markers:  ## Generate printable marker sheets into build/markers/
	$(PYTHON) scripts/print_markers.py --out build/markers

docker:  ## Build the container image
	docker build -t tello-navigation:latest .

docker-demo:  ## Run the demo inside the container
	docker compose up --build

clean:  ## Remove generated files, keeping the committed results
	# Not results/latest. It is the committed reference flight every number in
	# docs/RESULTS.md comes from, so deleting it here would make `make clean`
	# followed by `make results` quietly produce a document with nothing in it.
	rm -rf results/quick results/local results/ci build .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
