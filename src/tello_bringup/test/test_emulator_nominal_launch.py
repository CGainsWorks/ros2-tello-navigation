"""Launch test: the whole graph flies missions/hop.yaml against the emulator.

Take-off, two waypoints, landing, through the UDP driver, the PyAV decoder, the
marker node, the fusion node and the mission node, with the emulator standing in
for the aircraft. Asserts the outcome and the position error against the
emulator's ground truth. All of it is simulated.
"""

import sys
import unittest
from pathlib import Path

import launch_testing
import launch_testing.asserts
import numpy as np
import pytest
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulator_flight as harness  # noqa: E402

WAYPOINTS = {"start": np.array([1.00, 0.90, 1.20]), "across": np.array([3.00, 0.90, 1.20])}
"""missions/hop.yaml, repeated here so that a change to the mission file is a
visible change to this test rather than a silent one."""


@pytest.mark.launch_test
def generate_test_description():
    return harness.describe("nominal")


class TestNominalMission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.observer = harness.Observer()

    @classmethod
    def tearDownClass(cls):
        cls.observer.destroy_node()
        rclpy.shutdown()

    def test_flies_the_mission_and_lands(self, output, scenario):
        observer = self.observer
        landed = observer.wait_for(lambda: observer.landed, harness.FLIGHT_TIMEOUT)
        observer.linger(1.0)
        self.assertTrue(landed, f"never reached LANDED; failsafe events: {observer.failsafe}")

        final = [e for e in observer.failsafe if e["to"] == "LANDED"][-1]
        self.assertEqual(final["cause"], "NONE", observer.failsafe)
        self.assertNotIn("LINK_LOSS", observer.causes())

        arrived = [w for w in observer.waypoints if w["kind"] == "arrived"]
        self.assertEqual([w["name"] for w in arrived], list(WAYPOINTS), observer.waypoints)

        truth = harness.load_truth(output / "truth.csv")
        events = harness.load_events(output / "events.jsonl")
        down = harness.touchdown(truth)
        self.assertIsNotNone(down, "the emulator never saw the aircraft land")
        self.assertEqual(truth["flying"][-1], 0.0)
        self.assertEqual(truth["motors_cut"].max(), 0.0, "the motors were cut")
        self.assertFalse(any(e["event"] == "auto_land" for e in events),
                         "the aircraft landed itself; the graph should have landed it")
        received = [e["detail"] for e in events if e["event"] == "command"]
        for command in ("command", "streamon", "takeoff", "land"):
            self.assertIn(command, received)

        # True position at the moment the graph declared each arrival.
        arrival_errors = {
            w["name"]: float(
                np.linalg.norm(
                    harness.truth_position(truth, w["wall_time"]) - WAYPOINTS[w["name"]]
                )
            )
            for w in arrived
        }
        estimate = harness.estimate_errors(observer, truth)
        self.assertGreater(len(estimate), 50, "too few airborne nav_state samples to judge")
        rms = float(np.sqrt(np.mean(estimate**2)))

        summary = harness.write_summary(
            output,
            scenario,
            {
                "outcome": final["to"],
                "failsafe_events": observer.failsafe,
                "waypoints": observer.waypoints,
                "true_arrival_error_m": arrival_errors,
                "estimate_error_m": {
                    "rms": rms,
                    "p95": float(np.percentile(estimate, 95)),
                    "max": float(estimate.max()),
                    "samples": int(len(estimate)),
                },
                "touchdown": down,
            },
        )
        print(f"simulated summary: {summary}")

        # Acceptance bounds, not results. Arrival is declared within 0.20 m of
        # the *estimate*; the bound on the true error allows for the estimate
        # being wrong by as much again.
        for name, error in arrival_errors.items():
            self.assertLess(error, 0.40, f"{name}: {error:.2f} m from the waypoint in truth")
        self.assertLess(rms, 0.25, f"estimate RMS error {rms:.2f} m against truth")


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_every_process_exits_cleanly(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
