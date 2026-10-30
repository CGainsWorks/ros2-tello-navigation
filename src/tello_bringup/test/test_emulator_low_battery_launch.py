"""Launch test: the battery runs low in flight and the aircraft comes home.

The emulator starts at 30 per cent with the in-flight drain multiplied by eight,
the same settings as the offline ``low_battery`` failsafe case, so the 25 per
cent return threshold is crossed a few seconds after take-off. The failsafe must
declare LOW_BATTERY, return and land near the launch point, without the battery
reaching the critical threshold. Simulated: the battery model is the
simulator's, not a Tello's.
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

START_BATTERY = 0.30
DRAIN = 8.0
CRITICAL = 0.15
"""aircraft.yaml battery.critical_fraction."""


@pytest.mark.launch_test
def generate_test_description():
    return harness.describe("low_battery", battery=START_BATTERY, battery_drain=DRAIN)


class TestLowBattery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.observer = harness.Observer()

    @classmethod
    def tearDownClass(cls):
        cls.observer.destroy_node()
        rclpy.shutdown()

    def test_low_battery_returns_and_lands(self, output, scenario):
        observer = self.observer
        landed = observer.wait_for(lambda: observer.landed, harness.FLIGHT_TIMEOUT)
        observer.linger(1.0)
        self.assertTrue(landed, f"never reached LANDED; failsafe events: {observer.failsafe}")

        causes = observer.causes()
        self.assertIn("LOW_BATTERY", causes, observer.failsafe)
        self.assertNotIn("CRITICAL_BATTERY", causes, observer.failsafe)
        final = [e for e in observer.failsafe if e["to"] == "LANDED"][-1]
        self.assertEqual(final["cause"], "LOW_BATTERY")

        truth = harness.load_truth(output / "truth.csv")
        down = harness.touchdown(truth)
        self.assertIsNotNone(down, "the emulator never saw the aircraft land")
        self.assertEqual(truth["flying"][-1], 0.0)
        self.assertEqual(truth["motors_cut"].max(), 0.0, "the motors were cut")
        self.assertGreater(down["battery"], CRITICAL)

        launch = np.array([truth["x"][0], truth["y"][0]])
        from_launch = float(np.linalg.norm(np.asarray(down["position"][:2]) - launch))

        summary = harness.write_summary(
            output,
            scenario,
            {
                "outcome": final["to"],
                "failsafe_events": observer.failsafe,
                "battery": {"start": START_BATTERY, "drain_multiplier": DRAIN,
                            "at_touchdown": down["battery"]},
                "touchdown_from_launch_point_m": from_launch,
                "touchdown": down,
            },
        )
        print(f"simulated summary: {summary}")
        # Home is reached within 0.5 m of the *estimate*; allow the estimate's
        # own error on top.
        self.assertLess(from_launch, 0.8, f"landed {from_launch:.2f} m from the launch point")


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_every_process_exits_cleanly(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
