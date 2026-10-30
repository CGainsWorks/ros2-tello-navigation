"""Launch test: the radio link drops in flight and the graph falls silent.

The emulator drops every datagram in both directions from four seconds after
take-off for eleven seconds, the same schedule as the offline
``link-loss-early`` fault. The failsafe must declare LINK_LOSS, stop sending
``rc`` (so the aircraft's own dead-man timer is free to act), keep asking for a
landing and see the aircraft on the ground once the link returns. Simulated.
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

OUTAGE_START = 4.0
OUTAGE_DURATION = 11.0


@pytest.mark.launch_test
def generate_test_description():
    return harness.describe(
        "link_loss", outage_start=OUTAGE_START, outage_duration=OUTAGE_DURATION
    )


class TestLinkLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.observer = harness.Observer()

    @classmethod
    def tearDownClass(cls):
        cls.observer.destroy_node()
        rclpy.shutdown()

    def test_link_loss_lands_the_aircraft_without_rc(self, output, scenario):
        observer = self.observer
        landed = observer.wait_for(lambda: observer.landed, harness.FLIGHT_TIMEOUT)
        observer.linger(1.0)
        self.assertTrue(landed, f"never reached LANDED; failsafe events: {observer.failsafe}")

        link_loss = [e for e in observer.failsafe if e["cause"] == "LINK_LOSS"]
        self.assertTrue(link_loss, observer.failsafe)
        self.assertEqual(link_loss[0]["to"], "LAND")
        final = [e for e in observer.failsafe if e["to"] == "LANDED"][-1]
        self.assertEqual(final["cause"], "LINK_LOSS")

        truth = harness.load_truth(output / "truth.csv")
        events = harness.load_events(output / "events.jsonl")
        down = harness.touchdown(truth)
        self.assertIsNotNone(down, "the emulator never saw the aircraft land")
        self.assertEqual(truth["flying"][-1], 0.0)

        link_down = [e["wall_time"] for e in events if e["event"] == "link_down"]
        link_up = [e["wall_time"] for e in events if e["event"] == "link_up"]
        self.assertEqual(len(link_down), 1)
        self.assertEqual(len(link_up), 1)
        self.assertGreater(down["wall_time"], link_down[0], "landed before the outage began")

        # The headline property: from the moment the link came back until the
        # aircraft was on the ground, no stick command reached it. Anything the
        # graph sent during the outage was dropped by the emulator and is
        # logged as such, not as received.
        rc_after = [
            e
            for e in events
            if e["event"] == "command"
            and e["detail"].startswith("rc ")
            and link_up[0] <= e["wall_time"] <= down["wall_time"]
        ]
        self.assertEqual(rc_after, [], "rc reached the aircraft after LINK_LOSS")
        # How far the aircraft went while nobody could command it. Reported, not
        # bounded: the emulator, like the simulator, holds the last rc command
        # through an outage, so this is whatever the last command was, flown for
        # up to fifteen seconds. Whether a Tello does the same is in HARDWARE.md.
        window = (truth["wall_time"] >= link_down[0]) & (truth["wall_time"] <= down["wall_time"])
        path = np.stack([truth[a][window] for a in ("x", "y")], axis=1)
        travelled = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        landed_by = (
            "own dead-man timer"
            if any(e["event"] == "auto_land" for e in events)
            else "land command after the link returned"
        )

        summary = harness.write_summary(
            output,
            scenario,
            {
                "outcome": final["to"],
                "failsafe_events": observer.failsafe,
                "outage": {"start_after_takeoff_s": OUTAGE_START,
                           "duration_s": OUTAGE_DURATION},
                "commands_dropped_in_outage": sum(e["event"] == "dropped" for e in events),
                "rc_received_between_link_return_and_touchdown": len(rc_after),
                "landed_by": landed_by,
                "position_at_link_loss": harness.truth_position(truth, link_down[0]).tolist(),
                "horizontal_path_from_link_loss_to_touchdown_m": travelled,
                "touchdown": down,
            },
        )
        print(f"simulated summary: {summary}")


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_every_process_exits_cleanly(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
