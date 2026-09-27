import math
import unittest

from trainer.training_health import LossStabilityTracker


class TestLossStabilityTracker(unittest.TestCase):
    def test_stability_requires_five_finite_reports_at_log_cadence(self):
        tracker = LossStabilityTracker(required_observations=5, log_interval=100)

        self.assertFalse(tracker.observe(100, 1000, 2.0))
        self.assertFalse(tracker.observe(200, 1000, 1.8))
        self.assertFalse(tracker.observe(300, 1000, 1.6))
        self.assertFalse(tracker.observe(400, 1000, 1.5))
        self.assertTrue(tracker.observe(500, 1000, 1.4))

    def test_missing_or_out_of_order_report_resets_stability(self):
        tracker = LossStabilityTracker(required_observations=3, log_interval=10)
        self.assertFalse(tracker.observe(10, 100, 2.0))
        self.assertFalse(tracker.observe(20, 100, 2.0))
        self.assertFalse(tracker.observe(40, 100, 2.0))
        self.assertFalse(tracker.observe(30, 100, 2.0))
        self.assertFalse(tracker.observe(40, 100, 2.0))
        self.assertTrue(tracker.observe(50, 100, 2.0))

    def test_non_finite_loss_is_rejected(self):
        for loss in (math.nan, math.inf, -math.inf):
            with self.subTest(loss=loss):
                with self.assertRaises(ValueError):
                    LossStabilityTracker().observe(100, 1000, loss)

    def test_invalid_step_is_rejected(self):
        tracker = LossStabilityTracker()
        for step, total in ((0, 100), (101, 100)):
            with self.subTest(step=step, total=total):
                with self.assertRaises(ValueError):
                    tracker.observe(step, total, 1.0)

    def test_invalid_tracker_configuration_is_rejected(self):
        for observations, interval in ((0, 100), (5, 0)):
            with self.subTest(observations=observations, interval=interval):
                with self.assertRaises(ValueError):
                    LossStabilityTracker(observations, interval)


if __name__ == "__main__":
    unittest.main()
