import unittest

from trainer.trainer_utils import rescale_resume_step


class TestResumeStepScaling(unittest.TestCase):
    def test_preserves_step_when_global_batch_is_unchanged(self):
        self.assertEqual(
            rescale_resume_step(1000, 1, 4, saved_batch_size=40, current_batch_size=10),
            1000,
        )

    def test_scales_step_when_global_batch_grows(self):
        self.assertEqual(
            rescale_resume_step(1000, 1, 4, saved_batch_size=40, current_batch_size=40),
            250,
        )

    def test_scales_step_when_global_batch_shrinks(self):
        self.assertEqual(
            rescale_resume_step(1000, 4, 4, saved_batch_size=10, current_batch_size=5),
            2000,
        )

    def test_legacy_checkpoint_keeps_world_size_scaling(self):
        self.assertEqual(rescale_resume_step(1000, 1, 4), 250)


if __name__ == "__main__":
    unittest.main()
