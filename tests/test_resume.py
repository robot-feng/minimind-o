import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from trainer.trainer_utils import get_accumulation_window_size, get_epoch_sampler, get_optimizer_step, omni_checkpoint, rescale_resume_step


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


class TestTrainingSupport(unittest.TestCase):
    def test_optimizer_schedule_counts_accumulation_updates(self):
        self.assertEqual(get_optimizer_step(0, 1, 10, 4), 1)
        self.assertEqual(get_optimizer_step(0, 4, 10, 4), 1)
        self.assertEqual(get_optimizer_step(0, 5, 10, 4), 2)
        self.assertEqual(get_optimizer_step(1, 1, 10, 4), 4)

    def test_optimizer_schedule_handles_partial_final_accumulation(self):
        self.assertEqual(get_optimizer_step(0, 9, 10, 4), 3)
        self.assertEqual(get_optimizer_step(0, 10, 10, 4), 3)

    def test_optimizer_schedule_rejects_zero_accumulation(self):
        with self.assertRaises(ValueError):
            get_optimizer_step(0, 1, 10, 0)

    def test_accumulation_window_normalizes_short_final_group(self):
        self.assertEqual(get_accumulation_window_size(1, 10, 4), 4)
        self.assertEqual(get_accumulation_window_size(8, 10, 4), 4)
        self.assertEqual(get_accumulation_window_size(9, 10, 4), 2)
        self.assertEqual(get_accumulation_window_size(10, 10, 4), 2)

    def test_accumulation_window_rejects_out_of_range_batch(self):
        with self.assertRaises(ValueError):
            get_accumulation_window_size(0, 10, 4)

    def test_single_rank_sampler_shuffles_deterministically_per_epoch(self):
        dataset = list(range(16))
        epoch_zero = get_epoch_sampler(dataset, 0)
        same_epoch = get_epoch_sampler(dataset, 0)
        next_epoch = get_epoch_sampler(dataset, 1)
        self.assertEqual(epoch_zero, same_epoch)
        self.assertNotEqual(epoch_zero, list(range(len(dataset))))
        self.assertNotEqual(epoch_zero, next_epoch)
        self.assertEqual(sorted(epoch_zero), list(range(len(dataset))))

    def test_distributed_sampler_epoch_is_set_by_helper(self):
        from torch.utils.data import DistributedSampler

        dataset = list(range(16))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)
        first = list(get_epoch_sampler(dataset, 0, sampler))
        second = list(get_epoch_sampler(dataset, 1, sampler))
        self.assertNotEqual(first, second)

    def test_compiled_auxiliary_model_checkpoint_uses_unwrapped_keys(self):
        config = SimpleNamespace(use_moe=False, hidden_size=2)
        model = torch.compile(torch.nn.Linear(2, 2), backend="eager")
        critic = torch.compile(torch.nn.Linear(2, 1), backend="eager")
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            omni_checkpoint(config, weight="compile", model=model, optimizer=optimizer,
                            save_dir=directory, critic_model=critic)
            checkpoint = torch.load(
                os.path.join(directory, "compile_2_resume.pth"),
                map_location="cpu", weights_only=False,
            )
        self.assertEqual(set(checkpoint["critic_model"]), {"weight", "bias"})


if __name__ == "__main__":
    unittest.main()
