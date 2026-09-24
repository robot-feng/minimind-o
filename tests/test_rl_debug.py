import unittest

import torch

from trainer.rl_utils import format_agent_rollout_debug, format_rollout_debug


class TestRolloutDebugFormatting(unittest.TestCase):
    def test_groups_completions_and_rewards_by_prompt(self):
        output = format_rollout_debug(
            7, ["first prompt", "second prompt"],
            ["a1", "a2", "b1", "b2"], torch.tensor([0.1, 0.2, -0.1, -0.2]), 2,
        )

        self.assertIn("step=7, sample[0]", output)
        self.assertIn("step=7, sample[1]", output)
        self.assertLess(output.index("first prompt"), output.index("a1"))
        self.assertLess(output.index("a1"), output.index("a2"))
        self.assertIn("reward=0.2000", output)
        self.assertLess(output.index("second prompt"), output.index("b1"))

    def test_rejects_incomplete_text_generation_groups(self):
        with self.assertRaisesRegex(ValueError, r"prompts \* num_generations"):
            format_rollout_debug(1, ["prompt"], ["one", "two"], [0.1], 2)

    def test_formats_multi_turn_agent_episodes_and_outcomes(self):
        episodes = [
            {"prompt": "find a fact", "tools": [{"name": "search"}],
             "turns": ["call search", "answer"], "unfinished": False},
            {"prompt": "find a fact", "tools": [],
             "turns": ["answer"], "unfinished": True},
        ]
        output = format_agent_rollout_debug(4, episodes, [1.0, -0.5], 2)

        self.assertIn("step=4, sample[0]", output)
        self.assertIn("gen[0] tools=search", output)
        self.assertIn("turn[1] RESPONSE_BEGIN", output)
        self.assertIn("reward=1.0000, unfinished=False", output)
        self.assertIn("tools=(none)", output)
        self.assertIn("reward=-0.5000, unfinished=True", output)

    def test_rejects_misaligned_agent_episodes_and_rewards(self):
        episode = {"prompt": "p", "tools": [], "turns": [], "unfinished": False}
        with self.assertRaisesRegex(ValueError, "complete generation groups"):
            format_agent_rollout_debug(1, [episode, episode], [0.1], 2)


if __name__ == "__main__":
    unittest.main()
