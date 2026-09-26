import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "train_pretrain.py",
    "train_full_sft.py",
    "train_lora.py",
    "train_distillation.py",
    "train_dpo_omni.py",
    "train_ppo.py",
    "train_grpo.py",
    "train_agent.py",
    "train_sft_omni.py",
    "train_tokenizer.py",
)
EXPECTED_OPTIONS = {
    "train_ppo.py": ("--debug_mode", "--debug_interval", "--debug_log_ratio"),
    "train_grpo.py": ("--debug_mode", "--debug_interval"),
    "train_agent.py": ("--debug_mode", "--debug_interval"),
    "train_sft_omni.py": ("--vision_only",),
}


class TestTrainerCLI(unittest.TestCase):
    def test_public_training_entrypoints_load_and_parse_help(self):
        for entrypoint in ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "trainer" / entrypoint), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())
                for option in EXPECTED_OPTIONS.get(entrypoint, ()):
                    self.assertIn(option, result.stdout)

    def test_vision_only_rejects_audio_projector_mode(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "trainer" / "train_sft_omni.py"),
             "--vision_only", "--mode", "audio_proj"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)

    def test_agent_total_length_must_fit_a_generation(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "trainer" / "train_agent.py"),
             "--max_total_len", "8", "--max_gen_len", "8"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("max_total_len must exceed max_gen_len", result.stderr)


if __name__ == "__main__":
    unittest.main()
