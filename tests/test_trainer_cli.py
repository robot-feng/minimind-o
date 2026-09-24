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


if __name__ == "__main__":
    unittest.main()
