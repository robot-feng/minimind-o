import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "trainer" / "train_full_dense.sh"
DATA_FILES = ("sft_t2a.parquet", "sft_a2a.parquet", "sft_i2t.parquet")
STAGES = (
    "sft_full_t2a",
    "sft_full_a2a_proj",
    "sft_full_a2a",
    "sft_full_i2t_proj",
    "sft_full_i2t",
    "sft_full_a2a_final",
    "sft_omni",
)


class TestFullTrainingPipeline(unittest.TestCase):
    def test_i2t_dry_run_includes_training_and_post_training_comparison(self):
        script = ROOT / "trainer" / "train_i2t_eval.sh"
        env = os.environ.copy()
        env.update(DRY_RUN="1", NPROC_PER_NODE="4")
        result = subprocess.run(
            ["bash", str(script)], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.startswith("[dry-run]")]
        self.assertEqual(len(commands), 5, result.stdout)
        self.assertIn("--mode vision_proj", commands[0])
        self.assertIn("--from_weight sft_full_a2a", commands[0])
        self.assertIn("--mode all", commands[1])
        self.assertIn("--from_weight sft_i2t_mini_proj", commands[1])
        self.assertIn("--weight sft_full_a2a", commands[2])
        self.assertIn("--weight sft_i2t_mini", commands[3])
        self.assertIn("--compare", commands[4])
        self.assertIn("sft_i2t_mini.jsonl", commands[4])

    def test_dry_run_plans_resumable_dense_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temp = Path(temporary_dir)
            dataset_dir = temp / "dataset"
            dataset_dir.mkdir()
            for name in DATA_FILES:
                (dataset_dir / name).write_bytes(b"placeholder")

            env = os.environ.copy()
            env.update(
                DATASET_DIR=str(dataset_dir),
                DRY_RUN="1",
                LOG_FILE=str(temp / "pipeline.log"),
                NPROC_PER_NODE="4",
                BATCH_SIZE="32",
            )
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.startswith("[dry-run]")]
        self.assertEqual(len(commands), len(STAGES), result.stdout)
        for stage, command in zip(STAGES, commands):
            with self.subTest(stage=stage):
                self.assertIn(f"--save_weight {stage}", command)
                self.assertIn("--from_resume 1", command)
                self.assertIn("--use_moe 0", command)
                self.assertIn("--nproc_per_node 4", command)
                self.assertIn("--accumulation_steps", command)
        self.assertIn("--batch_size 8 --accumulation_steps 4", commands[2])
        self.assertIn("--batch_size 8 --accumulation_steps 4", commands[5])
        for index in (0, 1, 3, 4, 6):
            self.assertIn("--batch_size 32 --accumulation_steps 1", commands[index])
        self.assertIn("--vision_dir google/tipsv2-b14", commands[-1])
        for index in (3, 4, 6):
            self.assertIn("--vision_only", commands[index])
        for index in (0, 1, 2, 5):
            self.assertNotIn("--vision_only", commands[index])
        self.assertIn("Dry run complete; no training was started.", result.stdout)

    def test_dry_run_uses_conda_torchrun_when_shell_has_no_torchrun(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temp = Path(temporary_dir)
            dataset_dir = temp / "dataset"
            dataset_dir.mkdir()
            for name in DATA_FILES:
                (dataset_dir / name).write_bytes(b"placeholder")

            conda = temp / "bin" / "conda"
            conda.parent.mkdir()
            conda.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            conda.chmod(0o755)

            env = os.environ.copy()
            env.update(
                CONDA_EXE=str(conda),
                DATASET_DIR=str(dataset_dir),
                DRY_RUN="1",
                LOG_FILE=str(temp / "pipeline.log"),
                PATH=f"{conda.parent}:/usr/bin:/bin",
            )
            result = subprocess.run(
                ["/bin/bash", str(SCRIPT)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.startswith("[dry-run]")]
        self.assertEqual(len(commands), len(STAGES), result.stdout)
        self.assertTrue(
            all(f"{conda} run --no-capture-output -n minimind torchrun" in command
                for command in commands),
            result.stdout,
        )

    def test_missing_full_dataset_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            env = os.environ.copy()
            env.update(
                DATASET_DIR=temporary_dir,
                DRY_RUN="1",
                LOG_FILE=str(Path(temporary_dir) / "pipeline.log"),
            )
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing full training dataset", result.stdout + result.stderr)
        self.assertNotIn("[dry-run]", result.stdout)

    def test_dataset_checksum_mismatch_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temp = Path(temporary_dir)
            dataset_dir = temp / "dataset"
            dataset_dir.mkdir()
            for name in DATA_FILES:
                (dataset_dir / name).write_bytes(b"not the published dataset")

            env = os.environ.copy()
            env.update(
                DATASET_DIR=str(dataset_dir),
                DRY_RUN="0",
                LOG_FILE=str(temp / "pipeline.log"),
            )
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dataset checksum mismatch", result.stdout + result.stderr)
        self.assertNotIn("[dry-run]", result.stdout)


if __name__ == "__main__":
    unittest.main()
