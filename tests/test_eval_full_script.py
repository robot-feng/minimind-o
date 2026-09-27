import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_full_omni.sh"


class FullEvaluationScriptTests(unittest.TestCase):
    def test_dry_run_plans_reference_visual_and_multimodal_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            results_dir = temporary_path / "results"
            audio_dir = temporary_path / "audio"
            log_path = temporary_path / "evaluation.log"
            env = os.environ.copy()
            env.update({
                "DRY_RUN": "1",
                "PYTHON_BIN": sys.executable,
                "CONDA_BIN_DIR": str(Path(sys.executable).parent),
                "EVAL_RESULTS_DIR": str(results_dir),
                "EVAL_AUDIO_DIR": str(audio_dir),
                "EVAL_LOG": str(log_path),
            })
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )

            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("--mode 0,2,6", output.replace("\\,", ","))
            self.assertIn("--mode 4 --prompt_lang 0", output)
            self.assertIn("--weight sft_i2t_mini --mode 4 --text_only --prompt_lang 0", output)
            self.assertIn("--weight sft_omni --mode 4 --text_only --prompt_lang 0", output)
            self.assertIn("--mode 4 --text_only --prompt_lang 1", output)
            self.assertIn(str(results_dir / "sft_omni_image_audio_en.jsonl"), output)
            self.assertIn(str(results_dir / "sft_omni_image_text_zh.jsonl"), output)
            self.assertIn(str(results_dir / "sft_omni_image_audio_en_metrics.json"), output)
            self.assertIn(str(results_dir / "sft_omni_image_text_zh_metrics.json"), output)
            self.assertIn(str(results_dir / "sft_omni_vs_sft_i2t_mini_image_text_en.json"), output)
            self.assertIn(str(results_dir / "sft_omni_visual_comparison.png"), output)
            self.assertIn(str(audio_dir), output)
            self.assertFalse((results_dir / "sft_omni_image_audio_en.jsonl").exists())
            self.assertFalse((results_dir / "sft_omni_image_audio_en_metrics.json").exists())

    def test_comparison_plotter_writes_a_png(self):
        from PIL import Image

        comparison = {
            "before": {"mean_concept_recall": 0.25, "all_concepts_hit_rate": 0.0},
            "after": {"mean_concept_recall": 0.5, "all_concepts_hit_rate": 0.25},
            "per_image": [
                {
                    "source": f"image-{index:02d}-test.jpg",
                    "before_concept_recall": 0.25,
                    "after_concept_recall": 0.5,
                }
                for index in range(1, 10)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            comparison_path = Path(temporary_directory) / "comparison.json"
            image_path = Path(temporary_directory) / "comparison.png"
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "plot_visual_comparison.py"),
                 str(comparison_path), "--output", str(image_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            with Image.open(image_path) as chart:
                self.assertEqual(chart.format, "PNG")
                self.assertEqual(chart.size, (1600, 900))

    def test_comparison_plotter_rejects_missing_image_answers(self):
        comparison = {
            "before": {"mean_concept_recall": 0.25, "all_concepts_hit_rate": 0.0},
            "after": {"mean_concept_recall": 0.5, "all_concepts_hit_rate": 0.25},
            "missing_before": ["image-01-test.jpg"],
            "missing_after": [],
            "per_image": [{
                "source": "image-01-test.jpg",
                "before_concept_recall": None,
                "after_concept_recall": 0.5,
            }],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            comparison_path = Path(temporary_directory) / "comparison.json"
            image_path = Path(temporary_directory) / "comparison.png"
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "plot_visual_comparison.py"),
                 str(comparison_path), "--output", str(image_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing image answers", completed.stderr)
            self.assertFalse(image_path.exists())


if __name__ == "__main__":
    unittest.main()
