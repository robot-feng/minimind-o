import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from eval_omni import eval_sample, save_visual_result


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, open_thinking=False):
        return "serialized prompt"

    def __call__(self, text):
        return SimpleNamespace(data={"input_ids": [1, 3, 4]})

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(map(str, token_ids))


class FakeTextModel:
    def __init__(self):
        self.call = None

    def generate_text(self, input_ids, **kwargs):
        self.call = (input_ids, kwargs)
        generated = torch.tensor([[77, 78]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, generated), dim=1)


class TestTextOnlyEvaluation(unittest.TestCase):
    def test_results_jsonl_requires_text_only_visual_mode(self):
        entrypoint = Path(__file__).resolve().parents[1] / "eval_omni.py"
        for args in (
            ["--results_jsonl", "out/result.jsonl", "--mode", "4"],
            ["--text_only", "--results_jsonl", "out/result.jsonl", "--mode", "0"],
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, str(entrypoint), *args],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("--results_jsonl requires", result.stderr)

    def test_visual_result_is_saved_as_utf8_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            save_visual_result(str(path), "image", "猫.jpg", "请描述", "一只猫")
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(row, {
            "mode": "image", "source": "猫.jpg", "prompt": "请描述", "answer": "一只猫",
        })

    def test_missing_answer_is_not_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            save_visual_result(str(path), "video", "clip.avi", "描述", None)
            self.assertFalse(path.exists())

    def test_eval_sample_routes_visual_inputs_to_text_generation(self):
        model = FakeTextModel()
        tokenizer = FakeTokenizer()
        args = SimpleNamespace(
            device="cpu", open_thinking=0, text_only=True,
            max_new_tokens=12, temperature=0, top_p=1.0,
        )
        pixels = {"pixel_values": torch.ones(1, 3, 8, 8)}

        answer = eval_sample(
            model, tokenizer, args, 0, "describe this image", None,
            "unused.mp3", pixel_values=pixels,
        )

        self.assertEqual(answer, "77 78")
        input_ids, call_args = model.call
        self.assertEqual(tuple(input_ids.shape), (1, 3))
        self.assertEqual(call_args["eos_token_id"], tokenizer.eos_token_id)
        self.assertEqual(call_args["max_new_tokens"], 12)
        self.assertEqual(call_args["temperature"], 0)
        self.assertEqual(call_args["top_p"], 1.0)
        self.assertIs(call_args["pixel_values"], pixels)


if __name__ == "__main__":
    unittest.main()
