import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eval_omni import (
    eval_sample,
    format_image_prompt,
    generate_greedy_text,
    load_reference_vision,
    needs_audio_encoder,
    parse_modes,
    prepare_reference_image_inputs,
    resolve_hf_snapshot,
    save_visual_result,
)
from eval_visual_metrics import compare_visual_results, score_visual_results


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


class FakeReferenceModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, pixel_values=None, **kwargs):
        self.calls.append({
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
            **kwargs,
        })
        token = 5 if len(self.calls) == 1 else 2
        logits = torch.full((input_ids.size(0), 1, 8), -10.0)
        logits[:, :, token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=("cached",))


class FakeAudioModel:
    def generate(self, input_ids, *args, **kwargs):
        self.call = (input_ids, kwargs)
        yield torch.tensor([[77]], device=input_ids.device), [list(range(8))]
        yield torch.tensor([[77, 78]], device=input_ids.device), [list(range(8))]


class TestTextOnlyEvaluation(unittest.TestCase):
    def test_hf_snapshot_download_falls_back_to_official_endpoint(self):
        with patch.dict("os.environ", {
            "MINIMIND_HF_ENDPOINT": "https://custom-mirror.invalid",
            "HF_ENDPOINT": "https://configured-mirror.invalid",
        }), patch("huggingface_hub.snapshot_download", side_effect=[OSError("mirror failed"), "/cache/model"]) as download:
            resolved = resolve_hf_snapshot("org/reference-model")

        self.assertEqual(resolved, "/cache/model")
        self.assertEqual(
            [call.kwargs["endpoint"] for call in download.call_args_list],
            ["https://custom-mirror.invalid", "https://huggingface.co"],
        )

    def test_hf_snapshot_uses_existing_local_directory(self):
        with tempfile.TemporaryDirectory() as model_path, patch("huggingface_hub.snapshot_download") as download:
            self.assertEqual(resolve_hf_snapshot(model_path), model_path)

        download.assert_not_called()

    def test_reference_siglip_loader_uses_matching_hf_image_processor(self):
        encoder = torch.nn.Linear(2, 2)
        processor = object()
        with tempfile.TemporaryDirectory() as model_path:
            with patch("eval_omni.AutoModel.from_pretrained", return_value=encoder) as load_model, \
                    patch("eval_omni.AutoImageProcessor.from_pretrained", return_value=processor) as load_processor:
                loaded_encoder, loaded_processor = load_reference_vision(model_path)

        self.assertIs(loaded_encoder, encoder)
        self.assertIs(loaded_processor, processor)
        self.assertFalse(encoder.weight.requires_grad)
        self.assertTrue(load_model.call_args.kwargs["local_files_only"])
        self.assertTrue(load_processor.call_args.kwargs["local_files_only"])

    def test_reference_loader_keeps_tips_custom_processor(self):
        encoder, processor = object(), object()
        with patch("eval_omni.MiniMindOmni.load_vision", return_value=(encoder, processor)) as load_vision:
            loaded = load_reference_vision("google/tipsv2-b14")

        self.assertEqual(loaded, (encoder, processor))
        load_vision.assert_called_once_with("google/tipsv2-b14")

    def test_reference_prompt_keeps_upstream_text_then_image_layout(self):
        model = SimpleNamespace(_use_reference_image_layout=True)
        prompt = format_image_prompt(model, "<image>" * 64, " describe this image ")

        self.assertEqual(prompt, f"describe this image\n\n{'<image>' * 64}")

    def test_local_prompt_keeps_visual_tokens_before_text(self):
        prompt = format_image_prompt(SimpleNamespace(), "<image>" * 64, "describe this image")

        self.assertEqual(prompt, f"{'<image>' * 64}\n\ndescribe this image")

    def test_reference_image_input_strips_metadata_and_squeezes_one_frame(self):
        frames = torch.ones(1, 1, 3, 8, 8)
        result = prepare_reference_image_inputs({
            "pixel_values": frames,
            "static_image_mask": torch.tensor([True]),
        })

        self.assertEqual(set(result), {"pixel_values"})
        self.assertTrue(torch.equal(result["pixel_values"], frames[:, 0]))

    def test_reference_image_input_rejects_video_frames(self):
        with self.assertRaisesRegex(ValueError, "one image frame"):
            prepare_reference_image_inputs({"pixel_values": torch.ones(1, 2, 3, 8, 8)})

    def test_reference_greedy_generation_passes_image_only_on_prefill_and_stops_at_eos(self):
        model = FakeReferenceModel()
        prompt = torch.tensor([[1, 3, 4]])
        pixels = {"pixel_values": torch.ones(1, 1, 3, 8, 8), "static_image_mask": torch.tensor([True])}

        result = generate_greedy_text(model, prompt, eos_token_id=2, max_new_tokens=8, pixel_values=pixels)

        self.assertEqual(result.tolist(), [[1, 3, 4, 5, 2]])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(tuple(model.calls[0]["pixel_values"]["pixel_values"].shape), (1, 3, 8, 8))
        self.assertIsNone(model.calls[1]["pixel_values"])
        self.assertIsNotNone(model.calls[1]["past_key_values"])

    def test_eval_sample_uses_reference_greedy_fallback(self):
        model = FakeReferenceModel()
        args = SimpleNamespace(
            device="cpu", open_thinking=0, text_only=True,
            max_new_tokens=8, temperature=0, top_p=1.0,
        )

        answer = eval_sample(
            model, FakeTokenizer(), args, 0, "describe this image", None,
            "unused.mp3", pixel_values={"pixel_values": torch.ones(1, 1, 3, 8, 8)},
        )

        self.assertEqual(answer, "5 2")

    def test_results_jsonl_requires_visual_mode(self):
        entrypoint = Path(__file__).resolve().parents[1] / "eval_omni.py"
        for args in (
            ["--results_jsonl", "out/result.jsonl", "--mode", "0"],
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, str(entrypoint), *args],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 2)
        self.assertIn("--results_jsonl requires", result.stderr)

    def test_audio_encoder_loads_only_for_audio_input_modes(self):
        def args(mode, text_only=False):
            return SimpleNamespace(mode=mode, text_only=text_only)

        self.assertEqual(parse_modes("4,6"), {"4", "6"})
        self.assertEqual(parse_modes("-1"), set("0123456"))
        self.assertFalse(needs_audio_encoder(args("0,1,3,4,6")))
        self.assertTrue(needs_audio_encoder(args("2")))
        self.assertTrue(needs_audio_encoder(args("5")))
        self.assertFalse(needs_audio_encoder(args("2", text_only=True)))

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

    def test_audio_evaluation_returns_caption_and_can_save_jsonl_result(self):
        model = FakeAudioModel()
        tokenizer = FakeTokenizer()
        args = SimpleNamespace(
            device="cpu", open_thinking=0, text_only=False,
            max_new_tokens=12, temperature=0, top_p=1.0,
            decode_audio=0, output_dir="unused",
        )
        pixels = {"pixel_values": torch.ones(1, 3, 8, 8)}

        answer = eval_sample(
            model, tokenizer, args, 0, "Please describe this image.", None,
            "unused.mp3", pixel_values=pixels,
        )

        self.assertEqual(answer, "77 78")
        input_ids, call_args = model.call
        self.assertEqual(tuple(input_ids.shape), (1, 3))
        self.assertTrue(call_args["stream"])
        self.assertTrue(call_args["return_audio_codes"])
        self.assertIs(call_args["pixel_values"], pixels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio_results.jsonl"
            save_visual_result(str(path), "image", "cat.jpg", "Please describe this image.", answer)
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["answer"], "77 78")

    def test_audio_evaluation_returns_empty_caption_when_stream_has_no_text(self):
        class EmptyAudioModel:
            def generate(self, *args, **kwargs):
                return iter(())

        args = SimpleNamespace(
            device="cpu", open_thinking=0, text_only=False,
            max_new_tokens=12, temperature=0, top_p=1.0,
            decode_audio=0, output_dir="unused",
        )
        answer = eval_sample(
            EmptyAudioModel(), FakeTokenizer(), args, 0, "describe", None,
            "unused.mp3",
        )
        self.assertEqual(answer, "")


class TestVisualMetrics(unittest.TestCase):
    def test_metrics_match_english_chinese_aliases_and_ignore_video_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references.json"
            references.write_text(json.dumps({
                "cat.jpg": [
                    {"concept": "cat", "aliases": ["cat", "猫"]},
                    {"concept": "moon", "aliases": ["moon", "月亮"]},
                ],
                "fruit.jpg": [
                    {"concept": "apple", "aliases": ["apple", "苹果"]},
                ],
            }), encoding="utf-8")
            results = root / "results.jsonl"
            results.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
                {"mode": "image", "source": "cat.jpg", "answer": "橘猫在月亮下"},
                {"mode": "image", "source": "fruit.jpg", "answer": "A cat naps."},
                {"mode": "video", "source": "clip.avi", "answer": "cat"},
            ]) + "\n", encoding="utf-8")

            metrics = score_visual_results(results, references)

        self.assertEqual(metrics["expected_images"], 2)
        self.assertEqual(metrics["evaluated_images"], 2)
        self.assertEqual(metrics["response_count"], 2)
        self.assertEqual(metrics["missing_images"], [])
        self.assertAlmostEqual(metrics["mean_concept_recall"], 0.5)
        self.assertAlmostEqual(metrics["all_concepts_hit_rate"], 0.5)

    def test_english_concept_matching_uses_word_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references.json"
            references.write_text(json.dumps({
                "image.jpg": [{"concept": "cat", "aliases": ["cat"]}],
            }), encoding="utf-8")
            results = root / "results.jsonl"
            results.write_text(json.dumps({
                "mode": "image", "source": "image.jpg", "answer": "education"}),
                encoding="utf-8")

            metrics = score_visual_results(results, references)

        self.assertEqual(metrics["mean_concept_recall"], 0.0)
        self.assertEqual(metrics["all_concepts_hit_rate"], 0.0)

    def test_missing_reference_images_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references.json"
            references.write_text(json.dumps({
                "expected.jpg": [{"concept": "cat", "aliases": ["cat"]}],
            }), encoding="utf-8")
            results = root / "results.jsonl"
            results.write_text("", encoding="utf-8")

            metrics = score_visual_results(results, references)

        self.assertEqual(metrics["evaluated_images"], 0)
        self.assertEqual(metrics["missing_images"], ["expected.jpg"])

    def test_comparison_aligns_answers_by_source_and_reports_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references.json"
            references.write_text(json.dumps({
                "cat.jpg": [
                    {"concept": "cat", "aliases": ["cat", "猫"]},
                    {"concept": "moon", "aliases": ["moon", "月亮"]},
                ],
                "fruit.jpg": [{"concept": "apple", "aliases": ["apple", "苹果"]}],
            }), encoding="utf-8")
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            before.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
                {"mode": "image", "source": "cat.jpg", "answer": "猫在月亮下"},
                {"mode": "image", "source": "fruit.jpg", "answer": "水果"},
            ]) + "\n", encoding="utf-8")
            after.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
                {"mode": "image", "source": "fruit.jpg", "answer": "一个苹果"},
                {"mode": "image", "source": "cat.jpg", "answer": "猫"},
            ]) + "\n", encoding="utf-8")

            comparison = compare_visual_results(before, after, references)

        self.assertAlmostEqual(comparison["before"]["mean_concept_recall"], 0.5)
        self.assertAlmostEqual(comparison["after"]["mean_concept_recall"], 0.75)
        self.assertAlmostEqual(comparison["delta"]["mean_concept_recall"], 0.25)
        self.assertEqual(comparison["per_image"][0]["source"], "cat.jpg")
        self.assertEqual(comparison["per_image"][0]["before_answers"], ["猫在月亮下"])
        self.assertEqual(comparison["per_image"][0]["after_answers"], ["猫"])
        self.assertAlmostEqual(comparison["per_image"][0]["recall_delta"], -0.5)


if __name__ == "__main__":
    unittest.main()
