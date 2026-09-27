import io
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoTokenizer
from unittest.mock import patch

from dataset.omni_dataset import OmniDataset
from dataset.prepare_i2t_subset import prepare_i2t_subset
from trainer.train_sft_omni import omni_collate_fn


class TestI2TSubsetPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.source = self.root / "i2t.parquet"
        rows = 400
        table = pa.table({
            "conversations": pa.array(
                [json.dumps([{"role": "assistant", "content": f"caption {i}"}]) for i in range(rows)],
                type=pa.large_string(),
            ),
            "image_bytes": pa.array([f"image-{i}".encode() for i in range(rows)], type=pa.large_binary()),
        })
        pq.write_table(table, self.source, row_group_size=10)

    def tearDown(self):
        self.temporary_dir.cleanup()

    def test_samples_exact_count_deterministically_across_row_groups(self):
        first = self.root / "first.parquet"
        second = self.root / "second.parquet"
        stats = prepare_i2t_subset(self.source, first, max_samples=6, seed=9)
        prepare_i2t_subset(self.source, second, max_samples=6, seed=9)

        self.assertEqual(stats, (400, 6, 16))
        first_table = pq.read_table(first)
        self.assertTrue(first_table.equals(pq.read_table(second)))
        self.assertEqual(first_table.num_rows, 6)
        selected = [int(value.decode().split("-")[1]) for value in first_table["image_bytes"].to_pylist()]
        self.assertEqual(len(set(selected)), 6)
        self.assertGreater(len({value // 10 for value in selected}), 4)

    def test_caps_at_dataset_size(self):
        output = self.root / "all.parquet"
        _, sampled, _ = prepare_i2t_subset(self.source, output, max_samples=500, seed=1)
        self.assertEqual(sampled, 400)
        self.assertEqual(pq.read_table(output).num_rows, 400)

    def test_rejects_invalid_sizes_and_same_input_output(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare_i2t_subset(self.source, self.root / "empty.parquet", max_samples=0)
        with self.assertRaisesRegex(ValueError, "different"):
            prepare_i2t_subset(self.source, self.source, max_samples=10)


class FakeVisionProcessor:
    image_size = 32

    def __call__(self, images, return_tensors):
        if return_tensors != "pt" or images.mode != "RGB":
            raise ValueError("expected an RGB image and PyTorch tensors")
        return {"pixel_values": torch.ones(1, 3, self.image_size, self.image_size)}


class TestI2TTrainingDataset(unittest.TestCase):
    def make_dataset(self, directory, content, image_bytes, vision_processor):
        table = pa.table({
            "conversations": pa.array([json.dumps([
                {"role": "user", "content": content},
                {"role": "assistant", "content": "A reply."},
            ])], type=pa.large_string()),
            "image_bytes": pa.array([image_bytes], type=pa.large_binary()),
        })
        path = Path(directory) / "images.parquet"
        pq.write_table(table, path)
        tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")
        return OmniDataset(
            str(path), tokenizer, vision_processor=vision_processor,
            max_length=128, scheduled_sampling=0,
        )

    def test_image_example_loads_without_audio_and_preserves_visual_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (12, 8), "orange")
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            table = pa.table({
                "conversations": pa.array([json.dumps([
                    {"role": "user", "content": "<image> Describe this."},
                    {"role": "assistant", "content": "An orange square."},
                ])], type=pa.large_string()),
                "image_bytes": pa.array([encoded.getvalue()], type=pa.large_binary()),
            })
            path = Path(directory) / "i2t.parquet"
            pq.write_table(table, path)
            tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")
            dataset = OmniDataset(
                str(path), tokenizer, audio_processor=None,
                vision_processor=FakeVisionProcessor(), max_length=384,
                scheduled_sampling=0,
            )

            input_ids, labels, _, audio_inputs, _, pixels, _ = dataset[0]

        self.assertEqual(tuple(input_ids.shape), (9, 383))
        self.assertGreater((labels != -100).sum().item(), 0)
        self.assertIsNone(audio_inputs)
        self.assertEqual(tuple(pixels["pixel_values"].shape), (3, 32, 32))
        self.assertTrue(pixels["image_mask"].item())
        self.assertTrue(pixels["pixel_values"].is_contiguous())
        self.assertEqual((input_ids[-1] == dataset.image_token_id).sum().item(), 64)

    def test_long_visual_prompt_keeps_full_image_blocks_and_answer_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (12, 8), "orange")
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            table = pa.table({
                "conversations": pa.array([json.dumps([
                    {"role": "user", "content": "Describe every detail. " * 320 + "<image>"},
                    {"role": "assistant", "content": "An orange cat is sitting beside a book."},
                ])], type=pa.large_string()),
                "image_bytes": pa.array([encoded.getvalue()], type=pa.large_binary()),
            })
            path = Path(directory) / "long_i2t.parquet"
            pq.write_table(table, path)
            tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")
            dataset = OmniDataset(
                str(path), tokenizer, vision_processor=FakeVisionProcessor(), max_length=384,
                scheduled_sampling=0,
            )

            with patch("dataset.omni_dataset.random.random", return_value=0.99):
                input_ids, labels, _, _, _, pixels, _ = dataset[0]

        self.assertEqual(tuple(input_ids.shape), (9, 383))
        self.assertEqual((input_ids[-1] == dataset.image_token_id).sum().item(), 64)
        self.assertGreater((labels != -100).sum().item(), 0)
        self.assertTrue(pixels["pixel_values"].is_contiguous())

    def test_image_bytes_without_image_marker_do_not_create_visual_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            table = pa.table({
                "conversations": pa.array([json.dumps([
                    {"role": "user", "content": "Describe this."},
                    {"role": "assistant", "content": "A simple reply."},
                ])], type=pa.large_string()),
                "image_bytes": pa.array([b"unused placeholder"], type=pa.large_binary()),
            })
            path = Path(directory) / "text_only.parquet"
            pq.write_table(table, path)
            tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")
            processor = FakeVisionProcessor()
            dataset = OmniDataset(
                str(path), tokenizer, vision_processor=processor, max_length=128,
                scheduled_sampling=0,
            )
            _, _, _, _, _, pixels, _ = dataset[0]

        self.assertEqual(tuple(pixels["pixel_values"].shape), (3, 32, 32))
        self.assertFalse(pixels["image_mask"].item())
        self.assertEqual(pixels["pixel_values"].count_nonzero().item(), 0)

    def test_image_marker_requires_image_bytes_and_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_bytes = self.make_dataset(directory, "<image> describe", None, FakeVisionProcessor())
            with self.assertRaisesRegex(ValueError, "requires image_bytes"):
                missing_bytes[0]

            missing_encoder = self.make_dataset(directory, "<image> describe", b"image", None)
            with self.assertRaisesRegex(ValueError, "requires image_bytes"):
                missing_encoder[0]

    def test_training_rejects_multiple_images_in_one_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(directory, "<image> then <image>", b"image", FakeVisionProcessor())
            with self.assertRaisesRegex(ValueError, "exactly one image"):
                dataset[0]

    def test_training_collate_batches_one_image_per_sample(self):
        def sample(value):
            return (
                torch.zeros(9, 3, dtype=torch.long),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(8, 3, dtype=torch.long),
                None,
                0,
                {"pixel_values": torch.full((3, 2, 2), float(value))},
                torch.zeros(192),
            )

        batch = omni_collate_fn([sample(1), sample(2)])
        pixel_values = batch[5]
        self.assertEqual(tuple(pixel_values['pixel_values'].shape), (2, 3, 2, 2))
        self.assertEqual(pixel_values['image_mask'].tolist(), [True, True])
        torch.testing.assert_close(pixel_values['pixel_values'][:, 0, 0, 0], torch.tensor([1.0, 2.0]))

    def test_collate_fills_missing_images_with_zeros_without_reordering(self):
        def sample(image):
            return (
                torch.zeros(9, 3, dtype=torch.long),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(8, 3, dtype=torch.long),
                None,
                0,
                image,
                torch.zeros(192),
            )

        batch = omni_collate_fn([sample({"pixel_values": torch.ones(3, 2, 2)}), sample(None)])
        pixels = batch[5]
        self.assertEqual(pixels['image_mask'].tolist(), [True, False])
        pixels = pixels['pixel_values']
        self.assertEqual(tuple(pixels.shape), (2, 3, 2, 2))
        self.assertTrue(pixels[0].all())
        self.assertEqual(pixels[1].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
