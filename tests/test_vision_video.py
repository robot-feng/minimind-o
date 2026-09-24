import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image

from dataset.video import prepare_video_inputs, sample_video_frames
from model.model_omni import (
    MiniMindOmni,
    OmniCausalLMOutputWithPast,
    TIPSv2ImageProcessor,
    pool_patch_tokens,
)


class FakeCapture:
    def __init__(self, path):
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        if key == 7:  # cv2.CAP_PROP_FRAME_COUNT
            return 7
        if key == 5:  # cv2.CAP_PROP_FPS
            return 2
        return 0

    def set(self, key, value):
        self.index = int(value)
        return True

    def read(self):
        return True, np.full((8, 8, 3), self.index, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeTIPSv2:
    def encode_image(self, pixel_values):
        batch = pixel_values.size(0)
        tokens = torch.arange(batch * 16 * 3, device=pixel_values.device, dtype=torch.float32)
        return SimpleNamespace(patch_tokens=tokens.reshape(batch, 16, 3))


def make_model(image_token_len=4):
    model = object.__new__(MiniMindOmni)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(image_token_len=image_token_len, image_ids=[99], hidden_size=3)
    object.__setattr__(model, "vision_encoder", FakeTIPSv2())
    return model


class TestTIPSv2ImageProcessing(unittest.TestCase):
    def test_processor_resizes_rgb_without_normalizing(self):
        image = Image.new("RGB", (19, 11), (255, 128, 0))
        result = TIPSv2ImageProcessor()(images=image, return_tensors="pt")
        pixels = result["pixel_values"]
        self.assertEqual(tuple(pixels.shape), (1, 3, 448, 448))
        self.assertGreaterEqual(pixels.min().item(), 0.0)
        self.assertLessEqual(pixels.max().item(), 1.0)
        self.assertAlmostEqual(pixels[0, 0].mean().item(), 1.0, places=5)

    def test_pooling_preserves_grid_layout_and_shape(self):
        tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
        pooled = pool_patch_tokens(tokens, 4)
        self.assertEqual(tuple(pooled.shape), (1, 4, 1))
        torch.testing.assert_close(pooled.flatten(), torch.tensor([2.5, 4.5, 10.5, 12.5]))

    def test_pooling_handles_non_square_token_counts(self):
        tokens = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        self.assertEqual(tuple(pool_patch_tokens(tokens, 5).shape), (1, 5, 1))

    def test_pooling_rejects_invalid_rank(self):
        with self.assertRaises(ValueError):
            pool_patch_tokens(torch.zeros(1, 4), 2)

    def test_model_output_declares_audio_logits_for_compiled_forward(self):
        logits = torch.zeros(1, 2, 5)
        audio_logits = [torch.zeros(1, 2, 7)]
        output = OmniCausalLMOutputWithPast(logits=logits, audio_logits=audio_logits)
        self.assertIs(output.logits, logits)
        self.assertIs(output.audio_logits, audio_logits)


class TestVideoInput(unittest.TestCase):
    def test_uniform_sampling_keeps_frame_order_and_timestamps(self):
        capture = FakeCapture("sample.avi")
        with patch("dataset.video.cv2.VideoCapture", return_value=capture):
            frames = sample_video_frames("sample.avi", num_frames=3)
        self.assertEqual([timestamp for _, timestamp in frames], [0.0, 1.5, 3.0])
        self.assertEqual([int(np.asarray(image)[0, 0, 0]) for image, _ in frames], [0, 3, 6])
        self.assertTrue(capture.released)

    def test_invalid_frame_count_is_rejected_and_capture_released(self):
        capture = FakeCapture("sample.avi")
        capture.get = lambda key: 0
        with patch("dataset.video.cv2.VideoCapture", return_value=capture):
            with self.assertRaises(ValueError):
                sample_video_frames("sample.avi", num_frames=1)
        self.assertTrue(capture.released)

    def test_video_preparation_shapes_frames_and_adds_frame_markers(self):
        config = SimpleNamespace(image_special_token="<image>", image_token_len=4)
        capture = FakeCapture("sample.avi")
        with patch("dataset.video.cv2.VideoCapture", return_value=capture):
            pixels, prompt = prepare_video_inputs(
                "sample.avi", TIPSv2ImageProcessor(), config, device="cpu", num_frames=3
            )
        self.assertEqual(tuple(pixels["pixel_values"].shape), (1, 3, 3, 448, 448))
        self.assertEqual(prompt.count("<image>"), 12)
        self.assertIn("Frame 1 at 0.00s", prompt)
        self.assertIn("Frame 2 at 1.50s", prompt)
        self.assertIn("Frame 3 at 3.00s", prompt)

    def test_sampler_decodes_a_real_video_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
            self.assertTrue(writer.isOpened())
            for frame in range(5):
                writer.write(np.full((32, 32, 3), frame * 30, dtype=np.uint8))
            writer.release()

            frames = sample_video_frames(path, num_frames=3)
        self.assertEqual(len(frames), 3)
        self.assertEqual([timestamp for _, timestamp in frames], [0.0, 0.4, 0.8])
        self.assertEqual(frames[0][0].size, (32, 32))

    def test_video_markers_route_each_frame_to_its_own_token_block(self):
        model = make_model(image_token_len=4)
        tokens = torch.tensor([[1, 99, 99, 99, 99, 7, 99, 99, 99, 99]])
        hidden = torch.zeros(1, 10, 3)
        frames = torch.stack((torch.ones(4, 3), torch.full((4, 3), 2.0))).unsqueeze(0)
        result = model.count_vision_proj(tokens, hidden, frames, seqlen=10)
        torch.testing.assert_close(result[0, 1:5], torch.ones(4, 3))
        torch.testing.assert_close(result[0, 6:10], torch.full((4, 3), 2.0))

    def test_image_embedding_path_accepts_batch_and_frame_dimensions(self):
        model = make_model(image_token_len=4)
        one_image = model.get_image_embeddings({"pixel_values": torch.zeros(2, 3, 8, 8)})
        self.assertEqual(tuple(one_image.shape), (2, 4, 3))
        video = model.get_image_embeddings({"pixel_values": torch.zeros(2, 3, 3, 8, 8)})
        self.assertEqual(tuple(video.shape), (2, 3, 4, 3))


@unittest.skipUnless(os.environ.get("MINIMIND_RUN_TIPSV2_INTEGRATION") == "1",
                     "set MINIMIND_RUN_TIPSV2_INTEGRATION=1 to load the real TIPSv2 checkpoint")
class TestTIPSv2Checkpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder, cls.processor = MiniMindOmni.load_vision(
            os.environ.get("MINIMIND_TIPSV2_MODEL", "google/tipsv2-b14")
        )

    def test_checkpoint_encodes_and_pools_real_image_features(self):
        image = Image.new("RGB", (80, 52), (50, 140, 210))
        pixels = self.processor(images=image)["pixel_values"]
        with torch.inference_mode():
            features = self.encoder.encode_image(pixels).patch_tokens
        self.assertEqual(tuple(features.shape), (1, 1024, 768))
        self.assertEqual(tuple(pool_patch_tokens(features, 64).shape), (1, 64, 768))
        self.assertIsNone(self.encoder.text_encoder)

    def test_real_checkpoint_accepts_video_frame_batches(self):
        image = Image.new("RGB", (80, 52), (50, 140, 210))
        pixels = self.processor(images=image)["pixel_values"]
        model = make_model(image_token_len=64)
        object.__setattr__(model, "vision_encoder", self.encoder)
        video = pixels.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()
        features = model.get_image_embeddings({"pixel_values": video})
        self.assertEqual(tuple(features.shape), (1, 3, 64, 768))


if __name__ == "__main__":
    unittest.main()
