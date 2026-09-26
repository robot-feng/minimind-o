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
    OmniConfig,
    OmniCausalLMOutputWithPast,
    TIPSV2_MODEL_ID,
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

    def test_model_id_uses_huggingface_default_revision(self):
        encoder = nn.Module()
        encoder.text_encoder = nn.Linear(2, 2)
        with patch.dict(os.environ, {
            "MINIMIND_HF_ENDPOINT": "https://huggingface.co",
            "HF_ENDPOINT": "https://huggingface.co",
        }), patch("huggingface_hub.snapshot_download", return_value="/cached/tips") as download, \
                patch("model.model_omni.AutoModel.from_pretrained", return_value=encoder) as load:
            loaded, processor = MiniMindOmni.load_vision(TIPSV2_MODEL_ID)

        download.assert_called_once_with(TIPSV2_MODEL_ID, endpoint="https://huggingface.co")
        load.assert_called_once_with("/cached/tips", trust_remote_code=True, local_files_only=True)
        self.assertIsNone(loaded.text_encoder)
        self.assertIsInstance(processor, TIPSv2ImageProcessor)


class TestVideoInput(unittest.TestCase):
    def test_vision_projector_receives_gradients_without_talker(self):
        config = OmniConfig(
            hidden_size=12, num_hidden_layers=1, vocab_size=128,
            num_attention_heads=3, num_key_value_heads=1, intermediate_size=24,
            talker_hidden_size=16, num_talker_hidden_layers=1,
            image_hidden_size=3, image_token_len=4, max_position_embeddings=64,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None)
        object.__setattr__(model, "vision_encoder", FakeTIPSv2())
        model.vision_proj = nn.Linear(3, config.hidden_size, bias=False)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.vision_proj.parameters():
            parameter.requires_grad = True

        image_marker = config.image_ids[0]
        input_ids = torch.tensor([[1, image_marker, image_marker, image_marker, image_marker, 7]])
        result = model(
            input_ids,
            pixel_values={"pixel_values": torch.ones(1, 3, 8, 8)},
            text_only=True,
        )
        result.logits[..., 3].sum().backward()

        self.assertIsNotNone(model.vision_proj.weight.grad)
        self.assertTrue(all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not name.startswith("vision_proj.")
        ))

    def test_uniform_sampling_keeps_frame_order_and_timestamps(self):
        capture = FakeCapture("sample.avi")
        with patch("dataset.video.cv2.VideoCapture", return_value=capture):
            frames = sample_video_frames("sample.avi", num_frames=3)
        self.assertEqual([timestamp for _, timestamp in frames], [0.0, 1.5, 3.0])
        self.assertEqual([int(np.asarray(image)[0, 0, 0]) for image, _ in frames], [0, 3, 6])
        self.assertTrue(capture.released)

    def test_sampling_more_frames_than_video_returns_each_frame_once(self):
        capture = FakeCapture("short.avi")
        with patch("dataset.video.cv2.VideoCapture", return_value=capture):
            frames = sample_video_frames("short.avi", num_frames=10)
        self.assertEqual(len(frames), 7)
        self.assertEqual([int(np.asarray(image)[0, 0, 0]) for image, _ in frames], list(range(7)))
        self.assertEqual([timestamp for _, timestamp in frames], [i / 2 for i in range(7)])
        self.assertTrue(capture.released)

    def test_unopenable_video_is_rejected_and_capture_released(self):
        capture = FakeCapture("missing.avi")
        capture.isOpened = lambda: False
        with patch("dataset.video.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "Cannot open video"):
                sample_video_frames("missing.avi", num_frames=2)
        self.assertTrue(capture.released)

    def test_nonpositive_sample_count_is_rejected_before_opening_video(self):
        with patch("dataset.video.cv2.VideoCapture") as video_capture:
            with self.assertRaisesRegex(ValueError, "num_frames must be positive"):
                sample_video_frames("sample.avi", num_frames=0)
        video_capture.assert_not_called()

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

    def test_raw_video_tensor_keeps_frame_axis_for_single_and_multi_video_batches(self):
        config = OmniConfig(
            hidden_size=12, num_hidden_layers=1, vocab_size=128,
            num_attention_heads=3, num_key_value_heads=1, intermediate_size=24,
            talker_hidden_size=16, num_talker_hidden_layers=1,
            image_hidden_size=3, image_token_len=4, max_position_embeddings=64,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None).eval()
        object.__setattr__(model, "vision_encoder", FakeTIPSv2())
        model.vision_proj = nn.Linear(3, config.hidden_size, bias=False)
        input_ids = torch.tensor([[1] + [config.image_ids[0]] * 4 + [7] + [config.image_ids[0]] * 4])
        video = torch.stack((torch.ones(3, 8, 8), torch.full((3, 8, 8), 2.0))).unsqueeze(0)

        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size), torch.inference_mode():
                output = model(
                    input_ids.expand(batch_size, -1),
                    pixel_values=video.expand(batch_size, -1, -1, -1, -1),
                    text_only=True,
                )
            self.assertEqual(
                tuple(output.logits.shape),
                (batch_size, input_ids.size(1), config.vocab_size),
            )

    def test_text_generation_accepts_image_inputs_without_talker(self):
        config = OmniConfig(
            hidden_size=12, num_hidden_layers=1, vocab_size=128,
            num_attention_heads=3, num_key_value_heads=1, intermediate_size=24,
            talker_hidden_size=16, num_talker_hidden_layers=1,
            image_hidden_size=3, image_token_len=4, max_position_embeddings=64,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None).eval()
        object.__setattr__(model, "vision_encoder", FakeTIPSv2())
        model.vision_proj = nn.Linear(3, config.hidden_size, bias=False)
        input_ids = torch.tensor([[1] + [config.image_ids[0]] * 4 + [7]])
        pixels = {"pixel_values": torch.ones(1, 3, 8, 8)}

        with patch.object(model, "get_image_embeddings", wraps=model.get_image_embeddings) as encode:
            output = model.generate_text(
                input_ids, max_new_tokens=2, temperature=0, pixel_values=pixels
            )

        self.assertEqual(tuple(output.shape[:1]), (1,))
        self.assertGreater(output.size(1), input_ids.size(1))
        encode.assert_called_once()
        self.assertEqual(tuple(encode.call_args.args[0]["pixel_values"].shape), (1, 3, 8, 8))

    def test_text_generation_accepts_video_frame_inputs(self):
        config = OmniConfig(
            hidden_size=12, num_hidden_layers=1, vocab_size=128,
            num_attention_heads=3, num_key_value_heads=1, intermediate_size=24,
            talker_hidden_size=16, num_talker_hidden_layers=1,
            image_hidden_size=3, image_token_len=4, max_position_embeddings=64,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None).eval()
        object.__setattr__(model, "vision_encoder", FakeTIPSv2())
        model.vision_proj = nn.Linear(3, config.hidden_size, bias=False)
        input_ids = torch.tensor([[1] + [config.image_ids[0]] * 4 + [7]
                                  + [config.image_ids[0]] * 4 + [8]])
        pixels = {"pixel_values": torch.ones(1, 2, 3, 8, 8)}

        with patch.object(model, "get_image_embeddings", wraps=model.get_image_embeddings) as encode:
            output = model.generate_text(
                input_ids, max_new_tokens=2, temperature=0, pixel_values=pixels
            )

        self.assertEqual(tuple(output.shape[:1]), (1,))
        self.assertGreater(output.size(1), input_ids.size(1))
        encode.assert_called_once()
        self.assertEqual(tuple(encode.call_args.args[0]["pixel_values"].shape), (1, 2, 3, 8, 8))


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

    def test_real_checkpoint_conditions_thinker_logits_on_image_content(self):
        config = OmniConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
            talker_hidden_size=32, num_talker_hidden_layers=1,
            image_hidden_size=768, image_token_len=64,
            max_position_embeddings=256,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None).eval()
        object.__setattr__(model, "vision_encoder", self.encoder)
        images = [Image.new("RGB", (80, 52), color) for color in ((50, 140, 210), (210, 70, 40))]
        pixels = torch.cat([self.processor(images=image)["pixel_values"] for image in images])
        input_ids = torch.tensor([[1] + [config.image_ids[0]] * config.image_token_len + [3]]).expand(2, -1)

        with torch.inference_mode():
            image_features = model.get_image_embeddings({"pixel_values": pixels})
            image_logits = model(
                input_ids, pixel_values={"pixel_values": pixels}, text_only=True, logits_to_keep=1
            ).logits[:, -1]
            no_image_logits = model(input_ids, text_only=True, logits_to_keep=1).logits[:, -1]

        self.assertFalse(torch.allclose(image_features[0], image_features[1]))
        self.assertFalse(torch.allclose(image_logits[0], image_logits[1]))
        self.assertFalse(torch.allclose(image_logits[0], no_image_logits[0]))

    def test_real_checkpoint_reaches_thinker_for_image_and_video_inputs(self):
        config = OmniConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
            talker_hidden_size=32, num_talker_hidden_layers=1,
            image_hidden_size=768, image_token_len=64,
            max_position_embeddings=256,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None).eval()
        object.__setattr__(model, "vision_encoder", self.encoder)
        image = Image.new("RGB", (80, 52), (50, 140, 210))
        pixels = self.processor(images=image)["pixel_values"]
        image_ids = torch.tensor([[1] + [config.image_ids[0]] * config.image_token_len + [2]])
        video = pixels.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()
        video_ids = torch.tensor(
            [[1] + ([config.image_ids[0]] * config.image_token_len + [3]) * 2
             + [config.image_ids[0]] * config.image_token_len]
        )

        for input_ids, pixel_values in (
            (image_ids, {"pixel_values": pixels}),
            (video_ids, {"pixel_values": video}),
        ):
            with torch.inference_mode():
                output = model(input_ids, pixel_values=pixel_values, text_only=True)
                text_generated = model.generate_text(
                    input_ids, eos_token_id=2, max_new_tokens=1, temperature=0,
                    pixel_values=pixel_values,
                )
                generated = list(model.generate(
                    input_ids, eos_token_id=2, max_new_tokens=1,
                    temperature=0.8, top_p=1.0, stream=True,
                    pixel_values=pixel_values,
                ))
            self.assertEqual(output.logits.shape[:2], input_ids.shape)
            self.assertEqual(tuple(text_generated.shape), (1, input_ids.size(1) + 1))
            self.assertEqual(len(generated), 1)
            self.assertEqual(tuple(generated[0][0].shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
