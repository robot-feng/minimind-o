import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from dataset.image import format_visual_prompt, prepare_image_inputs
from model.model_omni import (
    MiniMindOmni,
    OmniConfig,
    TIPSV2_MODEL_ID,
    TIPSv2ImageProcessor,
    pool_patch_tokens,
)


class FakeTIPSv2:
    def __init__(self):
        self.encoded_images = 0

    def encode_image(self, pixels):
        batch = pixels.size(0)
        self.encoded_images += batch
        signal = pixels.mean(dim=(1, 2, 3), keepdim=True).reshape(batch, 1, 1)
        patches = torch.arange(16 * 3, device=pixels.device, dtype=pixels.dtype).reshape(1, 16, 3)
        return SimpleNamespace(patch_tokens=patches.expand(batch, -1, -1) + signal)


def make_model(image_token_len=4):
    model = object.__new__(MiniMindOmni)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(image_token_len=image_token_len, image_ids=[99], hidden_size=3)
    model.vision_proj = nn.Identity()
    object.__setattr__(model, "vision_encoder", FakeTIPSv2())
    return model


class TestTIPSv2ImageProcessing(unittest.TestCase):
    def test_processor_resizes_rgb_and_keeps_zero_to_one_range(self):
        image = Image.new("RGB", (19, 11), (255, 128, 0))
        pixels = TIPSv2ImageProcessor()(images=image, return_tensors="pt")["pixel_values"]
        self.assertEqual(tuple(pixels.shape), (1, 3, 448, 448))
        self.assertGreaterEqual(pixels.min().item(), 0.0)
        self.assertLessEqual(pixels.max().item(), 1.0)
        self.assertAlmostEqual(pixels[0, 0].mean().item(), 1.0, places=5)

    def test_processor_batches_images_in_input_order(self):
        processor = TIPSv2ImageProcessor()
        pixels = processor(images=[Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")])['pixel_values']
        self.assertEqual(tuple(pixels.shape), (2, 3, 448, 448))
        self.assertGreater(pixels[0, 0].mean(), pixels[1, 0].mean())

    def test_image_preparation_emits_one_image_and_one_marker_block(self):
        config = SimpleNamespace(image_special_token="<image>", image_token_len=4)
        pixels, prompt = prepare_image_inputs(
            Image.new("RGB", (12, 8), "orange"), TIPSv2ImageProcessor(), config,
        )
        self.assertEqual(tuple(pixels['pixel_values'].shape), (1, 3, 448, 448))
        self.assertEqual(prompt.count("<image>"), 4)
        self.assertNotIn("Frame", prompt)

    def test_visual_prompt_places_single_image_before_text(self):
        self.assertEqual(format_visual_prompt(" <image> ", " describe this "), "<image>\n\ndescribe this")
        self.assertEqual(format_visual_prompt("<image>", ""), "<image>")

    def test_pooling_preserves_spatial_grid_layout_and_shape(self):
        tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
        pooled = pool_patch_tokens(tokens, 4)
        self.assertEqual(tuple(pooled.shape), (1, 4, 1))
        torch.testing.assert_close(pooled.flatten(), torch.tensor([2.5, 4.5, 10.5, 12.5]))

    def test_pooling_handles_non_square_patch_count(self):
        tokens = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        self.assertEqual(tuple(pool_patch_tokens(tokens, 5).shape), (1, 5, 1))

    def test_pooling_rejects_invalid_rank_or_target_length(self):
        with self.assertRaises(ValueError):
            pool_patch_tokens(torch.zeros(1, 4), 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            pool_patch_tokens(torch.zeros(1, 4, 2), 0)

    def test_checkpoint_loader_discards_text_tower_and_freezes_vision(self):
        encoder = nn.Module()
        encoder.vision = nn.Linear(2, 2)
        encoder.text_encoder = nn.Linear(2, 2)
        with patch.dict(os.environ, {"MINIMIND_HF_ENDPOINT": "https://huggingface.co"}), \
                patch("huggingface_hub.snapshot_download", return_value="/cached/tips") as download, \
                patch("model.model_omni.AutoModel.from_pretrained", return_value=encoder) as load:
            loaded, processor = MiniMindOmni.load_vision(TIPSV2_MODEL_ID)

        download.assert_called_once_with(TIPSV2_MODEL_ID, endpoint="https://huggingface.co")
        load.assert_called_once_with("/cached/tips", trust_remote_code=True, local_files_only=True)
        self.assertIsNone(loaded.text_encoder)
        self.assertIsInstance(processor, TIPSv2ImageProcessor)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        self.assertFalse(loaded.training)


class TestSingleImageModelPath(unittest.TestCase):
    def test_encoder_returns_one_pooled_block_per_image(self):
        model = make_model(image_token_len=4)
        encoded = model.get_image_embeddings(torch.ones(2, 3, 8, 8))
        self.assertEqual(tuple(encoded.shape), (2, 4, 3))
        self.assertEqual(model.vision_encoder.encoded_images, 2)

    def test_encoder_rejects_video_frame_axis(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, "batch, channels, height, width"):
            model.get_image_embeddings(torch.ones(1, 4, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, "batch, channels, height, width"):
            model.encode_image_inputs(torch.ones(1, 4, 3, 8, 8))

    def test_image_mask_skips_missing_images(self):
        model = make_model()
        output = model.encode_image_inputs({
            'pixel_values': torch.zeros(2, 3, 8, 8),
            'image_mask': torch.tensor([False, False]),
        })
        self.assertEqual(tuple(output.shape), (2, 4, 3))
        self.assertEqual(model.vision_encoder.encoded_images, 0)

    def test_black_image_is_not_mistaken_for_a_missing_image(self):
        model = make_model()
        output = model.encode_image_inputs({
            'pixel_values': torch.zeros(1, 3, 8, 8),
            'image_mask': torch.tensor([True]),
        })
        self.assertEqual(tuple(output.shape), (1, 4, 3))
        self.assertEqual(model.vision_encoder.encoded_images, 1)

    def test_encoder_features_are_cast_to_projector_dtype(self):
        model = make_model()
        model.vision_proj = nn.Linear(3, 3, bias=False).half()
        output = model.encode_image_inputs(torch.ones(1, 3, 8, 8))
        self.assertEqual(output.dtype, torch.float16)

    def test_single_image_features_replace_exactly_one_marker_block(self):
        model = make_model(image_token_len=2)
        tokens = torch.tensor([[1, 99, 99, 7]])
        hidden = torch.zeros(1, 4, 3)
        features = torch.ones(1, 2, 3)
        output = model.count_vision_proj(tokens, hidden, features, seqlen=4)
        torch.testing.assert_close(output[0, 0], hidden[0, 0])
        torch.testing.assert_close(output[0, 1:3], features[0])
        torch.testing.assert_close(output[0, 3], hidden[0, 3])

    def test_non_image_sample_keeps_hidden_states(self):
        model = make_model()
        tokens = torch.tensor([[1, 7, 8]])
        hidden = torch.randn(1, 3, 3)
        result = model.count_vision_proj(tokens, hidden, torch.zeros(1, 4, 3), seqlen=3)
        torch.testing.assert_close(result, hidden)

    def test_marker_count_and_multiple_images_are_rejected(self):
        model = make_model(image_token_len=2)
        hidden = torch.zeros(1, 8, 3)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            model.count_vision_proj(torch.tensor([[99, 99, 99]]), hidden[:, :3], torch.zeros(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "one image-token block"):
            model.count_vision_proj(torch.tensor([[99, 99, 1, 99, 99]]), hidden[:, :5], torch.zeros(1, 2, 3))

    def test_projector_receives_finite_gradient_from_image_supervision(self):
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
        model.vision_proj.weight.requires_grad_(True)
        image_marker = config.image_ids[0]
        input_ids = torch.tensor([[1] + [image_marker] * config.image_token_len + [7, 8]])
        result = model(input_ids, pixel_values={"pixel_values": torch.ones(1, 3, 8, 8)}, text_only=True)
        loss = result.logits[:, -1].square().mean()
        loss.backward()

        gradient = model.vision_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)
        self.assertEqual(tuple(result.logits.shape[:2]), tuple(input_ids.shape))


@unittest.skipUnless(
    os.environ.get("MINIMIND_RUN_TIPSV2_INTEGRATION") == "1",
    "set MINIMIND_RUN_TIPSV2_INTEGRATION=1 to load the real TIPSv2 checkpoint",
)
class TestTIPSv2Checkpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("MINIMIND_TIPSV2_MODEL", TIPSV2_MODEL_ID)
        cls.encoder, cls.processor = MiniMindOmni.load_vision(path)
        cls.model = MiniMindOmni(
            OmniConfig(
                hidden_size=24, num_hidden_layers=1, vocab_size=128,
                num_attention_heads=3, num_key_value_heads=1, intermediate_size=48,
                talker_hidden_size=32, num_talker_hidden_layers=1,
                image_hidden_size=768, image_token_len=64, max_position_embeddings=128,
            ),
            audio_encoder_path=None,
            vision_model_path=None,
        ).eval()
        object.__setattr__(cls.model, "vision_encoder", cls.encoder)

    def test_real_checkpoint_emits_one_64_token_image_block(self):
        pixels = self.processor(images=Image.new("RGB", (64, 48), "orange"), return_tensors="pt")["pixel_values"]
        features = self.model.get_image_embeddings(pixels)
        self.assertEqual(tuple(features.shape), (1, 64, 768))
        self.assertTrue(torch.isfinite(features).all())

    def test_image_changes_visual_features(self):
        first = self.processor(images=Image.new("RGB", (64, 48), "orange"), return_tensors="pt")["pixel_values"]
        second = self.processor(images=Image.new("RGB", (64, 48), "blue"), return_tensors="pt")["pixel_values"]
        first_features = self.model.get_image_embeddings(first)
        second_features = self.model.get_image_embeddings(second)
        self.assertGreater((first_features - second_features).abs().max().item(), 1e-5)

    def test_real_vision_path_reaches_the_thinker(self):
        pixels = self.processor(images=Image.new("RGB", (64, 48), "orange"), return_tensors="pt")["pixel_values"]
        image_ids = self.model.config.image_ids
        input_ids = torch.tensor([[1] + [image_ids[0]] * self.model.config.image_token_len + [7]])
        with torch.inference_mode():
            output = self.model(input_ids, pixel_values={"pixel_values": pixels}, text_only=True)
        self.assertEqual(tuple(output.logits.shape[:2]), tuple(input_ids.shape))
        self.assertTrue(torch.isfinite(output.logits).all())


if __name__ == "__main__":
    unittest.main()
