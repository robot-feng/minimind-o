import json
import os
import tempfile
import unittest

import torch

from dataset.alignment_dataset import PreferenceDataset
from model.model_omni import MiniMindOmni, OmniConfig
from trainer.alignment_utils import dpo_loss, masked_sequence_logps, token_log_probs


class FakeEncoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class FakeTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return FakeEncoding(input_ids=[ord(char) + 3 for char in text])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        return "".join(f"<s>{m['role']}\n{m['content']}</s>\n" for m in messages)


class TestAlignmentLosses(unittest.TestCase):
    def test_token_log_probs_gathers_target_tokens(self):
        logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
        labels = torch.tensor([[1, 0]])
        actual = token_log_probs(logits, labels)
        expected = torch.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected)

    def test_mask_excludes_prompt_and_padding(self):
        logits = torch.zeros(1, 3, 4)
        labels = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[0, 1, 0]])
        scores = masked_sequence_logps(logits, labels, mask)
        torch.testing.assert_close(scores, torch.tensor([-torch.log(torch.tensor(4.0))]))

    def test_dpo_loss_and_accuracy_improve_when_policy_prefers_chosen(self):
        ref = torch.zeros(2)
        bad_loss, bad_accuracy = dpo_loss(ref, ref, ref, ref, beta=0.1)
        good_loss, good_accuracy = dpo_loss(torch.tensor([2.0, 2.0]), torch.tensor([0.0, 0.0]),
                                             ref, ref, beta=0.1)
        self.assertAlmostEqual(bad_loss.item(), 0.693147, places=5)
        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertEqual(bad_accuracy.item(), 0.0)
        self.assertEqual(good_accuracy.item(), 1.0)

    def test_dpo_loss_backpropagates(self):
        chosen = torch.tensor([0.2, 0.5], requires_grad=True)
        rejected = torch.zeros(2, requires_grad=True)
        loss, _ = dpo_loss(chosen, rejected, torch.zeros(2), torch.zeros(2))
        loss.backward()
        self.assertIsNotNone(chosen.grad)
        self.assertTrue(torch.all(chosen.grad < 0))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            token_log_probs(torch.zeros(1, 2, 3), torch.zeros(1, 1, dtype=torch.long))
        with self.assertRaises(ValueError):
            dpo_loss(torch.zeros(2), torch.zeros(1), torch.zeros(2), torch.zeros(2))


class TestPreferenceDataset(unittest.TestCase):
    def test_masks_assistant_messages_and_pads_sequences(self):
        sample = {
            "chosen": [{"role": "user", "content": "question"},
                       {"role": "assistant", "content": "preferred"}],
            "rejected": [{"role": "user", "content": "question"},
                         {"role": "assistant", "content": "other"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dpo.jsonl")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(sample) + "\n")
            dataset = PreferenceDataset(path, FakeTokenizer(), max_length=100)
            row = dataset[0]
        self.assertEqual(row["x_chosen"].shape, (99,))
        self.assertGreater(row["mask_chosen"].sum().item(), 0)
        self.assertLess(row["mask_chosen"].sum().item(), 100)
        self.assertGreater(row["mask_rejected"].sum().item(), 0)
        self.assertTrue(torch.all(row["mask_chosen"] <= 1))

    def test_rejects_rows_without_assistant_response(self):
        sample = {"chosen": [{"role": "user", "content": "q"}],
                  "rejected": [{"role": "user", "content": "q"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dpo.jsonl")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(sample) + "\n")
            dataset = PreferenceDataset(path, FakeTokenizer(), max_length=100)
            with self.assertRaisesRegex(ValueError, "no assistant response"):
                dataset[0]


class TestOmniTextTrainingSetup(unittest.TestCase):
    def test_audio_encoder_can_be_disabled_for_text_only_training(self):
        encoder, processor = MiniMindOmni.load_sensevoice(None)
        self.assertIsNone(encoder)
        self.assertIsNone(processor)

    def test_dpo_loss_backpropagates_through_tiny_omni_text_backbone(self):
        config = OmniConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=2,
            talker_hidden_size=32, num_talker_hidden_layers=1,
            image_hidden_size=8, image_token_len=4, spk_emb_size=4,
            max_position_embeddings=16,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.thinker.parameters():
            parameter.requires_grad_(True)
        for parameter in model.lm_head.parameters():
            parameter.requires_grad_(True)

        input_ids = torch.randint(0, config.vocab_size, (2, 6))
        labels = input_ids.clone()
        outputs = model(input_ids, text_only=True)
        policy_scores = masked_sequence_logps(outputs.logits, labels, torch.ones_like(labels))
        loss, _ = dpo_loss(policy_scores[:1], policy_scores[1:], torch.zeros(1), torch.zeros(1))
        loss.backward()

        text_grads = [p.grad for p in model.thinker.parameters() if p.requires_grad]
        frozen_grads = [p.grad for p in model.parameters() if not p.requires_grad]
        self.assertTrue(any(grad is not None for grad in text_grads))
        self.assertTrue(all(grad is None for grad in frozen_grads))
        self.assertIsNone(outputs.audio_logits)


if __name__ == "__main__":
    unittest.main()
