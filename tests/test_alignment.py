import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dataset.alignment_dataset import PreferenceDataset
from dataset.text_dataset import TextSFTDataset
from model.lora import LoRALinear, inject_lora, lora_state_dict, merge_lora
from model.model_omni import MiniMindOmni, OmniConfig
from trainer.alignment_utils import dpo_loss, masked_sequence_logps, token_log_probs
from trainer.agent_tools import calculate_agent_reward, execute_tool, parse_tool_calls, safe_math_eval
from trainer.rl_utils import grpo_cispo_loss, group_relative_advantages, score_responses
from trainer.rollout_engine import RolloutResult, SGLangRolloutEngine, TorchRolloutEngine
from trainer.training_losses import causal_lm_loss, distillation_objective, masked_distillation_loss
from trainer.ppo_utils import clipped_value_loss, generalized_advantage_estimate, ppo_policy_loss
from trainer.train_ppo import PPOValueModel, _trainable_value, train_batch
from trainer.train_agent import collect_agent_rollouts
from trainer.train_tokenizer import get_texts, train_tokenizer


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

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token) for token in ids)


class TestRollouts(unittest.TestCase):
    def test_batched_text_generation_and_rollout_scores(self):
        config = OmniConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=2,
            talker_hidden_size=32, num_talker_hidden_layers=1,
            image_hidden_size=8, image_token_len=4, spk_emb_size=4,
            max_position_embeddings=16,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None)
        tokenizer = FakeTokenizer()
        prompts = torch.tensor([[1, 3, 4, 5], [0, 0, 6, 7]])
        prompt_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        before_mode = model.training
        output = model.generate_text(prompts, attention_mask=prompt_mask,
                                     max_new_tokens=2, temperature=0,
                                     eos_token_id=2, pad_token_id=0)
        self.assertEqual(tuple(output.shape), (2, 6))
        self.assertTrue(torch.equal(output[:, :4], prompts))
        self.assertEqual(model.training, before_mode)

        engine = TorchRolloutEngine(model, tokenizer, temperature=0, top_p=1)
        result = engine.rollout(prompts, prompt_mask, num_generations=2, max_new_tokens=2)
        self.assertEqual(tuple(result.output_ids.shape), (4, 6))
        self.assertEqual(tuple(result.per_token_logps.shape), (4, 2))
        self.assertEqual(tuple(result.completion_mask.shape), (4, 2))
        self.assertEqual(len(result.completions), 4)
        self.assertFalse(result.output_ids.is_inference())

    def test_sglang_result_is_padded_and_keeps_token_logprobs(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return [{"meta_info": {"output_ids": [9, 2],
                                        "output_token_logprobs": [[-0.4, 9], [-0.1, 2]]}}]

        with unittest.mock.patch("requests.post", return_value=Response()) as post:
            engine = SGLangRolloutEngine(FakeTokenizer(), "http://localhost:8998", "/tmp/shared")
            prompt_ids = torch.tensor([[1, 3, 4]])
            attention_mask = torch.ones_like(prompt_ids)
            result = engine.rollout(prompt_ids, attention_mask, num_generations=1,
                                    max_new_tokens=2, temperature=0.7)
        self.assertEqual(tuple(result.output_ids.shape), (1, 5))
        self.assertEqual(result.completion_mask.sum().item(), 2)
        torch.testing.assert_close(result.per_token_logps, torch.tensor([[-0.4, -0.1]]))
        self.assertIn("input_ids", post.call_args.kwargs["json"])


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

    def test_ppo_value_head_reads_text_hidden_states_only(self):
        config = OmniConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=2,
            talker_hidden_size=32, num_talker_hidden_layers=1,
            image_hidden_size=8, image_token_len=4, spk_emb_size=4,
            max_position_embeddings=16,
        )
        critic = PPOValueModel(MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None))
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        values, aux = critic(input_ids, torch.ones_like(input_ids))
        self.assertEqual(tuple(values.shape), (2, 5))
        self.assertEqual(aux.ndim, 0)

    def test_ppo_trains_omni_text_critic_backbone_without_lm_head(self):
        config = OmniConfig(
            hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=2,
            talker_hidden_size=32, num_talker_hidden_layers=1,
            image_hidden_size=8, image_token_len=4, spk_emb_size=4,
            max_position_embeddings=16,
        )
        critic = PPOValueModel(MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None))
        _trainable_value(critic)
        trainable = {name for name, parameter in critic.named_parameters() if parameter.requires_grad}
        self.assertIn("value_head.weight", trainable)
        self.assertTrue(any(name.startswith("backbone.model.layers.") for name in trainable))
        self.assertFalse(any("lm_head" in name for name in trainable))
        self.assertFalse(any(name.startswith("backbone.talker.") for name in trainable))


class TestPPOLosses(unittest.TestCase):
    def test_gae_masks_padding_and_propagates_terminal_reward(self):
        rewards = torch.tensor([[0.0, 1.0, 0.0]])
        values = torch.zeros_like(rewards)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        advantages, returns = generalized_advantage_estimate(rewards, values, mask,
                                                               gamma=1.0, lam=1.0)
        self.assertAlmostEqual(advantages[0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(advantages[0, 1].item(), 0.0, places=6)
        self.assertEqual(advantages[0, 2].item(), 0.0)
        self.assertTrue(torch.equal(returns, torch.tensor([[1.0, 1.0, 0.0]])))

    def test_ppo_clipping_and_value_losses_backpropagate(self):
        new_logps = torch.tensor([[2.0]], requires_grad=True)
        old_logps = torch.zeros_like(new_logps)
        advantage = torch.ones_like(new_logps)
        mask = torch.ones_like(new_logps)
        policy_loss = ppo_policy_loss(new_logps, old_logps, advantage, mask, clip_epsilon=0.2)
        values = torch.tensor([[2.0]], requires_grad=True)
        value_loss = clipped_value_loss(values, torch.zeros_like(values), torch.ones_like(values), mask)
        (policy_loss + value_loss).backward()
        self.assertIsNotNone(new_logps.grad)
        self.assertIsNotNone(values.grad)

    def test_ppo_minibatches_accumulate_and_early_stop(self):
        class Batch:
            def __init__(self, ids):
                self.input_ids = ids
                self.attention_mask = torch.ones_like(ids)

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                self.attention_mask = self.attention_mask.to(device)
                return self

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, prompts, **kwargs):
                return Batch(torch.tensor([[1, 3, 4] for _ in prompts]))

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.head = torch.nn.Linear(8, 16)

            def forward(self, input_ids, logits_to_keep, **kwargs):
                hidden = self.embed(input_ids[:, -logits_to_keep:])
                logits = self.head(hidden)
                return SimpleNamespace(logits=logits, aux_loss=logits.sum() * 0)

        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.head = torch.nn.Linear(8, 1)

            def forward(self, input_ids, attention_mask=None):
                hidden = self.embed(input_ids)
                return self.head(hidden).squeeze(-1), hidden.sum() * 0

        class Rollout:
            def __init__(self, old_logp):
                self.old_logp = old_logp

            def rollout(self, prompt_ids, attention_mask, **kwargs):
                suffix = torch.tensor([[6, 7], [8, 9]], device=prompt_ids.device)
                completion_ids = suffix[:prompt_ids.size(0)]
                output_ids = torch.cat((prompt_ids, completion_ids), dim=1)
                mask = torch.ones_like(completion_ids)
                return RolloutResult(
                    output_ids, completion_ids,
                    torch.full(completion_ids.shape, self.old_logp, device=prompt_ids.device),
                    ["short response"] * prompt_ids.size(0),
                    attention_mask.sum(dim=1), mask,
                )

        def make_args(early_stop_kl, debug=False):
            return SimpleNamespace(
                device="cpu", max_seq_len=8, max_gen_len=2, reward_model=None,
                gamma=0.99, gae_lambda=0.95, ppo_epochs=2, mini_batch_size=1,
                accumulation_steps=2, clip_epsilon=0.2, value_clip=0.2,
                value_coef=0.5, beta=0.02, early_stop_kl=early_stop_kl,
                grad_clip=1.0, debug_mode=debug, debug_interval=1,
                debug_log_ratio=debug,
            )

        model, critic = Policy(), Critic()
        reference = Policy()
        reference.load_state_dict(model.state_dict())
        actor_optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with patch("trainer.train_ppo.Logger") as logger:
            metrics, _ = train_batch(
                model, critic, reference, actor_optimizer, critic_optimizer, scaler,
                nullcontext(), make_args(1000, debug=True), Tokenizer(), Rollout(-2.0),
                ["p1", "p2"], 1,
            )
        debug_output = "\n".join(str(call.args[0]) for call in logger.call_args_list)
        self.assertIn("CONTEXT_BEGIN", debug_output)
        self.assertIn("reward=", debug_output)
        self.assertIn("log_ratio max_abs=", debug_output)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["policy_loss"])))
        self.assertEqual(int(actor_optimizer.state[model.embed.weight]["step"]), 2)
        self.assertEqual(int(critic_optimizer.state[critic.embed.weight]["step"]), 2)

        stopped_model, stopped_critic = Policy(), Critic()
        stopped_actor_optimizer = torch.optim.AdamW(stopped_model.parameters(), lr=0.01)
        stopped_critic_optimizer = torch.optim.AdamW(stopped_critic.parameters(), lr=0.01)
        train_batch(
            stopped_model, stopped_critic, reference, stopped_actor_optimizer,
            stopped_critic_optimizer, scaler, nullcontext(), make_args(0.01),
            Tokenizer(), Rollout(-20.0), ["p1", "p2"], 1,
        )
        self.assertEqual(len(stopped_actor_optimizer.state), 0)
        self.assertEqual(len(stopped_critic_optimizer.state), 0)


class TestTextDatasets(unittest.TestCase):
    def test_sft_dataset_masks_user_and_system_tokens(self):
        sample = {"conversations": [{"role": "user", "content": "unique-question"},
                                     {"role": "assistant", "content": "unique-answer"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sft.jsonl")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(sample) + "\n")
            ids, labels = TextSFTDataset(path, FakeTokenizer(), max_length=128)[0]
        question = torch.tensor([ord(char) + 3 for char in "unique-question"])
        answer = torch.tensor([ord(char) + 3 for char in "unique-answer"])
        question_pos = next(i for i in range(len(ids) - len(question))
                            if torch.equal(ids[i:i + len(question)], question))
        answer_pos = next(i for i in range(len(ids) - len(answer))
                          if torch.equal(ids[i:i + len(answer)], answer))
        self.assertTrue(torch.all(labels[question_pos:question_pos + len(question)] == -100))
        self.assertTrue(torch.equal(labels[answer_pos:answer_pos + len(answer)], answer))


class TestLoRA(unittest.TestCase):
    def test_injection_is_zero_initialized_and_merge_preserves_outputs(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 3, bias=False)
                self.other = torch.nn.Linear(4, 3, bias=False)

        block = Block()
        inputs = torch.randn(2, 4)
        expected = block.q_proj(inputs)
        count = inject_lora(block, target_names=("q_proj",), rank=2, alpha=4)
        self.assertEqual(count, 1)
        self.assertIsInstance(block.q_proj, LoRALinear)
        self.assertEqual(block.q_proj.lora_A.device, block.q_proj.base.weight.device)
        torch.testing.assert_close(block.q_proj(inputs), expected)
        self.assertFalse(block.q_proj.base.weight.requires_grad)
        self.assertTrue(block.q_proj.lora_A.requires_grad)
        self.assertEqual(set(lora_state_dict(block)), {"q_proj.lora_A", "q_proj.lora_B"})
        with torch.no_grad():
            block.q_proj.lora_B.normal_()
        before_merge = block.q_proj(inputs)
        merge_lora(block)
        self.assertIsInstance(block.q_proj, torch.nn.Linear)
        torch.testing.assert_close(block.q_proj(inputs), before_merge)


class TestTextTrainingLosses(unittest.TestCase):
    def test_distillation_scales_moe_aux_loss_with_ce_weight(self):
        ce, kd, aux = torch.tensor(2.0), torch.tensor(4.0), torch.tensor(0.5)
        loss = distillation_objective(ce, kd, aux, alpha=0.25)
        torch.testing.assert_close(loss, torch.tensor(3.625))

    def test_distillation_objective_validates_alpha(self):
        with self.assertRaises(ValueError):
            distillation_objective(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.0), 1.1)

    def test_causal_lm_loss_ignores_masked_targets(self):
        logits = torch.zeros(1, 3, 5)
        labels = torch.tensor([[0, 1, -100]])
        loss = causal_lm_loss(logits, labels)
        torch.testing.assert_close(loss, torch.log(torch.tensor(5.0)))

    def test_distillation_zero_when_student_matches_teacher(self):
        logits = torch.randn(2, 4, 7)
        labels = torch.tensor([[1, 2, -100, -100], [3, 4, 5, -100]])
        loss = masked_distillation_loss(logits, logits.clone(), labels, temperature=2.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_distillation_is_masked_and_differentiable(self):
        student = torch.zeros(1, 3, 4, requires_grad=True)
        teacher = torch.zeros(1, 3, 4)
        teacher[0, 1, 1] = 8.0  # This predicts a masked target after the shift.
        labels = torch.tensor([[-100, 2, -100]])
        loss = masked_distillation_loss(student, teacher, labels)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_rejects_empty_targets_or_nonpositive_temperature(self):
        logits = torch.zeros(1, 2, 3)
        labels = torch.full((1, 2), -100)
        with self.assertRaises(ValueError):
            causal_lm_loss(logits, labels)
        with self.assertRaises(ValueError):
            masked_distillation_loss(logits, logits, labels, temperature=0)


class TestOnlineRLLosses(unittest.TestCase):
    def test_group_advantages_are_normalized_per_prompt(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 9.0, 9.0, 9.0])
        advantages = group_relative_advantages(rewards, num_generations=3)
        self.assertAlmostEqual(advantages[:3].mean().item(), 0.0, places=6)
        self.assertAlmostEqual(advantages[3:].abs().sum().item(), 0.0, places=6)
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.ones(3), num_generations=2)

    def test_grpo_policy_loss_is_zero_at_old_reference_policy(self):
        logps = torch.zeros(2, 3)
        advantages = torch.tensor([-1.0, 1.0])
        mask = torch.ones_like(logps)
        loss = grpo_cispo_loss(logps, logps, logps, advantages, mask)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_cispo_retains_gradient_after_ratio_clipping(self):
        policy = torch.full((1, 2), 3.0, requires_grad=True)
        old = torch.zeros_like(policy)
        ref = torch.zeros_like(policy)
        loss = grpo_cispo_loss(policy, old, ref, torch.ones(1), torch.ones_like(policy),
                               beta=0.0, loss_type="cispo", epsilon_high=1.1)
        loss.backward()
        self.assertTrue(torch.all(policy.grad < 0))

    def test_response_scoring_groups_samples_and_applies_heuristics(self):
        scores = score_responses(["prompt-a", "prompt-b"], ["x", "a " * 20, "b " * 20, "z"])
        self.assertEqual(tuple(scores.shape), (4,))
        self.assertGreater(scores[1].item(), scores[0].item())


class TestAgentTools(unittest.TestCase):
    def test_safe_math_and_tool_dispatch(self):
        self.assertEqual(safe_math_eval("2 + 3 * 4"), 14)
        self.assertEqual(execute_tool("calculate_math", {"expression": "7 * 8"}), {"result": "56"})
        with self.assertRaises(ValueError):
            safe_math_eval("__import__('os').system('false')")
        self.assertIsNone(execute_tool("calculate_math", {"expression": "1 / 0"}))

    def test_tool_calls_parse_and_reward_against_ground_truth(self):
        text = '<tool_call>{"name":"calculate_math","arguments":{"expression":"3+4"}}</tool_call>'
        calls = parse_tool_calls(text)
        self.assertEqual(calls[0]["name"], "calculate_math")
        tools = [{"type": "function", "function": {"name": "calculate_math"}}]
        reward = calculate_agent_reward("The answer is 7.", [text, "The answer is 7."],
                                        tools, ["7"])
        self.assertGreater(reward, 2.0)

    def test_invalid_tool_call_gets_no_ground_truth_credit(self):
        text = '<tool_call>{"name":"missing","arguments":{}}</tool_call>'
        tools = [{"type": "function", "function": {"name": "calculate_math"}}]
        reward = calculate_agent_reward("The answer is 8.", [text], tools, ["7"])
        self.assertLess(reward, 0.0)

    def test_agent_plain_answer_uses_reference_minimum_length(self):
        self.assertEqual(calculate_agent_reward("Short answer.", ["Short answer."], [], []), 0.5)

    def test_agent_reward_penalizes_unbalanced_tool_tags_on_every_turn(self):
        malformed = "This is a reasonable answer.<tool_call>"
        self.assertEqual(calculate_agent_reward(malformed, [malformed], [], []), 0.0)

        call = '<tool_call>{"name":"calculate_math","arguments":{"expression":"6 * 7"}}</tool_call>'
        unbalanced = call + "<tool_call> result 42"
        tools = [{"type": "function", "function": {"name": "calculate_math"}}]
        self.assertAlmostEqual(calculate_agent_reward(unbalanced, [unbalanced], tools, ["42"]), 2.5)

    def test_agent_reward_model_scores_answer_after_thinking(self):
        class RewardModel:
            response = None

            def score(self, prompt, response):
                self.response = response
                return 0.0

        text = "This is a sufficiently detailed chain of thought for the format check.</think>short"
        reward_model = RewardModel()
        calculate_agent_reward(text, [text], [], [], reward_model=reward_model, prompt="question")
        self.assertEqual(reward_model.response, "short")

    def test_agent_rollout_executes_tool_and_resumes_conversation(self):
        class Encoding:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                self.attention_mask = torch.ones_like(input_ids)

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                self.attention_mask = self.attention_mask.to(device)
                return self

        class Tokenizer:
            def __init__(self):
                self.rendered = []

            def apply_chat_template(self, messages, **kwargs):
                text = json.dumps(messages, ensure_ascii=False)
                self.rendered.append(text)
                return text

            def __call__(self, texts, **kwargs):
                rows = [torch.tensor([len(text) % 17 + 1, 2]) for text in texts]
                return Encoding(torch.stack(rows))

        class Engine:
            def __init__(self, completions):
                self.completion_batches = iter(completions)

            def rollout(self, input_ids, attention_mask, **kwargs):
                completions = next(self.completion_batches)
                ids = torch.arange(20, 20 + len(completions)).unsqueeze(1)
                return RolloutResult(
                    output_ids=torch.cat((input_ids, ids), dim=1),
                    completion_ids=ids,
                    per_token_logps=torch.full(ids.shape, -0.25),
                    completions=completions,
                    prompt_lens=attention_mask.sum(dim=1),
                    completion_mask=torch.ones_like(ids),
                )

        tool_call = '<tool_call>{"name":"calculate_math","arguments":{"expression":"6 * 7"}}</tool_call>'
        tokenizer = Tokenizer()
        episodes = collect_agent_rollouts(
            {
                "messages": [[{"role": "user", "content": "What is 6 times 7?"}]],
                "tools": [[{"type": "function", "function": {"name": "calculate_math"}}]],
                "gt": [["42"]],
            },
            Engine([[tool_call, "Other answer"], ["42"]]),
            tokenizer,
            SimpleNamespace(num_generations=2, max_turns=2, max_seq_len=64,
                             max_gen_len=8, device="cpu", thinking_ratio=0.0),
        )

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["turns"], [tool_call, "42"])
        self.assertEqual(episodes[0]["final"], "42")
        self.assertFalse(episodes[0]["unfinished"])
        self.assertEqual(len(episodes[0]["actions"]), 2)
        self.assertEqual(episodes[1]["turns"], ["Other answer"])
        self.assertEqual(len(episodes[1]["actions"]), 1)
        tool_message = json.loads(tokenizer.rendered[-1])[-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(json.loads(tool_message["content"]), {"result": "42"})

    def test_agent_rollout_marks_tool_call_at_turn_limit_unfinished(self):
        class Encoding:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                self.attention_mask = torch.ones_like(input_ids)

            def to(self, device):
                return self

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, texts, **kwargs):
                return Encoding(torch.ones(len(texts), 2, dtype=torch.long))

        class Engine:
            def rollout(self, input_ids, attention_mask, **kwargs):
                completion = '<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>'
                ids = torch.ones(len(input_ids), 1, dtype=torch.long)
                return RolloutResult(input_ids, ids, torch.zeros_like(ids, dtype=torch.float32),
                                     [completion] * len(input_ids), attention_mask.sum(1),
                                     torch.ones_like(ids))

        episodes = collect_agent_rollouts(
            {"messages": [[{"role": "user", "content": "Compute."}]],
             "tools": [[]], "gt": [["2"]]},
            Engine(), Tokenizer(),
            SimpleNamespace(num_generations=2, max_turns=1, max_seq_len=64,
                             max_gen_len=8, device="cpu", thinking_ratio=0.0),
        )
        self.assertEqual(len(episodes), 2)
        self.assertTrue(all(episode["unfinished"] for episode in episodes))
        self.assertEqual([len(episode["actions"]) for episode in episodes], [1, 1])


class TestTokenizerTraining(unittest.TestCase):
    def test_reads_pretrain_and_conversation_jsonl(self):
        rows = [{"text": "small pretraining text"},
                {"conversations": [{"role": "user", "content": "question"},
                                   {"role": "assistant", "content": "answer"}]}]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.jsonl")
            with open(path, "w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            self.assertEqual(list(get_texts(path, max_lines=1)), ["small pretraining text"])
            self.assertEqual(len(list(get_texts(path))), 2)

    def test_rejects_empty_data_and_saves_loadable_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_path = os.path.join(directory, "empty.jsonl")
            with open(empty_path, "w", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "没有可用文本"):
                list(get_texts(empty_path))

            data_path = os.path.join(directory, "data.jsonl")
            with open(data_path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"text": "MiniMind tokenizer test data for BPE."}) + "\n")
                stream.write(json.dumps({"text": "测试分词器训练与加载。"}) + "\n")
            output_dir = os.path.join(directory, "tokenizer")
            train_tokenizer(data_path, output_dir, vocab_size=300, max_lines=2)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "tokenizer.json")))
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"},
                 {"role": "assistant", "content": "hi"}], tokenize=False
            )
            self.assertIn("<|im_start|>user", prompt)


if __name__ == "__main__":
    unittest.main()
