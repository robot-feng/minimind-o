"""Small reward and policy-objective helpers for MiniMind-O online RL."""

import re

import torch
import torch.nn.functional as F


def group_relative_advantages(rewards, num_generations, eps=1e-4):
    if rewards.ndim != 1 or num_generations < 2 or rewards.numel() % num_generations:
        raise ValueError("rewards must be flat and divisible into groups of at least two")
    grouped = rewards.reshape(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + eps)).reshape(-1)


def grpo_cispo_loss(policy_logps, old_logps, reference_logps, advantages, mask,
                    beta=0.1, epsilon=0.2, loss_type="grpo", epsilon_high=5.0):
    if not (policy_logps.shape == old_logps.shape == reference_logps.shape == mask.shape):
        raise ValueError("policy, old, reference log-probs and mask must have the same shape")
    if advantages.shape != policy_logps.shape[:1]:
        raise ValueError("advantages must have one value per sampled completion")
    if beta < 0 or epsilon < 0 or epsilon_high <= 0:
        raise ValueError("beta/epsilon must be nonnegative and epsilon_high positive")
    if loss_type not in ("grpo", "cispo"):
        raise ValueError("loss_type must be 'grpo' or 'cispo'")
    mask = mask.to(policy_logps.dtype)
    ratio = (policy_logps - old_logps).exp()
    log_ratio_ref = reference_logps - policy_logps
    kl = log_ratio_ref.exp() - log_ratio_ref - 1
    advantage = advantages.to(policy_logps.dtype).unsqueeze(-1)
    if loss_type == "cispo":
        clipped = ratio.clamp(max=epsilon_high).detach()
        per_token = -(clipped * advantage * policy_logps - beta * kl)
    else:
        clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        per_token = -(surrogate - beta * kl)
    lengths = mask.sum(dim=-1)
    valid = lengths > 0
    if not valid.any():
        return policy_logps.sum() * 0
    per_sequence = (per_token * mask).sum(dim=-1) / lengths.clamp(min=1)
    return per_sequence[valid].mean()


def repetition_penalty(text, n=3, cap=0.5):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


def heuristic_response_reward(response):
    reward = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    answer = response
    if "</think>" in response:
        thinking, answer = response.split("</think>", 1)
        reward += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        reward += 0.25 if response.count("</think>") == 1 else -0.25
    return reward - repetition_penalty(answer)


class RewardModel:
    """Optional local reward model; heuristic rewards work without a checkpoint."""

    def __init__(self, model_path="", device="cpu", dtype=torch.float16):
        self.model = None
        self.tokenizer = None
        self.device = device
        if not model_path:
            return
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model.to(device).eval()

    @torch.no_grad()
    def score(self, prompt, response):
        if self.model is None:
            return 0.0
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        if hasattr(self.model, "get_score"):
            value = self.model.get_score(self.tokenizer, messages)
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            outputs = self.model(**inputs)
            scores = getattr(outputs, "scores", None)
            if scores is None:
                scores = outputs.logits
            value = scores.reshape(-1)[-1].item()
        return float(max(min(value, 3.0), -3.0))


def score_responses(prompts, responses, reward_model=None):
    if len(responses) % len(prompts):
        raise ValueError("responses must contain an equal number of generations per prompt")
    repeats = len(responses) // len(prompts)
    scores = [heuristic_response_reward(response) +
              (reward_model.score(prompts[i], response) if reward_model else 0.0)
              for i, prompt in enumerate(prompts) for response in responses[i * repeats:(i + 1) * repeats]]
    return torch.tensor(scores, dtype=torch.float32)


def _reward_values(rewards):
    return rewards.detach().float().cpu().tolist() if isinstance(rewards, torch.Tensor) else list(rewards)


def format_rollout_debug(step, prompts, completions, rewards, num_generations):
    """Format sampled prompt, completion, and reward diagnostics for RL training."""
    reward_values = _reward_values(rewards)
    expected = len(prompts) * num_generations
    if num_generations < 1 or len(completions) != expected or len(reward_values) != expected:
        raise ValueError("completions and rewards must match prompts * num_generations")
    lines = []
    for prompt_index, prompt in enumerate(prompts):
        lines.extend((f"[DEBUG] step={step}, sample[{prompt_index}]", "=" * 100,
                      f"{'=' * 30} CONTEXT_BEGIN {'=' * 30}", str(prompt),
                      f"{'=' * 31} CONTEXT_END {'=' * 31}"))
        start = prompt_index * num_generations
        for generation in range(num_generations):
            index = start + generation
            lines.extend((f"{'=' * 28} gen[{generation}] RESPONSE_BEGIN {'=' * 28}",
                          str(completions[index]),
                          f"{'=' * 29} gen[{generation}] RESPONSE_END {'=' * 29}",
                          f"[DEBUG] gen[{generation}] reward={reward_values[index]:.4f}"))
        lines.append("=" * 100)
    return "\n".join(lines)


def format_agent_rollout_debug(step, episodes, rewards, num_generations):
    """Format multi-turn agent trajectories grouped by source prompt."""
    reward_values = _reward_values(rewards)
    if (num_generations < 1 or len(episodes) != len(reward_values)
            or len(episodes) % num_generations):
        raise ValueError("episodes and rewards must form complete generation groups")
    lines = []
    for prompt_index in range(len(episodes) // num_generations):
        lines.extend((f"[DEBUG] step={step}, sample[{prompt_index}]", "=" * 100))
        start = prompt_index * num_generations
        for generation, index in enumerate(range(start, start + num_generations)):
            episode = episodes[index]
            tools = ", ".join(tool.get("name", "") for tool in episode["tools"])
            lines.extend((f"[DEBUG] gen[{generation}] tools={tools or '(none)'}",
                          f"{'=' * 28} CONTEXT_BEGIN {'=' * 28}",
                          str(episode["prompt"]),
                          f"{'=' * 29} CONTEXT_END {'=' * 29}"))
            for turn, response in enumerate(episode["turns"]):
                lines.extend((f"{'=' * 28} turn[{turn}] RESPONSE_BEGIN {'=' * 28}",
                              str(response),
                              f"{'=' * 29} turn[{turn}] RESPONSE_END {'=' * 29}"))
            lines.append(
                f"[DEBUG] reward={reward_values[index]:.4f}, unfinished={episode['unfinished']}"
            )
        lines.append("=" * 100)
    return "\n".join(lines)
