"""PPO advantage and clipped-objective helpers."""

import torch


def generalized_advantage_estimate(rewards, values, mask, gamma=0.99, lam=0.95):
    if not (rewards.shape == values.shape == mask.shape) or rewards.ndim != 2:
        raise ValueError("rewards, values and mask must share shape [batch,time]")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0, 1]")
    mask = mask.to(values.dtype)
    advantages = torch.zeros_like(values)
    last = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
    for t in reversed(range(values.size(1))):
        if t + 1 < values.size(1):
            next_mask = mask[:, t + 1]
            next_value = values[:, t + 1] * next_mask
        else:
            next_mask = torch.zeros_like(last)
            next_value = torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last = (delta + gamma * lam * next_mask * last) * mask[:, t]
        advantages[:, t] = last
    returns = (advantages + values) * mask
    valid = mask.bool()
    if valid.any():
        mean, std = advantages[valid].mean(), advantages[valid].std(unbiased=False)
        advantages = ((advantages - mean) / (std + 1e-8)) * mask
    return advantages, returns


def ppo_policy_loss(new_logps, old_logps, advantages, mask, clip_epsilon=0.2,
                    reference_logps=None, beta=0.0):
    if not (new_logps.shape == old_logps.shape == advantages.shape == mask.shape):
        raise ValueError("PPO log-probs, advantages and mask must have identical shapes")
    ratio = (new_logps - old_logps).exp()
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    per_token = -surrogate
    if reference_logps is not None and beta:
        log_ratio_ref = reference_logps - new_logps
        per_token = per_token + beta * (log_ratio_ref.exp() - log_ratio_ref - 1)
    lengths = mask.sum(dim=-1)
    valid = lengths > 0
    if not valid.any():
        return new_logps.sum() * 0
    return ((per_token * mask).sum(dim=-1) / lengths.clamp(min=1))[valid].mean()


def clipped_value_loss(new_values, old_values, returns, mask, clip_epsilon=0.2):
    if not (new_values.shape == old_values.shape == returns.shape == mask.shape):
        raise ValueError("PPO values, returns and mask must have identical shapes")
    clipped_values = old_values + (new_values - old_values).clamp(-clip_epsilon, clip_epsilon)
    per_token = torch.maximum((new_values - returns).square(), (clipped_values - returns).square())
    valid = mask.sum()
    if valid == 0:
        return new_values.sum() * 0
    return 0.5 * (per_token * mask).sum() / valid
