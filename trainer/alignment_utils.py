"""Preference optimization losses shared by MiniMind-O alignment trainers."""

import torch
import torch.nn.functional as F


def token_log_probs(logits, labels):
    """Return log p(label[t] | input[:t]) for aligned [batch, time] tensors."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits must be [batch,time,vocab] and labels [batch,time]")
    return F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def masked_sequence_logps(logits, labels, mask):
    """Sum token log probabilities over response tokens, excluding prompt/padding."""
    if mask.shape != labels.shape:
        raise ValueError("mask and labels must have the same shape")
    return (token_log_probs(logits, labels) * mask.to(logits.dtype)).sum(dim=-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    """Return the DPO loss and pairwise preference accuracy."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not (policy_chosen.shape == policy_rejected.shape == reference_chosen.shape == reference_rejected.shape):
        raise ValueError("all DPO sequence score tensors must have matching shapes")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()
    return loss, accuracy
