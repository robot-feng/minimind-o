"""Token-masked language-model and white-box distillation objectives."""

import torch
import torch.nn.functional as F


def causal_lm_loss(logits, labels):
    shifted_logits, shifted_labels = logits[..., :-1, :], labels[..., 1:]
    valid = shifted_labels.ne(-100)
    if not valid.any():
        raise ValueError("batch contains no supervised next-token targets")
    return F.cross_entropy(shifted_logits.float().reshape(-1, shifted_logits.size(-1)),
                           shifted_labels.reshape(-1), ignore_index=-100)


def masked_distillation_loss(student_logits, teacher_logits, labels, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    student = student_logits[..., :-1, :].float() / temperature
    teacher = teacher_logits[..., :-1, :].float() / temperature
    mask = labels[..., 1:].ne(-100)
    if not mask.any():
        raise ValueError("batch contains no supervised next-token targets")
    per_token = F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1),
                         reduction="none").sum(dim=-1)
    return (per_token * mask).sum() / mask.sum() * temperature**2
