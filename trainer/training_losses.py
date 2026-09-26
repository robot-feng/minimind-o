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


def omni_sft_losses(output, text_labels, audio_labels, vision_only=False):
    """Return text, audio, and combined losses for one Omni SFT batch."""
    if output.logits is None or output.logits.shape[:2] != text_labels.shape:
        raise ValueError("text logits and labels must have matching batch and sequence dimensions")
    text_losses = F.cross_entropy(
        output.logits.float().reshape(-1, output.logits.size(-1)),
        text_labels.reshape(-1), reduction="none", ignore_index=-100,
    )
    text_mask = text_labels.reshape(-1).ne(-100)
    text_loss = (text_losses * text_mask).sum() / text_mask.sum().clamp_min(1)
    aux_loss = output.aux_loss if output.aux_loss is not None else text_loss.new_zeros(())

    if vision_only:
        return text_loss, text_loss.new_zeros(()), text_loss + aux_loss
    if output.audio_logits is None or len(output.audio_logits) != 8 or audio_labels.ndim != 3:
        raise ValueError("eight audio logits and (batch, 8, sequence) audio labels are required")
    if audio_labels.shape != (text_labels.size(0), 8, text_labels.size(1)):
        raise ValueError("audio labels must match the text batch and sequence dimensions")

    audio_loss = output.audio_logits[0].sum() * 0
    for layer, logits in enumerate(output.audio_logits):
        targets = audio_labels[:, layer, :].reshape(-1)
        layer_losses = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), targets,
            reduction="none", ignore_index=-100,
        )
        valid = targets.ne(-100)
        weighted = layer_losses * valid * (1 + targets.eq(2050) * 9)
        audio_loss = audio_loss + weighted.sum() / valid.sum().clamp_min(1)
    audio_loss = audio_loss / 8
    return text_loss, audio_loss, text_loss + audio_loss + aux_loss


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


def distillation_objective(ce_loss, kd_loss, aux_loss, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * (ce_loss + aux_loss) + (1 - alpha) * kd_loss
