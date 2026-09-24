"""Minimal LoRA adapters for MiniMind-O linear layers."""

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank, self.scaling = rank, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs):
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + update * self.scaling


def inject_lora(module, target_names=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                rank=8, alpha=16, dropout=0.0):
    """Replace matching Linear children under ``module`` and return count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            count += 1
        else:
            count += inject_lora(child, target_names, rank, alpha, dropout)
    return count


def lora_state_dict(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()
            if ".lora_A" in name or ".lora_B" in name}


def merge_lora(module):
    """Fold adapter weights into base layers in place."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            with torch.no_grad():
                child.base.weight.add_((child.lora_B @ child.lora_A).to(child.base.weight.dtype), alpha=child.scaling)
            setattr(module, name, child.base)
        else:
            merge_lora(child)
