"""Small text preference dataset used by DPO-style MiniMind-O trainers."""

import json
import random

import torch
from torch.utils.data import Dataset

from dataset.text_dataset import postprocess_chat_prompt


class PreferenceDataset(Dataset):
    """Read MiniMind ``dpo.jsonl`` rows and mask assistant response tokens."""

    def __init__(self, path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id or 0
        self.bos_ids = self._encode(f"{tokenizer.bos_token}assistant\n")
        self.eos_ids = self._encode(f"{tokenizer.eos_token}\n")
        self.samples = []
        with open(path, encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                if not isinstance(sample.get("chosen"), list) or not isinstance(sample.get("rejected"), list):
                    raise ValueError(f"{path}:{line_no}: chosen and rejected must be message lists")
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def _encode(self, text):
        encoded = self.tokenizer(text, add_special_tokens=False)
        return encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]

    def _render(self, messages, remove_empty_think=None):
        messages = [dict(message) for message in messages]
        tools = None
        for message in messages:
            if message.get("role") == "system" and message.get("tools"):
                tools = message["tools"]
                if isinstance(tools, str):
                    tools = json.loads(tools)
            if isinstance(message.get("tool_calls"), str):
                message["tool_calls"] = json.loads(message["tool_calls"])
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )
        return postprocess_chat_prompt(prompt, remove_empty_think=remove_empty_think)

    def _loss_mask(self, ids):
        mask = [0] * len(ids)
        i = 0
        while i < len(ids):
            if ids[i:i + len(self.bos_ids)] == self.bos_ids:
                start = i + len(self.bos_ids)
                end = start
                while end < len(ids) and ids[end:end + len(self.eos_ids)] != self.eos_ids:
                    end += 1
                stop = min(end + len(self.eos_ids), len(ids))
                mask[start:stop] = [1] * max(0, stop - start)
                i = max(stop, i + 1)
            else:
                i += 1
        return mask

    def _encode_branch(self, messages, remove_empty_think=None):
        ids = self._encode(self._render(messages, remove_empty_think))[:self.max_length]
        mask = self._loss_mask(ids)
        if not any(mask):
            raise ValueError("DPO conversation has no assistant response tokens; check tokenizer chat template")
        ids += [self.pad_id] * (self.max_length - len(ids))
        mask += [0] * (self.max_length - len(mask))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, index):
        sample = self.samples[index]
        remove_empty_think = random.random() > 0.2
        chosen_ids, chosen_mask = self._encode_branch(sample["chosen"], remove_empty_think)
        rejected_ids, rejected_mask = self._encode_branch(sample["rejected"], remove_empty_think)
        return {
            "x_chosen": chosen_ids[:-1],
            "y_chosen": chosen_ids[1:],
            "mask_chosen": chosen_mask[1:],
            "x_rejected": rejected_ids[:-1],
            "y_rejected": rejected_ids[1:],
            "mask_rejected": rejected_mask[1:],
        }
