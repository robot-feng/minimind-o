"""Text-only datasets shared by MiniMind-O language training stages."""

import json
import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

_SYSTEM_PROMPTS = (
    "You are a helpful AI assistant.",
    "You are MiniMind, a lightweight and helpful assistant.",
    "你是一个可靠的 AI 助手，请尽力提供准确、有帮助的回答。",
)


def preprocess_conversations(conversations, add_system_ratio=0.2):
    """Mirror MiniMind's light system-prompt augmentation for plain SFT data."""
    messages = [dict(message) for message in conversations]
    if any(message.get("tools") for message in messages):
        return messages
    if messages and messages[0].get("role") != "system" and random.random() < add_system_ratio:
        messages.insert(0, {"role": "system", "content": random.choice(_SYSTEM_PROMPTS)})
    return messages


def postprocess_chat_prompt(prompt, remove_empty_think=None):
    marker = "<think>\n\n</think>\n\n"
    if marker in prompt and (remove_empty_think if remove_empty_think is not None else random.random() > 0.2):
        return prompt.replace(marker, "")
    return prompt


def _read_messages(messages):
    parsed, tools = [], None
    for item in messages:
        message = dict(item)
        if message.get("role") == "system" and message.get("tools"):
            tools = message["tools"]
            if isinstance(tools, str):
                tools = json.loads(tools)
        if isinstance(message.get("tool_calls"), str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        parsed.append(message)
    return parsed, tools


class TextPretrainDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = load_dataset("json", data_files=path, split="train")
        self.tokenizer, self.max_length = tokenizer, max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = str(self.samples[index]["text"])
        ids = self.tokenizer(text, add_special_tokens=False, truncation=True,
                             max_length=self.max_length - 2).input_ids
        ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
        valid_len = min(len(ids), self.max_length)
        ids = ids[:self.max_length] + [self.pad_id] * max(0, self.max_length - len(ids))
        labels = ids.copy()
        labels[valid_len:] = [-100] * (self.max_length - valid_len)
        return torch.tensor(ids), torch.tensor(labels)


class TextSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.samples = load_dataset("json", data_files=path, split="train")
        self.tokenizer, self.max_length = tokenizer, max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        self.assistant_prefix = self._encode(f"{tokenizer.bos_token}assistant\n")
        self.message_end = self._encode(f"{tokenizer.eos_token}\n")

    def __len__(self):
        return len(self.samples)

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _format(self, conversations):
        messages, tools = _read_messages(preprocess_conversations(conversations))
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )
        return postprocess_chat_prompt(prompt)

    def __getitem__(self, index):
        sample = self.samples[index]
        ids = self._encode(self._format(sample["conversations"]))[:self.max_length]
        labels = [-100] * len(ids)
        pos = 0
        while pos < len(ids):
            if ids[pos:pos + len(self.assistant_prefix)] == self.assistant_prefix:
                start = pos + len(self.assistant_prefix)
                end = start
                while end < len(ids) and ids[end:end + len(self.message_end)] != self.message_end:
                    end += 1
                stop = min(end + len(self.message_end), len(ids))
                labels[start:stop] = ids[start:stop]
                pos = max(stop, pos + 1)
            else:
                pos += 1
        valid_len = len(ids)
        if not any(label != -100 for label in labels):
            raise ValueError(f"SFT sample {index} has no assistant targets after truncation")
        ids += [self.pad_id] * (self.max_length - valid_len)
        labels += [-100] * (self.max_length - valid_len)
        return torch.tensor(ids), torch.tensor(labels)


class RLAIFPromptDataset(Dataset):
    def __init__(self, path, tokenizer, thinking_ratio=0.5):
        self.samples = load_dataset("json", data_files=path, split="train")
        self.tokenizer, self.thinking_ratio = tokenizer, thinking_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import random

        messages, tools = _read_messages(preprocess_conversations(self.samples[index]["conversations"]))
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            open_thinking=random.random() < self.thinking_ratio, tools=tools,
        )
        return {"prompt": prompt}


class AgentPromptDataset(Dataset):
    def __init__(self, path):
        self.samples = load_dataset("json", data_files=path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = _read_messages(sample["conversations"])
        ground_truth = sample.get("gt", [])
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        return {"messages": messages[:-1], "tools": tools, "gt": ground_truth}
