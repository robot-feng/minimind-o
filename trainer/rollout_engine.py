"""Swappable native Torch and SGLang rollout backends for MiniMind-O RL."""

import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


def unwrap_model(model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, "_orig_mod", model)


@torch.no_grad()
def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    model = unwrap_model(model)
    if input_ids.is_inference():
        input_ids = input_ids.clone()
    logits = model(input_ids, attention_mask=attention_mask,
                   logits_to_keep=n_keep + 1, text_only=True).logits[:, :-1].float()
    targets = input_ids[:, -n_keep:]
    return torch.stack([
        F.log_softmax(row_logits, dim=-1).gather(-1, row_targets[:, None]).squeeze(-1)
        for row_logits, row_targets in zip(logits, targets)
    ])


@dataclass
class RolloutResult:
    output_ids: torch.Tensor
    completion_ids: torch.Tensor
    per_token_logps: torch.Tensor
    completions: List[str]
    prompt_lens: torch.Tensor
    completion_mask: torch.Tensor


class TorchRolloutEngine:
    def __init__(self, policy_model, tokenizer, temperature=0.8, top_p=0.95):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p

    def rollout(self, prompt_ids, attention_mask, num_generations=1,
                max_new_tokens=256, temperature=None):
        model = unwrap_model(self.policy_model)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        prompts = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = attention_mask.repeat_interleave(num_generations, dim=0)
        with torch.no_grad():
            generated = model.generate_text(
                prompts, attention_mask=prompt_mask, max_new_tokens=max_new_tokens,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p, eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_id,
            )
            # generate_text is inference-mode; copy back to ordinary tensors for
            # the later policy-gradient forward/backward pass.
            output_ids = generated.clone()
            completion_ids = output_ids[:, prompts.size(1):].clone()
            completion_mask = completion_ids.ne(pad_id).long()
            if self.tokenizer.eos_token_id is not None:
                for row, tokens in enumerate(completion_ids):
                    eos = tokens.eq(self.tokenizer.eos_token_id).nonzero()
                    if eos.numel():
                        end = int(eos[0, 0]) + 1
                        completion_mask[row, :end] = 1
                        completion_mask[row, end:] = 0
            full_mask = torch.cat((prompt_mask, completion_mask), dim=1)
            logps = compute_per_token_logps(self.policy_model, output_ids,
                                            completion_ids.size(1), full_mask)
        completions = []
        for tokens, mask in zip(completion_ids, completion_mask):
            valid = tokens[mask.bool()].tolist()
            text = self.tokenizer.decode(valid, skip_special_tokens=False)
            eos_text = self.tokenizer.eos_token or ""
            completions.append(text[:-len(eos_text)] if eos_text and text.endswith(eos_text) else text)
        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=logps,
            completions=completions,
            prompt_lens=prompt_mask.sum(dim=1),
            completion_mask=completion_mask,
        )

    def update_policy(self, model):
        self.policy_model = model


class SGLangRolloutEngine:
    """HTTP rollout backend for a shared-filesystem SGLang server."""

    def __init__(self, tokenizer, base_url, shared_ckpt_path, timeout=120):
        import requests
        self.http = requests
        self.tokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.shared_ckpt_path = os.path.abspath(shared_ckpt_path)
        self.timeout = timeout

    def rollout(self, prompt_ids, attention_mask, num_generations=1,
                max_new_tokens=256, temperature=0.8):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        prompts = [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, attention_mask)]
        expanded = [ids for prompt in prompts for ids in [prompt] * num_generations]
        response = self.http.post(
            f"{self.base_url}/generate",
            json={"input_ids": expanded,
                  "sampling_params": {"temperature": temperature, "max_new_tokens": max_new_tokens,
                                      "top_p": 0.95,
                                      "stop_token_ids": [self.tokenizer.eos_token_id]
                                      if self.tokenizer.eos_token_id is not None else []},
                  "return_logprob": True},
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list):
            results = [results]
        rows, logps, completions = [], [], []
        for item in results:
            meta = item.get("meta_info", {})
            tokens = meta.get("output_ids", item.get("output_ids", []))
            raw_logps = meta.get("output_token_logprobs", [])
            token_logps = []
            for value in raw_logps:
                if isinstance(value, (list, tuple)):
                    token_logps.append(float(value[0]))
                elif isinstance(value, (int, float)):
                    token_logps.append(float(value))
            if len(token_logps) < len(tokens):
                token_logps = [0.0] * (len(tokens) - len(token_logps)) + token_logps
            token_logps = token_logps[-len(tokens):] if tokens else []
            rows.append(tokens)
            logps.append(token_logps)
            completions.append(self.tokenizer.decode(tokens, skip_special_tokens=False))
        batch = len(rows)
        prompt_width = prompt_ids.size(1)
        response_width = max(1, max(map(len, rows), default=0))
        output_ids = torch.full((batch, prompt_width + response_width), pad_id,
                                dtype=torch.long, device=prompt_ids.device)
        completion_ids = torch.full((batch, response_width), pad_id,
                                    dtype=torch.long, device=prompt_ids.device)
        completion_mask = torch.zeros((batch, response_width), dtype=torch.long,
                                      device=prompt_ids.device)
        per_token_logps = torch.zeros((batch, response_width), dtype=torch.float32,
                                      device=prompt_ids.device)
        prompt_lens = attention_mask.sum(dim=1).repeat_interleave(num_generations)
        eos_text = self.tokenizer.eos_token or ""
        for i, (prompt, tokens, token_logps) in enumerate(zip(expanded, rows, logps)):
            prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=prompt_ids.device)
            output_ids[i, prompt_width - len(prompt):prompt_width] = prompt_tensor
            if tokens:
                token_tensor = torch.tensor(tokens, dtype=torch.long, device=prompt_ids.device)
                completion_ids[i, :len(tokens)] = token_tensor
                completion_mask[i, :len(tokens)] = 1
                per_token_logps[i, :len(tokens)] = torch.tensor(token_logps, device=prompt_ids.device)
                output_ids[i, prompt_width:prompt_width + len(tokens)] = token_tensor
            if eos_text and completions[i].endswith(eos_text):
                completions[i] = completions[i][:-len(eos_text)]
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_lens, completion_mask)

    def update_policy(self, model):
        ok = True
        message = ""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            try:
                raw = unwrap_model(model)
                os.makedirs(self.shared_ckpt_path, exist_ok=True)
                raw.save_pretrained(self.shared_ckpt_path, safe_serialization=False)
                self.tokenizer.save_pretrained(self.shared_ckpt_path)
                response = self.http.post(
                    f"{self.base_url}/update_weights_from_disk",
                    json={"model_path": self.shared_ckpt_path}, timeout=self.timeout,
                )
                response.raise_for_status()
            except Exception as error:
                ok, message = False, str(error)
        if torch.distributed.is_initialized():
            status = torch.tensor([int(ok)], device=next(model.parameters()).device)
            torch.distributed.broadcast(status, src=0)
            torch.distributed.barrier()
            ok = bool(status.item())
        if not ok:
            raise RuntimeError(f"SGLang policy update failed: {message}")

    def health(self):
        try:
            return self.http.get(f"{self.base_url}/health", timeout=5).ok
        except Exception:
            return False


def create_rollout_engine(engine_type="torch", policy_model=None, tokenizer=None, **kwargs):
    if tokenizer is None:
        raise ValueError("tokenizer is required for rollouts")
    if engine_type == "torch":
        if policy_model is None:
            raise ValueError("policy_model is required for torch rollouts")
        return TorchRolloutEngine(policy_model, tokenizer,
                                  temperature=kwargs.get("temperature", 0.8),
                                  top_p=kwargs.get("top_p", 0.95))
    if engine_type == "sglang":
        return SGLangRolloutEngine(tokenizer, kwargs["base_url"], kwargs["shared_ckpt_path"],
                                   kwargs.get("timeout", 120))
    raise ValueError(f"unsupported rollout engine: {engine_type}")
