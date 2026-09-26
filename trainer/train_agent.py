"""Multi-turn tool-use Agent RL for MiniMind-O (GRPO or CISPO objective)."""

import argparse
import copy
import json
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.text_dataset import AgentPromptDataset
from model.model_omni import OmniConfig
from trainer.agent_tools import calculate_agent_reward, execute_tool, parse_tool_calls
from trainer.rl_utils import (
    RewardModel, format_agent_rollout_debug, group_relative_advantages, grpo_cispo_loss,
)
from trainer.rollout_engine import create_rollout_engine, unwrap_model
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, get_epoch_sampler, get_lr, init_distributed_mode, init_omni_model,
    is_main_process, log_model_params, omni_checkpoint, setup_seed,
)


def _trainable_text(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.thinker.parameters():
        parameter.requires_grad_(True)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(True)


def _collate(batch):
    return {"messages": [row["messages"] for row in batch],
            "tools": [row["tools"] for row in batch], "gt": [row["gt"] for row in batch]}


def collect_agent_rollouts(batch, engine, tokenizer, args):
    count = len(batch["messages"]) * args.num_generations
    episodes = [{"messages": [dict(m) for m in batch["messages"][i // args.num_generations]],
                 "tools": batch["tools"][i // args.num_generations],
                 "gt": batch["gt"][i // args.num_generations], "turns": [],
                 "actions": [], "unfinished": False, "final": "",
                 "prompt": next((m.get("content", "") for m in reversed(batch["messages"][i // args.num_generations])
                                 if m.get("role") == "user"), "")}
                for i in range(count)]
    active = list(range(count))

    for turn in range(args.max_turns):
        if not active:
            break
        contexts = [tokenizer.apply_chat_template(
            episodes[i]["messages"], tokenize=False, add_generation_prompt=True,
            tools=episodes[i]["tools"], open_thinking=(turn == 0 and random.random() < args.thinking_ratio),
        ) for i in active]
        prompt_limit = min(args.max_seq_len, args.max_total_len - args.max_gen_len)
        encoded = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True,
                            max_length=prompt_limit, add_special_tokens=False).to(args.device)
        result = engine.rollout(encoded.input_ids, encoded.attention_mask,
                                num_generations=1, max_new_tokens=args.max_gen_len)
        next_active = []
        for row, episode_id in enumerate(active):
            episode = episodes[episode_id]
            prompt_mask = encoded.attention_mask[row].bool()
            context_ids = encoded.input_ids[row][prompt_mask].tolist()
            response_mask = result.completion_mask[row].bool()
            response_ids = result.completion_ids[row][response_mask].tolist()
            old_logps = result.per_token_logps[row][response_mask].tolist()
            response = result.completions[row]
            if response_ids:
                episode["actions"].append({"context_ids": context_ids,
                                           "response_ids": response_ids,
                                           "old_logps": old_logps})
            episode["turns"].append(response)
            calls = parse_tool_calls(response)
            episode["final"] = response
            if not calls:
                continue
            episode["messages"].append({"role": "assistant", "content": response})
            for call in calls:
                name, arguments = call.get("name", ""), call.get("arguments", {})
                result_data = execute_tool(name, arguments)
                result_text = json.dumps(result_data or {"error": "tool not found"}, ensure_ascii=False)[:2048]
                episode["messages"].append({"role": "tool", "content": result_text})
            if turn + 1 == args.max_turns:
                episode["unfinished"] = True
            else:
                next_active.append(episode_id)
        active = next_active
    return episodes


def _action_policy_logps(model, action, device, autocast_ctx):
    context = action["context_ids"]
    response = action["response_ids"]
    input_ids = torch.tensor([context + response], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with autocast_ctx:
        output = model(input_ids, attention_mask=attention_mask,
                       logits_to_keep=len(response) + 1, text_only=True)
        logits = output.logits[:, :-1].float()
        targets = input_ids[:, -len(response):]
        logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logps.squeeze(0), output.aux_loss


def train_step(episodes, model, reference, optimizer, scaler, autocast_ctx,
               args, tokenizer, global_step):
    rewards = torch.tensor([
        calculate_agent_reward(episode["final"], episode["turns"], episode["tools"],
                               episode["gt"], episode["unfinished"], args.reward_model,
                               episode["prompt"])
        for episode in episodes
    ], device=args.device)
    if args.debug_mode and is_main_process() and global_step % args.debug_interval == 0:
        Logger(format_agent_rollout_debug(global_step, episodes, rewards, args.num_generations))
    advantages = group_relative_advantages(rewards, args.num_generations)
    actions = [(i, action) for i, episode in enumerate(episodes) for action in episode["actions"]]
    token_count = sum(len(action["response_ids"]) for _, action in actions)
    if not token_count:
        if isinstance(model, DistributedDataParallel):
            # Keep DDP ranks in sync if one rank receives only empty rollouts.
            token = tokenizer.bos_token_id
            if token is None:
                token = tokenizer.eos_token_id or 0
            input_ids = torch.full((1, 2), token, dtype=torch.long, device=args.device)
            with autocast_ctx:
                output = model(input_ids, attention_mask=torch.ones_like(input_ids),
                               logits_to_keep=1, text_only=True)
                zero_loss = output.logits.sum() * 0 + output.aux_loss * 0
            scaler.scale(zero_loss).backward()
        return rewards.mean().item(), 0

    for action_index, (episode_id, action) in enumerate(actions):
        last_action = action_index == len(actions) - 1
        sync = not (isinstance(model, DistributedDataParallel) and not last_action)
        sync_context = nullcontext() if sync else model.no_sync()
        with sync_context:
            policy_logps, aux_loss = _action_policy_logps(model, action, args.device, autocast_ctx)
            with torch.no_grad():
                ref_logps, _ = _action_policy_logps(reference, action, args.device, autocast_ctx)
            old_logps = torch.tensor(action["old_logps"], device=args.device, dtype=policy_logps.dtype)
            mask = torch.ones_like(policy_logps)
            advantage = advantages[episode_id:episode_id + 1]
            policy_loss = grpo_cispo_loss(
                policy_logps.unsqueeze(0), old_logps.unsqueeze(0), ref_logps.unsqueeze(0),
                advantage, mask.unsqueeze(0), beta=args.beta, epsilon=args.epsilon,
                loss_type=args.loss_type, epsilon_high=args.epsilon_high,
            )
            weight = len(action["response_ids"]) / token_count
            step_loss = (policy_loss * weight + aux_loss / len(actions)) / args.accumulation_steps
            scaler.scale(step_loss).backward()
        if not last_action:
            del policy_logps, ref_logps, old_logps, policy_loss, step_loss

    return rewards.mean().item(), token_count


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O multi-turn Agent RL")
    parser.add_argument("--data_path", default="../dataset/agent_rl.jsonl")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--resume_dir", default="../checkpoints")
    parser.add_argument("--save_weight", default="agent_omni")
    parser.add_argument("--from_weight", default="sft_zero")
    parser.add_argument("--from_resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-rank prompt batch")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--max_total_len", type=int, default=2500,
                        help="per-action context plus maximum response token budget")
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--loss_type", choices=("grpo", "cispo"), default="cispo")
    parser.add_argument("--thinking_ratio", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--reward_model_path", default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_compile", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MiniMind-O-Agent-RL")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    parser.add_argument("--rollout_engine", choices=("torch", "sglang"), default="torch")
    parser.add_argument("--sglang_base_url", default="http://localhost:8998")
    parser.add_argument("--sglang_shared_path", default="../out/sglang_agent")
    args = parser.parse_args()
    if (args.num_generations < 2 or args.max_turns < 1 or args.max_seq_len < 1
            or args.max_gen_len < 1 or args.max_total_len <= args.max_gen_len
            or args.accumulation_steps < 1 or args.debug_interval < 1):
        parser.error("num_generations >= 2; max_turns, max_seq_len, accumulation_steps "
                     "and debug_interval must be positive; max_total_len must exceed max_gen_len")

    local_rank = init_distributed_mode()
    args.device = (f"cuda:{local_rank}" if dist.is_initialized()
                   else args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    config = OmniConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe),
                        max_position_embeddings=args.max_total_len)
    resume = omni_checkpoint(config, weight=args.save_weight, save_dir=args.resume_dir,
                             batch_size=args.batch_size) if args.from_resume else None
    model, tokenizer = init_omni_model(
        config, from_weight=args.from_weight, tokenizer_path=args.tokenizer_path,
        audio_encoder_path=None, vision_model_path=None, save_dir=args.save_dir,
        device=args.device, from_resume=args.from_resume,
    )
    _trainable_text(model)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    if resume:
        model.load_state_dict(resume["model"], strict=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.reward_model = RewardModel(args.reward_model_path, args.device,
                                    torch.float16 if args.dtype == "float16" else torch.bfloat16)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16" and "cuda" in args.device))
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.dtype)
    autocast_ctx = torch.autocast("cuda", dtype=autocast_dtype) if "cuda" in args.device and autocast_dtype else nullcontext()
    dataset = AgentPromptDataset(args.data_path)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if dist.is_initialized() else None
    start_epoch, start_step = (resume.get("epoch", 0), resume.get("step", 0)) if resume else (0, 0)
    if args.use_wandb and is_main_process():
        import swanlab
        wandb = swanlab.init(project=args.wandb_project)
    else:
        wandb = None
    log_model_params(model)
    if args.use_compile:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    engine = create_rollout_engine(
        args.rollout_engine, policy_model=model, tokenizer=tokenizer,
        base_url=args.sglang_base_url, shared_ckpt_path=args.sglang_shared_path,
    )
    engine.update_policy(model)

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_sampler = get_epoch_sampler(dataset, epoch, sampler, seed=args.seed)
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(epoch_sampler, args.batch_size, skip)
        loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=_collate,
                            num_workers=args.num_workers, pin_memory="cuda" in args.device)
        total_steps = len(loader) + skip
        for step, batch in enumerate(loader, start=skip + 1):
            global_step = epoch * total_steps + step
            lr = get_lr(global_step, args.epochs * total_steps, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            episodes = collect_agent_rollouts(batch, engine, tokenizer, args)
            reward, response_tokens = train_step(episodes, model, reference, optimizer,
                                                 scaler, autocast_ctx, args, tokenizer, global_step)
            global_response_tokens = torch.tensor(response_tokens, device=args.device)
            if dist.is_initialized():
                dist.all_reduce(global_response_tokens, op=dist.ReduceOp.SUM)
            update_now = global_step % args.accumulation_steps == 0 or step == total_steps
            if update_now:
                if global_response_tokens.item() > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    engine.update_policy(model)
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_interval == 0 or step == total_steps:
                Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{total_steps}), "
                       f"reward:{reward:.4f}, actions:{len(episodes)}, response_tokens:{response_tokens}, lr:{lr:.8f}")
                if wandb:
                    wandb.log({"reward": reward, "response_tokens": response_tokens, "learning_rate": lr})
            if (step % args.save_interval == 0 or step == total_steps) and is_main_process():
                raw = unwrap_model(model)
                moe_suffix = "_moe" if config.use_moe else ""
                path = os.path.join(args.save_dir, f"{args.save_weight}_{config.hidden_size}{moe_suffix}.pth")
                torch.save({k: v.detach().half().cpu() for k, v in raw.state_dict().items()}, path)
                omni_checkpoint(config, weight=args.save_weight, model=model, optimizer=optimizer,
                                scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                                save_dir=args.resume_dir, batch_size=args.batch_size)
        engine.update_policy(model)
        start_step = 0
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
