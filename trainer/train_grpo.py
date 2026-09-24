"""MiniMind-O GRPO/CISPO for text-only RLAIF prompts."""

import argparse
import copy
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.text_dataset import RLAIFPromptDataset
from model.model_omni import OmniConfig
from trainer.rl_utils import (
    RewardModel, format_rollout_debug, group_relative_advantages, grpo_cispo_loss,
    score_responses,
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


def _policy_logps(model, input_ids, n_keep, attention_mask):
    output = model(input_ids, attention_mask=attention_mask,
                   logits_to_keep=n_keep + 1, text_only=True)
    logits = output.logits[:, :-1].float()
    targets = input_ids[:, -n_keep:]
    logps = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logps, output.aux_loss


def train_epoch(epoch, loader, total_steps, start_step, model, reference,
                rollout, reward_model, optimizer, scaler, autocast_ctx, args,
                config, tokenizer, wandb=None):
    model.train()
    started = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        lr = get_lr(epoch * total_steps + step, args.epochs * total_steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        prompts = batch["prompt"]
        encoded = tokenizer(prompts, padding=True, truncation=True,
                            max_length=args.max_seq_len, return_tensors="pt").to(args.device)
        result = rollout.rollout(encoded.input_ids, encoded.attention_mask,
                                 num_generations=args.num_generations,
                                 max_new_tokens=args.max_gen_len)
        completion_mask = result.completion_mask.to(args.device)
        full_mask = torch.cat((encoded.attention_mask.repeat_interleave(args.num_generations, dim=0),
                               completion_mask), dim=1)
        rewards = score_responses(prompts, result.completions, reward_model).to(args.device)
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            Logger(format_rollout_debug(step, prompts, result.completions, rewards,
                                        args.num_generations))
        advantages = group_relative_advantages(rewards, args.num_generations)
        n_keep = result.completion_ids.size(1)

        with autocast_ctx:
            policy_logps, aux_loss = _policy_logps(model, result.output_ids, n_keep, full_mask)
        with torch.no_grad():
            reference_logps, _ = _policy_logps(reference, result.output_ids, n_keep, full_mask)
        loss = grpo_cispo_loss(
            policy_logps, result.per_token_logps.detach(), reference_logps,
            advantages, completion_mask, beta=args.beta, epsilon=args.epsilon,
            loss_type=args.loss_type, epsilon_high=args.epsilon_high,
        )
        scaled_loss = (loss + aux_loss) / args.accumulation_steps
        scaler.scale(scaled_loss).backward()

        final_batch = step == total_steps
        if step % args.accumulation_steps == 0 or final_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or final_batch:
            lengths = completion_mask.sum(dim=1)
            log_ratio = (reference_logps - policy_logps).detach()
            kl = ((log_ratio.exp() - log_ratio - 1) * completion_mask).sum() / completion_mask.sum().clamp(min=1)
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{total_steps}), "
                   f"reward:{rewards.mean().item():.4f}, kl:{kl.item():.4f}, "
                   f"policy:{loss.item():.4f}, response_len:{lengths.float().mean().item():.1f}, "
                   f"lr:{lr:.8f}")
            if wandb:
                wandb.log({"reward": rewards.mean().item(), "kl_ref": kl.item(),
                           "policy_loss": loss.item(), "response_len": lengths.float().mean().item()})

        if (step % args.save_interval == 0 or final_batch) and is_main_process():
            raw = unwrap_model(model)
            moe_suffix = "_moe" if config.use_moe else ""
            path = os.path.join(args.save_dir, f"{args.save_weight}_{config.hidden_size}{moe_suffix}.pth")
            torch.save({k: v.detach().half().cpu() for k, v in raw.state_dict().items()}, path)
            omni_checkpoint(config, weight=args.save_weight, model=model, optimizer=optimizer,
                            scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                            save_dir=args.resume_dir, batch_size=args.batch_size)
        rollout.update_policy(model)
        del encoded, result, completion_mask, full_mask, rewards, advantages
        del policy_logps, reference_logps, loss, scaled_loss
    rollout.update_policy(model)


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O GRPO / CISPO")
    parser.add_argument("--data_path", default="../dataset/rlaif.jsonl")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--resume_dir", default="../checkpoints")
    parser.add_argument("--save_weight", default="grpo_omni")
    parser.add_argument("--from_weight", default="sft_zero")
    parser.add_argument("--from_resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-rank prompt batch")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--loss_type", choices=("grpo", "cispo"), default="grpo")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--reward_model_path", default="", help="Optional local reward model; defaults to heuristic rewards")
    parser.add_argument("--thinking_ratio", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_compile", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MiniMind-O-GRPO")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    parser.add_argument("--rollout_engine", choices=("torch", "sglang"), default="torch")
    parser.add_argument("--sglang_base_url", default="http://localhost:8998")
    parser.add_argument("--sglang_shared_path", default="../out/sglang_grpo")
    args = parser.parse_args()
    if args.num_generations < 2 or args.accumulation_steps < 1 or args.debug_interval < 1:
        parser.error("num_generations must be >= 2; accumulation_steps and debug_interval must be positive")

    local_rank = init_distributed_mode()
    args.device = (f"cuda:{local_rank}" if dist.is_initialized()
                   else args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    config = OmniConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe), max_position_embeddings=args.max_seq_len + args.max_gen_len)
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

    reward_model = RewardModel(args.reward_model_path, args.device,
                               torch.float16 if args.dtype == "float16" else torch.bfloat16)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16" and "cuda" in args.device))
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.dtype)
    autocast_ctx = torch.autocast("cuda", dtype=autocast_dtype) if "cuda" in args.device and autocast_dtype else nullcontext()
    dataset = RLAIFPromptDataset(args.data_path, tokenizer, args.thinking_ratio)
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
    rollout = create_rollout_engine(
        args.rollout_engine, policy_model=model, tokenizer=tokenizer,
        base_url=args.sglang_base_url, shared_ckpt_path=args.sglang_shared_path,
    )
    rollout.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        epoch_sampler = get_epoch_sampler(dataset, epoch, sampler, seed=args.seed)
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(epoch_sampler, args.batch_size, skip)
        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory="cuda" in args.device)
        total_steps = len(loader) + skip
        train_epoch(epoch, loader, total_steps, skip, model, reference, rollout,
                    reward_model, optimizer, scaler, autocast_ctx, args,
                    config, tokenizer, wandb)
        start_step = 0
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
