"""MiniMind-O PPO with an explicit value head and GAE advantages."""

import argparse
import copy
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.text_dataset import RLAIFPromptDataset
from model.model_omni import OmniConfig
from trainer.ppo_utils import clipped_value_loss, generalized_advantage_estimate, ppo_policy_loss
from trainer.rl_utils import RewardModel, score_responses
from trainer.rollout_engine import create_rollout_engine, unwrap_model
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, get_epoch_sampler, get_lr, init_distributed_mode, init_omni_model,
    is_main_process, log_model_params, omni_checkpoint, setup_seed,
)


class PPOValueModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        output = self.backbone(input_ids, attention_mask=attention_mask,
                               text_only=True, return_hidden_states=True)
        return self.value_head(output.hidden_states).squeeze(-1), output.aux_loss


def _trainable_value(critic):
    for name, parameter in critic.named_parameters():
        parameter.requires_grad_(name.startswith("backbone.model.") or name.startswith("value_head."))


def _policy_logps(model, input_ids, n_keep, attention_mask):
    output = model(input_ids, attention_mask=attention_mask,
                   logits_to_keep=n_keep + 1, text_only=True)
    logits = output.logits[:, :-1].float()
    targets = input_ids[:, -n_keep:]
    logps = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logps, output.aux_loss


def train_batch(model, critic, reference, actor_optimizer, critic_optimizer,
                scaler, autocast_ctx, args, tokenizer, rollout, prompts):
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_seq_len,
                        return_tensors="pt").to(args.device)
    result = rollout.rollout(encoded.input_ids, encoded.attention_mask,
                             num_generations=1, max_new_tokens=args.max_gen_len)
    completion_mask = result.completion_mask.to(args.device).float()
    n_keep = result.completion_ids.size(1)
    prompt_mask = encoded.attention_mask
    full_mask = torch.cat((prompt_mask, completion_mask.to(prompt_mask.dtype)), dim=1)
    rewards = score_responses(prompts, result.completions, args.reward_model).to(args.device)
    positions = (encoded.input_ids.size(1) - 1) + torch.arange(n_keep, device=args.device)

    with torch.no_grad():
        old_value_seq, _ = unwrap_model(critic)(result.output_ids, attention_mask=full_mask)
        old_values = old_value_seq.gather(1, positions.expand(result.output_ids.size(0), -1))
        old_values = old_values * completion_mask
        reference_logps, _ = _policy_logps(reference, result.output_ids, n_keep, full_mask)

    token_rewards = torch.zeros_like(old_values)
    lengths = completion_mask.sum(dim=1).long()
    valid = lengths > 0
    rows = torch.arange(len(prompts), device=args.device)[valid]
    token_rewards[rows, lengths[valid] - 1] = rewards[valid]
    advantages, returns = generalized_advantage_estimate(
        token_rewards, old_values, completion_mask,
        gamma=args.gamma, lam=args.gae_lambda,
    )
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    batch_size = result.output_ids.size(0)
    mini_batch_size = max(1, min(args.mini_batch_size, batch_size))
    pending_batches = 0
    stop_updates = False
    policy_loss_value = value_loss_value = kl_value = 0.0
    metric_count = kl_count = 0

    def apply_update():
        nonlocal pending_batches
        if pending_batches == 0:
            return
        scaler.unscale_(actor_optimizer)
        scaler.unscale_(critic_optimizer)
        correction = args.accumulation_steps / pending_batches
        if correction != 1:
            for optimizer in (actor_optimizer, critic_optimizer):
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
        scaler.step(actor_optimizer)
        scaler.step(critic_optimizer)
        scaler.update()
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        pending_batches = 0

    for ppo_epoch in range(args.ppo_epochs):
        permutation = torch.randperm(batch_size, device=args.device)
        for offset in range(0, batch_size, mini_batch_size):
            indices = permutation[offset:offset + mini_batch_size]
            with autocast_ctx:
                policy_logps, actor_aux = _policy_logps(
                    model, result.output_ids[indices], n_keep, full_mask[indices]
                )
                value_seq, critic_aux = critic(
                    result.output_ids[indices], attention_mask=full_mask[indices]
                )
                values = value_seq.gather(1, positions.expand(len(indices), -1))
                log_ratio = policy_logps - result.per_token_logps[indices].detach()
                response_mask = completion_mask[indices]
                approx_kl = (0.5 * log_ratio.detach().square() * response_mask).sum() / response_mask.sum().clamp(min=1)
                if dist.is_initialized():
                    dist.all_reduce(approx_kl, op=dist.ReduceOp.AVG)
                kl_value += approx_kl.item()
                kl_count += 1

                if approx_kl.detach().item() > args.early_stop_kl:
                    scaler.scale((policy_logps.sum() + values.sum()) * 0).backward()
                    stop_updates = True
                    apply_update()
                    break

                policy_loss = ppo_policy_loss(
                    policy_logps, result.per_token_logps[indices].detach(), advantages[indices],
                    response_mask, clip_epsilon=args.clip_epsilon,
                    reference_logps=reference_logps[indices], beta=args.beta,
                )
                value_loss = clipped_value_loss(
                    values, old_values[indices], returns[indices], response_mask,
                    args.value_clip,
                )
                total_loss = policy_loss + args.value_coef * value_loss + actor_aux + critic_aux

            scaler.scale(total_loss / args.accumulation_steps).backward()
            pending_batches += 1
            policy_loss_value += policy_loss.detach().item()
            value_loss_value += value_loss.detach().item()
            metric_count += 1
            at_epoch_end = offset + mini_batch_size >= batch_size
            if pending_batches >= args.accumulation_steps or at_epoch_end:
                apply_update()
        if stop_updates:
            break

    count = max(metric_count, 1)
    return {"reward": rewards.mean().item(), "policy_loss": policy_loss_value / count,
            "value_loss": value_loss_value / count, "kl": kl_value / max(kl_count, 1),
            "response_len": completion_mask.sum(dim=1).float().mean().item()}, result


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O PPO")
    parser.add_argument("--data_path", default="../dataset/rlaif.jsonl")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--resume_dir", default="../checkpoints")
    parser.add_argument("--save_weight", default="ppo_actor_omni")
    parser.add_argument("--from_weight", default="sft_zero")
    parser.add_argument("--from_resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-rank prompt batch")
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--critic_learning_rate", type=float, default=1e-5)
    parser.add_argument("--ppo_epochs", "--ppo_update_iters", dest="ppo_epochs", type=int, default=1)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", "--lam", dest="gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--value_clip", "--cliprange_value", dest="value_clip", type=float, default=0.2)
    parser.add_argument("--value_coef", "--vf_coef", dest="value_coef", type=float, default=0.5)
    parser.add_argument("--beta", "--kl_coef", dest="beta", type=float, default=0.02)
    parser.add_argument("--early_stop_kl", type=float, default=0.25)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--reward_model_path", default="")
    parser.add_argument("--thinking_ratio", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_compile", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MiniMind-O-PPO")
    parser.add_argument("--rollout_engine", choices=("torch", "sglang"), default="torch")
    parser.add_argument("--sglang_base_url", default="http://localhost:8998")
    parser.add_argument("--sglang_shared_path", default="../out/sglang_ppo")
    args = parser.parse_args()
    if (args.ppo_epochs < 1 or args.mini_batch_size < 1 or args.max_gen_len < 1
            or args.accumulation_steps < 1 or args.early_stop_kl <= 0):
        parser.error("ppo_epochs, mini_batch_size, max_gen_len and accumulation_steps must be positive; early_stop_kl must be > 0")

    local_rank = init_distributed_mode()
    args.device = (f"cuda:{local_rank}" if dist.is_initialized()
                   else args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    config = OmniConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe),
                        max_position_embeddings=args.max_seq_len + args.max_gen_len)
    resume = omni_checkpoint(config, weight=args.save_weight, save_dir=args.resume_dir,
                             batch_size=args.batch_size) if args.from_resume else None
    model, tokenizer = init_omni_model(
        config, from_weight=args.from_weight, tokenizer_path=args.tokenizer_path,
        audio_encoder_path=None, vision_model_path=None, save_dir=args.save_dir,
        device=args.device, from_resume=args.from_resume,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.thinker.parameters():
        parameter.requires_grad_(True)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(True)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    critic = PPOValueModel(copy.deepcopy(model)).to(args.device)
    _trainable_value(critic)
    if resume:
        model.load_state_dict(resume["model"], strict=False)
        if "critic_model" in resume:
            critic.load_state_dict(resume["critic_model"], strict=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.reward_model = RewardModel(args.reward_model_path, args.device,
                                    torch.float16 if args.dtype == "float16" else torch.bfloat16)
    actor_optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    critic_optimizer = optim.AdamW([p for p in critic.parameters() if p.requires_grad], lr=args.critic_learning_rate)
    if resume:
        actor_optimizer.load_state_dict(resume["optimizer"])
        if "critic_optimizer" in resume:
            critic_optimizer.load_state_dict(resume["critic_optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16" and "cuda" in args.device))
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
        critic = torch.compile(critic)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        critic = DistributedDataParallel(critic, device_ids=[local_rank], broadcast_buffers=False)
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
        for step, batch in enumerate(loader, start=skip + 1):
            global_step = epoch * total_steps + step
            lr = get_lr(global_step, args.epochs * total_steps, args.learning_rate)
            critic_lr = get_lr(global_step, args.epochs * total_steps, args.critic_learning_rate)
            for group in actor_optimizer.param_groups:
                group["lr"] = lr
            for group in critic_optimizer.param_groups:
                group["lr"] = critic_lr
            metrics, _ = train_batch(model, critic, reference, actor_optimizer,
                                     critic_optimizer, scaler, autocast_ctx, args,
                                     tokenizer, rollout, batch["prompt"])
            rollout.update_policy(model)
            if step % args.log_interval == 0 or step == total_steps:
                Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{total_steps}), "
                       f"reward:{metrics['reward']:.4f}, kl:{metrics['kl']:.4f}, "
                       f"actor:{metrics['policy_loss']:.4f}, critic:{metrics['value_loss']:.4f}, "
                       f"response_len:{metrics['response_len']:.1f}, lr:{lr:.8f}, "
                       f"critic_lr:{critic_lr:.8f}")
                if wandb:
                    wandb.log(metrics | {"learning_rate": lr, "critic_learning_rate": critic_lr})
            if (step % args.save_interval == 0 or step == total_steps) and is_main_process():
                raw = unwrap_model(model)
                moe_suffix = "_moe" if config.use_moe else ""
                path = os.path.join(args.save_dir, f"{args.save_weight}_{config.hidden_size}{moe_suffix}.pth")
                torch.save({k: v.detach().half().cpu() for k, v in raw.state_dict().items()}, path)
                omni_checkpoint(config, weight=args.save_weight, model=model, optimizer=actor_optimizer,
                                scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                                save_dir=args.resume_dir, batch_size=args.batch_size,
                                critic_model=critic, critic_optimizer=critic_optimizer)
        start_step = 0
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
