"""DPO fine-tuning for MiniMind-O's text-generation head.

The trainer uses the MiniMind-O model/checkpoint format and accepts the
``chosen``/``rejected`` conversation rows used by MiniMind's dpo.jsonl. DPO is
text-only here; multimodal SFT remains available through train_sft_omni.py.
"""

import argparse
import copy
import os
import sys
import time
import warnings
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.alignment_dataset import PreferenceDataset
from model.model_omni import OmniConfig
from trainer.alignment_utils import dpo_loss, masked_sequence_logps
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, get_epoch_sampler, get_lr, init_distributed_mode, init_omni_model,
    is_main_process, log_model_params, omni_checkpoint, setup_seed,
)

warnings.filterwarnings("ignore")


def _raw_model(model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, "_orig_mod", model)


def train_epoch(epoch, loader, iters, model, reference, optimizer, scaler,
                autocast_ctx, args, config, start_step=0, wandb=None):
    model.train()
    reference.eval()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        chosen = batch["x_chosen"].to(args.device)
        rejected = batch["x_rejected"].to(args.device)
        labels = torch.cat((batch["y_chosen"], batch["y_rejected"]), dim=0).to(args.device)
        masks = torch.cat((batch["mask_chosen"], batch["mask_rejected"]), dim=0).to(args.device)
        input_ids = torch.cat((chosen, rejected), dim=0)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_logits = reference(input_ids, text_only=True).logits
                ref_scores = masked_sequence_logps(ref_logits, labels, masks)
            outputs = model(input_ids, text_only=True)
            policy_scores = masked_sequence_logps(outputs.logits, labels, masks)
            half = chosen.size(0)
            loss, accuracy = dpo_loss(
                policy_scores[:half], policy_scores[half:],
                ref_scores[:half], ref_scores[half:], beta=args.beta,
            )
            aux_loss = outputs.aux_loss
            scaled_loss = (loss + aux_loss) / args.accumulation_steps

        scaler.scale(scaled_loss).backward()
        final_batch = step == iters
        if step % args.accumulation_steps == 0 or final_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or final_batch:
            elapsed = time.time() - started
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss:{loss.item():.4f}, dpo:{loss.item():.4f}, "
                f"pair_acc:{accuracy.item():.3f}, aux:{aux_loss.item():.4f}, "
                f"lr:{lr:.8f}, elapsed:{elapsed / 60:.1f}min"
            )
            if wandb:
                wandb.log({"loss": loss.item(), "dpo_loss": loss.item(),
                           "pair_accuracy": accuracy.item(), "aux_loss": aux_loss.item(),
                           "learning_rate": lr})

        if (step % args.save_interval == 0 or final_batch) and is_main_process():
            raw = _raw_model(model)
            moe_suffix = "_moe" if config.use_moe else ""
            path = os.path.join(args.save_dir, f"{args.save_weight}_{config.hidden_size}{moe_suffix}.pth")
            state = {key: value.detach().half().cpu() for key, value in raw.state_dict().items()
                     if not key.startswith(("audio_encoder.", "vision_encoder."))}
            torch.save(state, path)
            omni_checkpoint(config, weight=args.save_weight, model=model, optimizer=optimizer,
                            scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                            save_dir=args.resume_dir, batch_size=args.batch_size)
        del input_ids, labels, masks, outputs, ref_logits, ref_scores, policy_scores, scaled_loss


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O DPO")
    parser.add_argument("--data_path", default="../dataset/dpo.jsonl")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--resume_dir", default="../checkpoints")
    parser.add_argument("--save_weight", default="dpo_omni")
    parser.add_argument("--from_weight", default="sft_zero")
    parser.add_argument("--from_resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-rank batch size")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--use_compile", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MiniMind-O-DPO")
    args = parser.parse_args()
    if args.beta <= 0 or args.accumulation_steps < 1:
        parser.error("beta must be positive and accumulation_steps must be at least 1")

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    else:
        args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    config = OmniConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe))
    resume = omni_checkpoint(config, weight=args.save_weight, save_dir=args.resume_dir,
                             batch_size=args.batch_size) if args.from_resume else None
    model, tokenizer = init_omni_model(
        config, from_weight=args.from_weight, tokenizer_path=args.tokenizer_path,
        audio_encoder_path=None, vision_model_path=None, save_dir=args.save_dir,
        device=args.device, from_resume=args.from_resume,
    )
    # Clone the initial policy before loading a DPO resume checkpoint so the
    # reference remains fixed at the original SFT/base weights.
    reference = copy.deepcopy(model)
    reference.eval().requires_grad_(False)
    # DPO rows contain text conversations. Only update the text backbone and
    # vocabulary head; keep the unobserved audio/vision paths fixed.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.thinker.parameters():
        parameter.requires_grad_(True)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(True)
    if resume:
        model.load_state_dict(resume["model"], strict=False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16" and "cuda" in args.device))
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.dtype)
    autocast_ctx = (torch.autocast("cuda", dtype=autocast_dtype)
                    if "cuda" in args.device and autocast_dtype else nullcontext())
    train_set = PreferenceDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if dist.is_initialized() else None
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
        model = DistributedDataParallel(model, device_ids=[local_rank])
    for epoch in range(start_epoch, args.epochs):
        epoch_sampler = get_epoch_sampler(train_set, epoch, sampler, seed=args.seed)
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(epoch_sampler, args.batch_size, skip)
        loader = DataLoader(train_set, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory="cuda" in args.device)
        total_steps = len(loader) + skip
        train_epoch(epoch, loader, total_steps, model, reference, optimizer, scaler,
                    autocast_ctx, args, config, start_step=skip, wandb=wandb)
        start_step = 0
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
