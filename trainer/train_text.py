"""Shared text-only training loop for MiniMind-O's language backbone."""

import argparse
import copy
import os
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.text_dataset import TextPretrainDataset, TextSFTDataset
from model.lora import inject_lora, lora_state_dict, merge_lora
from model.model_omni import OmniConfig
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, get_epoch_sampler, get_lr, init_distributed_mode, init_omni_model,
    is_main_process, log_model_params, omni_checkpoint, setup_seed,
)
from trainer.training_losses import causal_lm_loss, masked_distillation_loss

warnings.filterwarnings("ignore")


def _unwrap(model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, "_orig_mod", model)


def _text_only_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.thinker.parameters():
        parameter.requires_grad_(True)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(True)


def _train_epoch(task, epoch, loader, total_steps, start_step, model, teacher,
                 optimizer, scaler, autocast_ctx, args, config, wandb=None):
    model.train()
    if teacher:
        teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids, labels = input_ids.to(args.device), labels.to(args.device)
        lr = get_lr(epoch * total_steps + step, args.epochs * total_steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast_ctx:
            output = model(input_ids, text_only=True)
            ce_loss = causal_lm_loss(output.logits, labels)
            aux_loss = output.aux_loss
            kd_loss = ce_loss.new_zeros(())
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids, text_only=True).logits
                kd_loss = masked_distillation_loss(
                    output.logits, teacher_logits, labels, temperature=args.temperature
                )
                objective = args.alpha * ce_loss + (1 - args.alpha) * kd_loss
            else:
                objective = ce_loss
            objective = (objective + aux_loss) / args.accumulation_steps

        scaler.scale(objective).backward()
        final_batch = step == total_steps
        if step % args.accumulation_steps == 0 or final_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or final_batch:
            Logger(
                f"{task} Epoch:[{epoch + 1}/{args.epochs}]({step}/{total_steps}), "
                f"loss:{objective.item() * args.accumulation_steps:.4f}, ce:{ce_loss.item():.4f}, "
                f"kd:{kd_loss.item():.4f}, aux:{aux_loss.item():.4f}, lr:{lr:.8f}, "
                f"elapsed:{(time.time() - start_time) / 60:.1f}min"
            )
            if wandb:
                wandb.log({"loss": objective.item() * args.accumulation_steps,
                           "ce_loss": ce_loss.item(), "distillation_loss": kd_loss.item(),
                           "aux_loss": aux_loss.item(), "learning_rate": lr})

        if (step % args.save_interval == 0 or final_batch) and is_main_process():
            raw = _unwrap(model)
            moe_suffix = "_moe" if config.use_moe else ""
            path = os.path.join(args.save_dir, f"{args.save_weight}_{config.hidden_size}{moe_suffix}.pth")
            state = (lora_state_dict(raw) if task == "lora" else raw.state_dict())
            torch.save({key: value.detach().half().cpu() for key, value in state.items()}, path)
            omni_checkpoint(config, weight=args.save_weight, model=model, optimizer=optimizer,
                            scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                            save_dir=args.resume_dir, batch_size=args.batch_size)
        del output, ce_loss, aux_loss, kd_loss, objective


def main(task):
    defaults = {
        "pretrain": ("../dataset/pretrain_t2t_mini.jsonl", "none", "pretrain_text"),
        "sft": ("../dataset/sft_t2t_mini.jsonl", "llm", "full_sft_text"),
        "lora": ("../dataset/sft_t2t_mini.jsonl", "sft_zero", "lora_text"),
        "distill": ("../dataset/sft_t2t_mini.jsonl", "sft_zero", "distill_student"),
    }
    data_default, weight_default, save_default = defaults[task]
    parser = argparse.ArgumentParser(description=f"MiniMind-O text {task}")
    parser.add_argument("--data_path", default=data_default)
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--resume_dir", default="../checkpoints")
    parser.add_argument("--save_weight", default=save_default)
    parser.add_argument("--from_weight", "--from_student_weight", dest="from_weight", default=weight_default)
    parser.add_argument("--from_resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16, help="Per-rank batch size")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--hidden_size", "--student_hidden_size", dest="hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", "--student_num_layers", dest="num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", "--student_use_moe", dest="use_moe", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--use_compile", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=f"MiniMind-O-{task.title()}")
    parser.add_argument("--teacher_weight", "--from_teacher_weight", dest="teacher_weight", default="full_sft")
    parser.add_argument("--teacher_hidden_size", type=int, default=768)
    parser.add_argument("--teacher_num_hidden_layers", "--teacher_num_layers", dest="teacher_num_hidden_layers", type=int, default=8)
    parser.add_argument("--teacher_use_moe", type=int, choices=(0, 1), default=0)
    parser.add_argument("--alpha", type=float, default=0.5, help="CE weight for distillation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--from_lora", default="")
    parser.add_argument("--merge_lora", action="store_true")
    args = parser.parse_args()
    if args.accumulation_steps < 1 or args.max_seq_len < 2:
        parser.error("accumulation_steps must be positive and max_seq_len at least 2")
    if task == "distill" and not 0 <= args.alpha <= 1:
        parser.error("alpha must be in [0, 1]")

    local_rank = init_distributed_mode()
    args.device = (f"cuda:{local_rank}" if dist.is_initialized()
                   else args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
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
    _text_only_trainable(model)
    if task == "lora":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        count = inject_lora(model.thinker, rank=args.lora_rank, alpha=args.lora_alpha,
                            dropout=args.lora_dropout)
        if count == 0:
            raise RuntimeError("No eligible thinker Linear layers found for LoRA")
        if args.from_lora:
            adapter = torch.load(args.from_lora, map_location="cpu", weights_only=True)
            model.load_state_dict(adapter, strict=False)
    if resume:
        model.load_state_dict(resume["model"], strict=False)

    teacher = None
    if task == "distill":
        teacher_config = OmniConfig(hidden_size=args.teacher_hidden_size,
                                    num_hidden_layers=args.teacher_num_hidden_layers,
                                    use_moe=bool(args.teacher_use_moe))
        teacher, _ = init_omni_model(
            teacher_config, from_weight=args.teacher_weight, tokenizer_path=args.tokenizer_path,
            audio_encoder_path=None, vision_model_path=None, save_dir=args.save_dir,
            device=args.device,
        )
        teacher.eval().requires_grad_(False)

    optimizer_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(optimizer_params, lr=args.learning_rate)
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16" and "cuda" in args.device))
    if resume and "scaler" in resume:
        scaler.load_state_dict(resume["scaler"])
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.dtype)
    autocast_ctx = torch.autocast("cuda", dtype=autocast_dtype) if "cuda" in args.device and autocast_dtype else nullcontext()

    dataset_cls = TextPretrainDataset if task == "pretrain" else TextSFTDataset
    dataset = dataset_cls(args.data_path, tokenizer, max_length=args.max_seq_len)
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
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        epoch_sampler = get_epoch_sampler(dataset, epoch, sampler, seed=args.seed)
        skip = start_step if epoch == start_epoch else 0
        batches = SkipBatchSampler(epoch_sampler, args.batch_size, skip)
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=args.num_workers,
                            pin_memory="cuda" in args.device)
        total_steps = len(loader) + skip
        _train_epoch(task, epoch, loader, total_steps, skip, model, teacher,
                     optimizer, scaler, autocast_ctx, args, config, wandb)
        start_step = 0

    if args.merge_lora and task == "lora" and is_main_process():
        merged = copy.deepcopy(_unwrap(model))
        merge_lora(merged.thinker)
        moe_suffix = "_moe" if config.use_moe else ""
        merged_path = os.path.join(args.save_dir,
                                   f"{args.save_weight}_merged_{config.hidden_size}{moe_suffix}.pth")
        torch.save({key: value.detach().half().cpu() for key, value in merged.state_dict().items()}, merged_path)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    main(os.environ.get("MINIMIND_TEXT_TASK", "sft"))
