import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_omni import OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.training_losses import omni_sft_losses
from trainer.trainer_utils import get_lr, get_optimizer_step, get_accumulation_window_size, Logger, is_main_process, init_distributed_mode, setup_seed, init_omni_model, omni_checkpoint, SkipBatchSampler, log_model_params

warnings.filterwarnings('ignore')


def omni_collate_fn(batch):
    """自定义collate函数，处理变长audio_inputs和pixel_values"""
    input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = zip(*batch)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    audio_labels = torch.stack(audio_labels)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)
    valid_audios = [a for a in audio_inputs if a is not None]
    if valid_audios:
        max_t = max(a.size(1) for a in valid_audios)
        padded = [a if a.size(1) == max_t else torch.nn.functional.pad(a, (0, 0, 0, max_t - a.size(1))) for a in valid_audios]
        audio_inputs = torch.cat(padded, dim=0)
    else:
        audio_inputs = None
    valid_images = [p for p in pixel_values if p is not None]
    if valid_images:
        if hasattr(valid_images[0], 'keys'):
            keys = set.intersection(*[set(d.keys()) for d in valid_images])
            pixel_values = {k: torch.cat([d[k] for d in valid_images], dim=0) for k in keys}
        else:
            pixel_values = torch.cat(valid_images, dim=0)
    else:
        pixel_values = None
    spk_emb = torch.stack(spk_emb)
    return input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    next_save_step = ((start_step // args.save_interval) + 1) * args.save_interval
    optimizer_steps_per_epoch = (iters + args.accumulation_steps - 1) // args.accumulation_steps
    for step, (input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        audio_labels = audio_labels.to(args.device)
        audio_lens = audio_lens.to(args.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(args.device)
        if pixel_values is not None:
            if hasattr(pixel_values, 'keys'):
                pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(args.device)
        spk_emb = spk_emb.to(args.device)
        optimizer_step = get_optimizer_step(epoch, step, iters, args.accumulation_steps)
        lr = get_lr(optimizer_step, args.epochs * optimizer_steps_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(
                input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens,
                pixel_values=pixel_values, spk_emb=spk_emb, text_only=args.vision_only,
            )
            text_loss, audio_loss, total_loss = omni_sft_losses(
                res, labels, audio_labels, vision_only=args.vision_only,
            )
            accumulation_window_size = get_accumulation_window_size(
                step, iters, args.accumulation_steps
            )
            loss = total_loss / accumulation_window_size

        scaler.scale(loss).backward()
        should_step = step % args.accumulation_steps == 0 or step == iters
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            text_loss_val = text_loss.item() if isinstance(text_loss, torch.Tensor) else 0
            audio_loss_val = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, text: {text_loss_val:.4f}, audio: {audio_loss_val:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: 
                wandb.log({"loss": current_loss, "text_loss": text_loss_val, 
                          "audio_loss": audio_loss_val, "lr": current_lr, "epoch_time": eta_min})

        should_save = step == iters or (step >= next_save_step and should_step)
        if should_save and is_main_process():
            model.eval()
            moe_suffix = '_moe' if omni_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.')}
            torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)
            omni_checkpoint(omni_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                          batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
                          scaler=scaler)
            model.train()
        if should_save:
            next_save_step = ((step // args.save_interval) + 1) * args.save_interval

        del input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb, res, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-O SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_omni', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2a_mini.parquet", help="训练数据路径（parquet格式）")
    parser.add_argument("--audio_encoder_dir", type=str, default="../model/SenseVoiceSmall", help="音频encoder路径(SenseVoice)")
    parser.add_argument("--vision_dir", type=str, default="google/tipsv2-b14", help="TIPSv2视觉模型 ID 或本地路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_backbone', default='none', type=str, choices=['none', 'all', 'last1'], help="冻结主干模型: none=全量训练, all=只训练audio层, last1=只训练最后1层+audio层")
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'audio_proj', 'vision_proj'], help="训练模式: all=全量训练, audio_proj=只训练audio_proj, vision_proj=只训练vision_proj")
    parser.add_argument('--vision_only', action='store_true', help="仅训练Thinker视觉文本路径，跳过音频编码与Talker；适用于I2T数据")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-O-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    if args.accumulation_steps < 1:
        parser.error("--accumulation_steps must be at least 1")
    if args.save_interval < 1:
        parser.error("--save_interval must be at least 1")
    if args.vision_only and args.mode == 'audio_proj':
        parser.error("--vision_only cannot be combined with --mode audio_proj")

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    omni_config = OmniConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    ckp_data = omni_checkpoint(
        omni_config, weight=args.save_weight, save_dir='../checkpoints', batch_size=args.batch_size
    ) if args.from_resume == 1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-O-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_omni_model(omni_config, from_weight=args.from_weight,
                                        audio_encoder_path=None if args.vision_only else args.audio_encoder_dir,
                                        vision_model_path=args.vision_dir,
                                        save_dir=args.save_dir, device=args.device,
                                        freeze_backbone=args.freeze_backbone, from_resume=args.from_resume)
    
    if args.use_compile == 1:
        model = torch.compile(model)
    
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    
    if args.mode == 'audio_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.audio_proj.parameters(): p.requires_grad = True
    elif args.mode == 'vision_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.vision_proj.parameters(): p.requires_grad = True
    if args.vision_only:
        for module in (model.talker, model.audio_proj):
            for p in module.parameters(): p.requires_grad = False
    log_model_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable: {trainable:.2f}M | Mode: {args.mode} | Vision-only: {args.vision_only} | Freeze: {args.freeze_backbone} | Compile: {"on" if args.use_compile else "off"}')
    
    # scheduled_sampling 现在会自动保护 image/audio token 的连续性
    train_ds = OmniDataset(
        args.data_path, 
        tokenizer, 
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=args.max_seq_len,
        image_token_len=model.config.image_token_len
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        saved_accumulation_steps = ckp_data.get('accumulation_steps')
        if saved_accumulation_steps is not None and saved_accumulation_steps != args.accumulation_steps:
            raise ValueError(
                f"Checkpoint uses accumulation_steps={saved_accumulation_steps}, "
                f"but this run requested {args.accumulation_steps}; resume with the saved value."
            )
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=omni_collate_fn, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
