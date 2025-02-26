import os
local_rank = int(os.environ.get('LOCAL_RANK', 0))
local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) >= local_world_size, \
    'CUDA_VISIBLE_DEVICES must have enough devices for LOCAL_WORLD_SIZE'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local_rank]

import sys
import wandb
import tomllib
import argparse
import time
import tomli_w
import subprocess
from functools import partial
from typing import List, Literal, Tuple
from loguru import logger
from gpt.config import Config, AccumAdamWConfig
from gpt.model import GPTConfig, GPT
from gpt.data import OpenWebTextDataset, ALL_SPLITS
from collections import deque
import torch
import torch.distributed as dist
from decent_dp.ddp import DecentralizedDataParallel as DecentDP
from decent_dp.optim import AccumAdamW
from decent_dp.ddp import LR_SCHEDULER_FN_TYPE, OPTIM_FN_TYPE
logger.remove(0)
logger.add(sys.stdout, level='DEBUG', format='[{time:HH:mm:ss} {level:>5s}] {message}')


def load_config(path: str) -> Config:
    with open(path, 'rb') as f:
        raw = tomllib.load(f)
    return Config.model_validate(raw)


def initialize(cfg: Config):
    assert dist.is_available(), 'Distributed PyTorch is not available'
    assert torch.cuda.is_available(), 'CUDA is not available'
    dist.init_process_group(backend='nccl')
    torch.manual_seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if cfg.train.network.rank == 0:
        os.makedirs(cfg.run_dir, exist_ok=True)
        logger.add(os.path.join(cfg.run_dir, 'train.log'), level='INFO', format='[{time:HH:mm:ss} {level:>5s}] {message}')
        with open(os.path.join(cfg.run_dir, 'config.dump.toml'), 'wb') as f:
            tomli_w.dump(cfg.model_dump(), f)
    dist.barrier()


def construct_model(cfg: Config):
    model_args = dict(n_layer=cfg.model.n_layers,
                      n_head=cfg.model.n_heads,
                      n_embd=cfg.model.n_embd,
                      block_size=cfg.train.context_length,
                      bias=cfg.model.bias,
                      vocab_size=cfg.model.vocab_size,
                      dropout=cfg.model.dropout)
    gpt_cfg = GPTConfig(**model_args) # type: ignore
    return GPT(gpt_cfg)


def construct_optimizer_fn(cfg: Config) -> OPTIM_FN_TYPE:
    optim_cfg = cfg.train.optim
    match cfg.train.optim.name:
        case 'accumadamw':
            assert isinstance(optim_cfg, AccumAdamWConfig)
            def accumadamw_fn(params: List[Tuple[str, torch.Tensor]],
                              lr: float,
                              betas: Tuple[float, float],
                              eps: float,
                              weight_decay: float,
                              accum_steps: int) -> torch.optim.Optimizer:
                decay_params = [p for _, p in params if p.dim() >= 2]
                no_decay_params = [p for _, p in params if p.dim() < 2]
                param_groups = [
                    {'params': decay_params, 'weight_decay': weight_decay},
                    {'params': no_decay_params, 'weight_decay': 0.0}
                ]
                return AccumAdamW(param_groups,
                                  lr=lr,
                                  betas=betas,
                                  eps=eps,
                                  weight_decay=weight_decay,
                                  accum_iter=accum_steps)
            return partial(accumadamw_fn,
                           lr=cfg.train.lr_schedule.lr,
                           betas=optim_cfg.betas,
                           eps=optim_cfg.eps,
                           weight_decay=optim_cfg.weight_decay,
                           accum_steps=optim_cfg.accum_steps)
        case _:
            raise NotImplementedError(f'Optimizer {optim_cfg.name} is not implemented')
    return optimizer


def construct_scheduler_fn(cfg: Config) -> LR_SCHEDULER_FN_TYPE:
    match cfg.train.lr_schedule.name:
        case 'cosine':
            def cosine_fn(optimizer: torch.optim.Optimizer,
                          warmup_steps: int,
                          n_steps: int,
                          eta_min: float) -> torch.optim.lr_scheduler.LRScheduler:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                                     1.0 / warmup_steps,
                                                                     total_iters=warmup_steps)
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                              n_steps - warmup_steps,
                                                                              eta_min=eta_min)
                return torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                             schedulers=[warmup_scheduler, cosine_scheduler],
                                                             milestones=[warmup_steps])
            return partial(cosine_fn,
                           warmup_steps=cfg.train.lr_schedule.warmup_steps,
                           n_steps=cfg.train.n_steps,
                           eta_min=cfg.train.lr_schedule.eta_min)
        case _:
            raise NotImplementedError(f'LR scheduler {cfg.train.lr_schedule.name} is not implemented')


def construct_dataset(cfg: Config):
    match cfg.dataset:
        case 'openwebtext':
            return OpenWebTextDataset(cfg)
        case _:
            raise NotImplementedError(f'Dataset {cfg.dataset} is not implemented')


def train_steps(model: DecentDP,
                dataset: OpenWebTextDataset,
                scaler: torch.GradScaler,
                start_step: int,
                cfg: Config) -> float:
    # synchronize for accurate timing
    torch.cuda.synchronize()
    dist.barrier()

    start_time = time.time_ns()

    model.train()
    for _ in range(start_step, start_step + cfg.log.eval_interval):
        with torch.autocast(enabled=cfg.train.amp, device_type='cuda'):
            x, y = dataset.get_batch()
            _, loss = model(x, y)
        scaler.scale(loss).backward()
    
    model.global_avg()
    torch.cuda.synchronize()
    return (time.time_ns() - start_time) / 1e6


@torch.no_grad()
def evaluate_losses(model: torch.nn.Module,
                    dataset: OpenWebTextDataset,
                    cfg: Config) -> dict[Literal['train', 'val'], float]:
    model.eval()
    losses = {}
    for split in ALL_SPLITS:
        _losses = []
        for _ in range(cfg.log.eval_steps):
            x, y = dataset.get_batch(split)
            with torch.autocast(enabled=cfg.train.amp, device_type='cuda'):
                _, loss = model(x, y)
                _losses.append(loss.item())
        avg_loss = torch.FloatTensor(_losses).mean().cuda()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss.div_(cfg.train.network.world_size)
        losses[split] = avg_loss.item()
    return losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)

    assert cfg.train.backend == 'decentdp', 'Only DecentDP backend is supported'
    initialize(cfg)

    model = construct_model(cfg)
    model = model.cuda()

    if cfg.train.compile:
        model.forward = torch.compile(model.forward)

    optimizer_fn = construct_optimizer_fn(cfg)
    scheduler_fn = construct_scheduler_fn(cfg)
    scaler = torch.GradScaler(enabled=cfg.train.amp)

    model = DecentDP(model,
                     optim_fn=optimizer_fn,
                     lr_scheduler_fn=scheduler_fn,
                     topology=cfg.train.topology,
                     scaler=scaler,
                     grad_clip_norm=cfg.train.grad_norm_clip,
                     bucket_size_in_mb=100)

    dataset = construct_dataset(cfg)

    current_step = 0
    total_train_time = 0
    best_val_loss = float('inf')
    last_save_step = 0

    if cfg.checkpoint:
        raise NotImplementedError('Checkpoint loading is not implemented')
    dist.barrier()

    if cfg.log.wandb_log and (cfg.train.network.rank == 0):
        wandb.init(project=cfg.log.wandb_project,
                   config=cfg.model_dump(),
                   name=cfg.log.run_name,
                   dir=os.environ.get('TMPDIR', '/tmp/'))

    # evaluate initial model
    losses = evaluate_losses(model, dataset, cfg) # type: ignore
    if cfg.train.network.rank == 0:
        logger.info(f'Initial train loss: {losses["train"]:.6f}, val loss: {losses["val"]:.6f}, step: {current_step}')
        if cfg.log.wandb_log:
            wandb.log(data={'val_loss': losses["val"],
                            'train_loss': losses["train"],
                            'total_train_time': total_train_time,
                            'lr': 0.0},
                      step=current_step,
                      commit=True)

    # start training
    for _ in range(cfg.train.n_steps // cfg.log.eval_interval):
        train_time_ms = train_steps(model, dataset, scaler, current_step, cfg) # type: ignore
        current_step += cfg.log.eval_interval
        total_train_time += train_time_ms / 1000
        losses = evaluate_losses(model, dataset, cfg) # type: ignore
        if cfg.log.wandb_log and (cfg.train.network.rank == 0):
            wandb.log(data={'val_loss': losses["val"],
                            'train_loss': losses["train"],
                            'train_time': train_time_ms / 1000,
                            'time_ms_per_step': train_time_ms / cfg.log.eval_interval,
                            'total_train_time': total_train_time,
                            'lr': model._lr_schedulers[0].get_last_lr()[0]}, # type: ignore
                      step=current_step,
                      commit=True)
        if cfg.train.network.rank == 0:
            logger.info(f'Step {current_step:>6d}, Val loss: {losses["val"]:.6f}, Train loss: {losses["train"]:.6f}, ' \
                        + f'Train time: {train_time_ms / 1000:.3f} s ' \
                        + f'({train_time_ms / cfg.log.eval_interval:.3f} ms/step), Total train time: {total_train_time:.3f} s')
            torch.save({'model': model.state_dict(),
                        'optimizers': [opt.state_dict() for opt in model._optims],
                        'lr_schedulers': [sched.state_dict() for sched in model._lr_schedulers if sched is not None],
                        'step': current_step,
                        'total_train_time': total_train_time,
                        'best_val_loss': best_val_loss}, os.path.join(cfg.run_dir, 'checkpoint.pth'))
            if losses["val"] < best_val_loss:
                last_save_step = current_step
                best_val_loss = losses["val"]
                logger.info(f'New best model found. Saving...')
                if os.path.exists(os.path.join(cfg.run_dir, 'best_model.pth')):
                    os.remove(os.path.join(cfg.run_dir, 'best_model.pth'))
                subprocess.check_call(['cp', os.path.join(cfg.run_dir, 'checkpoint.pth'), os.path.join(cfg.run_dir, 'best_model.pth')])
                logger.info(f'Saved model to {os.path.join(cfg.run_dir, "best_model.pth")}')

    dist.destroy_process_group()
