import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD, AdamW
from torch.optim.lr_scheduler import LRScheduler
from typing import TYPE_CHECKING, List, Tuple
from decent_dp.ddp import OPTIM_FN_TYPE, LR_SCHEDULER_FN_TYPE
from decent_dp.optim import AccumAdam
from functools import partial
from cnn.conf import Config, AdamConfig, AccumAdamConfig, SGDConfig, AdamWConfig

def get_params(model: nn.Module, weight_decay: float) -> list:
    bn_params = [v for n, v in model.named_parameters() if ('bn' in n) or ('bias' in n)]
    rest_params = [v for n, v in model.named_parameters() if not (('bn' in n) or ('bias' in n))]
    return [
        {"params": bn_params, "weight_decay": 0},
        {"params": rest_params, "weight_decay": weight_decay}
    ]

def get_params_from_list(params: List[Tuple[str, torch.Tensor]], weight_decay: float) -> list:
    bn_params = [v for n, v in params if ('bn' in n) or ('bias' in n)]
    rest_params = [v for n, v in params if not (('bn' in n) or ('bias' in n))]
    return [
        {"params": bn_params, "weight_decay": 0},
        {"params": rest_params, "weight_decay": weight_decay}
    ]

def get_optim(cfg: Config, model: nn.Module) -> Optimizer:
    optim_cfg = cfg.train.optim
    match optim_cfg.name.lower():
        case "adam":
            assert isinstance(optim_cfg, AdamConfig)
            return Adam(get_params(model, optim_cfg.weight_decay),
                        lr=cfg.train.lr,
                        betas=(optim_cfg.beta1,optim_cfg.beta2),
                        eps=optim_cfg.epsilon)
        case "sgd":
            assert isinstance(optim_cfg, SGDConfig)
            return SGD(get_params(model, optim_cfg.weight_decay),
                       lr=cfg.train.lr,
                       momentum=optim_cfg.momentum)
        case "adamw":
            assert isinstance(optim_cfg, AdamWConfig)
            return AdamW(get_params(model, optim_cfg.weight_decay),
                         lr=cfg.train.lr,
                         betas=(optim_cfg.beta1, optim_cfg.beta2),
                         eps=optim_cfg.epsilon)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.train.optim.name}")

def get_optim_fn(cfg: Config) -> OPTIM_FN_TYPE:
    optim_cfg = cfg.train.optim
    match optim_cfg.name.lower():
        case "adam":
            assert isinstance(optim_cfg, AdamConfig)
            def adam_optim_fn(params: List[Tuple[str, torch.Tensor]],
                              lr: float,
                              betas: Tuple[float, float],
                              eps: float,
                              weight_decay: float):
                return Adam(get_params_from_list(params, weight_decay),
                            lr=lr,
                            betas=betas,
                            eps=eps)
            return partial(adam_optim_fn,
                           lr=cfg.train.lr,
                           betas=(optim_cfg.beta1, optim_cfg.beta2),
                           eps=optim_cfg.epsilon,
                           weight_decay=optim_cfg.weight_decay)
        case "sgd":
            assert isinstance(optim_cfg, SGDConfig)
            def sgd_optim_fn(params: List[Tuple[str, torch.Tensor]],
                             lr: float,
                             momentum: float,
                             weight_decay: float):
                return SGD(get_params_from_list(params, weight_decay),
                           lr=lr,
                           momentum=momentum)
            return partial(sgd_optim_fn,
                           lr=cfg.train.lr,
                           momentum=optim_cfg.momentum,
                           weight_decay=optim_cfg.weight_decay)
        case "adamw":
            assert isinstance(optim_cfg, AdamWConfig)
            def adamw_optim_fn(params: List[Tuple[str, torch.Tensor]],
                               lr: float,
                               betas: Tuple[float, float],
                               eps: float,
                               weight_decay: float):
                return AdamW(get_params_from_list(params, weight_decay),
                             lr=lr,
                             betas=betas,
                             eps=eps)
            return partial(adamw_optim_fn,
                           lr=cfg.train.lr,
                           betas=(optim_cfg.beta1, optim_cfg.beta2),
                           eps=optim_cfg.epsilon,
                           weight_decay=optim_cfg.weight_decay)
        case "accumadam":
            assert isinstance(optim_cfg, AccumAdamConfig)
            def accumadam_optim_fn(params: List[Tuple[str, torch.Tensor]],
                                  lr: float,
                                  betas: Tuple[float, float],
                                  eps: float,
                                  weight_decay: float,
                                  accum_iter: int):
                return AccumAdam(get_params_from_list(params, weight_decay),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 accum_iter=accum_iter)
            return partial(accumadam_optim_fn,
                           lr=cfg.train.lr,
                           betas=(optim_cfg.beta1, optim_cfg.beta2),
                           eps=optim_cfg.epsilon,
                           weight_decay=optim_cfg.weight_decay,
                           accum_iter=optim_cfg.accum_iter)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.train.optim.name}")

def get_lr_scheduler(cfg: Config, optim: Optimizer, num_steps_per_epoch: int) -> LRScheduler:
    match cfg.train.lr_scheduler.name.lower():
        case "cosine":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=cfg.train.lr_scheduler.warmup_decay,
                total_iters=num_steps_per_epoch * cfg.train.lr_scheduler.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=num_steps_per_epoch * (cfg.train.max_epochs - cfg.train.lr_scheduler.warmup_epochs),
                eta_min=1e-5
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optim,
                schedulers=[warmup_lr_scheduler, main_scheduler],
                milestones=[num_steps_per_epoch * cfg.train.lr_scheduler.warmup_epochs]
            )
        case _:
            raise ValueError(f"Unknown LR scheduler: {cfg.train.lr_scheduler.name}")

def get_lr_scheduler_fn(cfg: Config, num_steps_per_epoch: int) -> LR_SCHEDULER_FN_TYPE:
    match cfg.train.lr_scheduler.name.lower():
        case "cosine":
            def cosine_lr_scheduler_fn(optim: Optimizer,
                                       num_steps_per_epoch: int,
                                       warmup_decay: float,
                                       warmup_epochs: int,
                                       max_epochs: int):
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optim,
                    start_factor=warmup_decay,
                    total_iters=num_steps_per_epoch * warmup_epochs
                )
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim,
                    T_max=num_steps_per_epoch * (max_epochs - warmup_epochs),
                    eta_min=1e-5
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    optim,
                    schedulers=[warmup_lr_scheduler, main_scheduler],
                    milestones=[num_steps_per_epoch * warmup_epochs]
                )
            return partial(cosine_lr_scheduler_fn,
                           num_steps_per_epoch=num_steps_per_epoch,
                           warmup_decay=cfg.train.lr_scheduler.warmup_decay,
                           warmup_epochs=cfg.train.lr_scheduler.warmup_epochs,
                           max_epochs=cfg.train.max_epochs)
        case _:
            raise ValueError(f"Unknown LR scheduler: {cfg.train.lr_scheduler.name}")