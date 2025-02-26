import os
import torch
import torch.distributed as dist
from typing import List, cast
from statistics import mean
from collections import deque
from loguru import logger


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append((value, n))
        self.count += n
        self.total += value * n

    @property
    def avg(self):
        count = 0.
        total = 0.
        for i in reversed(range(len(self.deque))):
            if count + self.deque[i][1] >= self.window_size:
                total += self.deque[i][0] * (self.window_size - count)
                count = self.window_size
                break
            count += self.deque[i][1]
            total += self.deque[i][0] * self.deque[i][1]
        if count == 0:
            return 0
        return total / count

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(avg=self.avg, global_avg=self.global_avg)


@torch.no_grad()
def get_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    if target.ndim == 2:
        target = target.max(dim=1)[1]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])
    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        res.append(correct_k * (100.0 / batch_size))
    return res


def initialize_dist() -> None:
    if ("WORLD_SIZE" in os.environ) and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        if dist.get_rank() == 0:
            logger.info(f"Initialized the process group with {dist.get_world_size()} processes.")
        dist.barrier()


def gather_statistics(train_loss, val_loss, val_acc1, val_acc5, val_samples) -> List[float]:
    log = [train_loss, val_loss, val_acc1, val_acc5, val_samples]
    if dist.is_available() and dist.is_initialized():
        object_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(object_list, log)
        object_list = cast(List[List[float]], object_list)
        
        log[0] = mean([x[0] for x in object_list])
        total_val_samples = sum([x[4] for x in object_list])
        log[1] = sum([x[1] for x in object_list]) / total_val_samples
        log[2] = sum([x[2] for x in object_list]) / total_val_samples
        log[3] = sum([x[3] for x in object_list]) / total_val_samples
        log[4] = total_val_samples
    else:
        log[1] /= log[4]
        log[2] /= log[4]
        log[3] /= log[4]
    return log
