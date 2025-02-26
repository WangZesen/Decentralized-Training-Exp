import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterator, List
from src.data.dataset import WMTDataset

class DistributedTokenBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 dataset: "WMTDataset",
                 seed: int,
                 max_tokens: int,
                 accum_steps: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = True,
                 batch_efficiency: float = 0.35):
        self._dataset = dataset
        self._seed = seed
        self._max_tokens = max_tokens
        self._accum_steps = accum_steps
        self._shuffle = shuffle
        self._batch_efficiency = batch_efficiency
        self._drop_last = drop_last

        if dist.is_available() and dist.is_initialized():
            self._num_replicas = dist.get_world_size()
            self._rank = dist.get_rank()
        else:
            self._num_replicas = 1
            self._rank = 0

        self._epoch = 0
        self._create_batches()

    def _create_batches(self):
        src_token_stats, tgt_token_stats = self._dataset.get_token_stats()
        data = list(zip(src_token_stats, tgt_token_stats, list(range(len(src_token_stats)))))
        data.sort(key=lambda x: - x[0] - x[1])

        self._batches = []
        batch = []
        src_num_tokens = 0
        tgt_num_tokens = 0
        src_max_len = 0
        tgt_max_len = 0

        for i in range(len(data)):
            next_src_num_tokens = src_num_tokens + data[i][0]
            next_tgt_num_tokens = tgt_num_tokens + data[i][1]
            next_src_max_len = max(src_max_len, data[i][0])
            next_tgt_max_len = max(tgt_max_len, data[i][1])
            if (next_src_num_tokens + next_tgt_num_tokens > self._max_tokens * 2) or \
                (next_src_num_tokens / (next_src_max_len * (len(batch) + 1)) < self._batch_efficiency) or \
                (next_tgt_num_tokens / (next_tgt_max_len * (len(batch) + 1)) < self._batch_efficiency):

                self._batches.append(batch)
                batch = []
                src_num_tokens = 0
                tgt_num_tokens = 0
                src_max_len = 0
                tgt_max_len = 0

            batch.append(data[i][2])
            src_num_tokens += data[i][0]
            tgt_num_tokens += data[i][1]
            src_max_len = max(src_max_len, data[i][0])
            tgt_max_len = max(tgt_max_len, data[i][1])

        if (not self._drop_last) and (len(batch) > 0):
            self._batches.append(batch)

    def __iter__(self) -> Iterator[List[int]]:
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._epoch + self._seed)
            indices = torch.randperm(len(self._batches), generator=g).tolist()
        else:
            indices = list(range(len(self._batches)))

        num_batches = len(self._batches)
        num_batches_per_replica = num_batches // (self._num_replicas * self._accum_steps) * self._accum_steps
        for i in range(num_batches_per_replica):
            index = i * self._num_replicas + self._rank
            yield self._batches[indices[index]]

    def __len__(self) -> int:
        num_batches = len(self._batches)
        return num_batches // (self._num_replicas * self._accum_steps) * self._accum_steps

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
