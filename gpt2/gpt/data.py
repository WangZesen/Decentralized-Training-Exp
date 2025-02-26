import os
import torch
import numpy as np
from loguru import logger
from gpt.config import Config
from typing import Literal, Tuple, List

ALL_SPLITS: List[Literal['train', 'val']] = ['train', 'val']

class OpenWebTextDataset:
    def __init__(self, cfg: Config):
        assert os.path.exists(os.path.join(cfg.data_dir, 'train.bin')), 'OpenWebText train dataset not found'
        assert os.path.exists(os.path.join(cfg.data_dir, 'val.bin')), 'OpenWebText val dataset not found'
        self._cfg = cfg
        if self._cfg.in_memory:
            self._in_memory = True
            self._load_to_memory()
            # report memory usage
            train_size = self._train_data.nbytes / 1_000_000
            val_size = self._val_data.nbytes / 1_000_000
            if cfg.train.network.rank == 0:
                logger.debug(f'Loaded OpenWebText dataset to memory. Train: {train_size:.2f}MB, Val: {val_size:.2f}MB')
                logger.debug(f'Train: {self._train_data.shape}, Val: {self._val_data.shape}')
        else:
            self._in_memory = False
            if cfg.train.network.rank == 0:
                logger.debug(f'Insufficient memory to load OpenWebText dataset to memory. Use memmap instead.')
        self._rng = np.random.default_rng(cfg.train.seed)


    def _load_to_memory(self):
        data = np.fromfile(os.path.join(self._cfg.data_dir, 'train.bin'), dtype=np.uint16)
        self._train_data = np.array(data)
        data = np.fromfile(os.path.join(self._cfg.data_dir, 'val.bin'), dtype=np.uint16)
        self._val_data = np.array(data)


    def _get_data_and_indices(self, split: Literal['train', 'val']) -> Tuple[np.ndarray, np.ndarray]:
        if self._in_memory:
            data = self._train_data if split == 'train' else self._val_data
        else:
            data = np.memmap(os.path.join(self._cfg.data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        indices = self._rng.integers(0,
                                    len(data) - self._cfg.train.context_length,
                                    size=(self._cfg.train.network.world_size, self._cfg.train.local_batch_size))
        return data, indices


    def get_batch(self, split: Literal['train', 'val'] = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        data, indices = self._get_data_and_indices(split)
        rank = self._cfg.train.network.rank
        x = torch.stack([torch.from_numpy(data[i:i+self._cfg.train.context_length].astype(np.int64)) for i in indices[rank]])
        y = torch.stack([torch.from_numpy(data[i+1:i+self._cfg.train.context_length+1].astype(np.int64)) for i in indices[rank]])
        x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        return x, y


    def sync_rng_state(self, current_step: int):
        # initial evaluation
        for split in ALL_SPLITS:
            for _ in range(self._cfg.log.eval_steps):
                _, _ = self._get_data_and_indices(split)

        # start training
        n_loop = round(current_step / self._cfg.log.eval_interval)
        for i in range(n_loop):
            for _ in range(self._cfg.log.eval_interval):
                _, _ = self._get_data_and_indices('train')

            # skip the last evaluation as it will be done at the start of the training
            if i < n_loop - 1:
                for split in ALL_SPLITS:
                    for _ in range(self._cfg.log.eval_steps):
                        _, _ = self._get_data_and_indices(split)
