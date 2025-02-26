import os
import time
import psutil
import torch
import struct
from torch.utils.data import Dataset
import torch.distributed as dist
from typing import List, Tuple
from loguru import logger
from src.conf import Config


class WMTDataset(Dataset):
    def __init__(self,
                 cfg: "Config",
                 tokenizer_dir: str,
                 split: str = "train"):

        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        
        if rank == 0:
            logger.info(f"{'='*30} Loading {split} data {'='*30}")

        src_file = os.path.join(tokenizer_dir, f"{split}.{cfg.data.src_lang}")
        tgt_file = os.path.join(tokenizer_dir, f"{split}.{cfg.data.tgt_lang}")
        
        start_time = time.time()
        self._src_start_pos, self._src_len, self._src_raw = self.load_from_bin(src_file)
        self._tgt_start_pos, self._tgt_len, self._tgt_raw = self.load_from_bin(tgt_file)

        total_src_tokens = sum(self._src_len)
        total_tgt_tokens = sum(self._tgt_len)

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        elapsed_time = time.time() - start_time
        if rank == 0:
            logger.info(f"Total number of data pairs: {len(self._src_len)}")
            logger.info(f"Total number of tokens in source examples: {total_src_tokens}")
            logger.info(f"Total number of tokens in target examples: {total_tgt_tokens}")
            logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB")
            logger.info(f"Time elapsed for loading the data: {elapsed_time:.2f} seconds")

    def load_from_bin(self, data_dir: str):
        with open(data_dir, 'rb') as f:
            data = f.read()
        
        start_pos = []
        lengths = []

        pos = 0
        while pos < len(data) - 1:
            length = struct.unpack(">i", data[pos:pos+4])[0]
            lengths.append(length)
            pos += 4
            start_pos.append(pos)
            pos += length * 2

        return start_pos, lengths, data

    def __len__(self):
        return len(self._src_len)

    def __getitem__(self, i: int):
        src_sample = list(struct.unpack('H' * self._src_len[i], self._src_raw[self._src_start_pos[i]:self._src_start_pos[i] + 2*self._src_len[i]]))
        tgt_sample = list(struct.unpack('H' * self._tgt_len[i], self._tgt_raw[self._tgt_start_pos[i]:self._tgt_start_pos[i] + 2*self._tgt_len[i]]))
        # pad at the batch level.
        return torch.tensor(src_sample), torch.tensor(tgt_sample)

    def get_token_stats(self) -> Tuple[List[int], List[int]]:
        return self._src_len, self._tgt_len


def get_datasets(cfg: "Config", tokenizer_dir: str):
    train_ds, valid_ds, test_ds =  WMTDataset(cfg, tokenizer_dir, split="train"), \
        WMTDataset(cfg, tokenizer_dir, split="valid"), \
        WMTDataset(cfg, tokenizer_dir, split="test")
    return train_ds, valid_ds, test_ds


def get_dataset(cfg: "Config", tokenizer_dir: str, split: str = "test"):
    test_ds = WMTDataset(cfg, tokenizer_dir, split=split)
    return test_ds

