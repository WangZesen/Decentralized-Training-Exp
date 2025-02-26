import math
import torch
from functools import partial
from torch.utils.data import DataLoader
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

from src.data.dataset import WMTDataset
from src.conf import Config

from .sampler import DistributedTokenBatchSampler

def generate_positional_encoding(max_len: int, d_model: int):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe.squeeze()


def generate_concatenated_subsequent_mask(subseq_lengths: List[int]) -> torch.Tensor:
    max_length = sum(subseq_lengths)
    concatenated_subsequent_mask = torch.full((max_length, max_length), float('-inf'))
    start = 0
    for length in subseq_lengths:
        end = start + length
        subsequent_mask = torch.triu(torch.full((length, length), float('-inf'), dtype=torch.float32), diagonal=1)
        concatenated_subsequent_mask[start:end, start:end] = subsequent_mask
        start = end
    return concatenated_subsequent_mask[1:, 1:]


def generate_concatenated_memory_mask(src_lengths: List[int], tgt_lengths: List[int]) -> torch.Tensor:
    sum_src_length = sum(src_lengths)
    sum_tgt_length = sum(tgt_lengths)
    concatenated_memory_mask = torch.full((sum_tgt_length, sum_src_length), float('-inf'))
    src_start = 0
    tgt_start = 0
    for src_length, tgt_length in zip(src_lengths, tgt_lengths):
        concatenated_memory_mask[tgt_start:tgt_start + tgt_length, src_start:src_start + src_length] = 0.
        src_start += src_length
        tgt_start += tgt_length
    return concatenated_memory_mask[1:, :]


def generate_concatenated_positional_encoding(subseq_lengths: List[int], pe: torch.Tensor) -> torch.Tensor:
    max_length = sum(subseq_lengths)
    positional_encoding_index = torch.zeros((max_length, pe.size(1)), dtype=torch.int32)
    start = 0
    for length in subseq_lengths:
        end = start + length
        positional_encoding_index[start:end, :] = pe[:length, :]
        start = end
    return positional_encoding_index


def unbatched_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_token_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    src_batch, tgt_batch = zip(*batch)
    src_num_tokens = sum(src.size(0) for src in src_batch)
    tgt_num_tokens = sum(tgt.size(0) for tgt in tgt_batch)
    return pad_sequence(src_batch, batch_first=True, padding_value=pad_token_idx), \
        pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_idx), \
        src_num_tokens + tgt_num_tokens


def get_dataloaders(cfg: "Config",
                   train_dataset: "WMTDataset",
                   val_dataset: "WMTDataset",
                   test_dataset: "WMTDataset",
                   tokenizer: Tokenizer,
                   token_pad: str,
                   train_shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    assert cfg.train is not None
    
    train_sampler = DistributedTokenBatchSampler(
        dataset=train_dataset,
        seed=cfg.train.reproduce.seed,
        max_tokens=cfg.train.max_tokens_per_local_batch,
        shuffle=train_shuffle,
        batch_efficiency=cfg.data.batch_efficiency
    )
    val_sampler = DistributedTokenBatchSampler(
        dataset=val_dataset,
        seed=cfg.train.reproduce.seed,
        max_tokens=cfg.train.max_tokens_per_local_batch,
        shuffle=False,
        batch_efficiency=cfg.data.batch_efficiency
    )
    _unbatched_collate_fn = partial(unbatched_collate_fn, pad_token_idx=tokenizer.token_to_id(token_pad))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=_unbatched_collate_fn,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=_unbatched_collate_fn,
        pin_memory=True
    )
    return train_loader, val_loader


def get_dataloader(cfg: "Config",
                   dataset: "WMTDataset",
                   tokenizer: Tokenizer,
                   token_pad: str,
                   drop_last: bool = False,
                   shuffle: bool = False) -> DataLoader:
    assert cfg.train is not None

    _unbatched_collate_fn = partial(unbatched_collate_fn, pad_token_idx=tokenizer.token_to_id(token_pad))
    sampler = DistributedTokenBatchSampler(
        dataset=dataset,
        seed=cfg.train.reproduce.seed,
        max_tokens=cfg.train.max_tokens_per_local_batch,
        shuffle=shuffle,
        drop_last=drop_last,
        batch_efficiency=cfg.data.batch_efficiency
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_unbatched_collate_fn,
        pin_memory=True
    )
    return loader
