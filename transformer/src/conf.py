import os
import argparse
import tomllib
import hashlib
from typing import Optional, Tuple, Literal, Union
from pydantic import BaseModel, Field, computed_field

PROJECT_DIR = os.path.relpath(os.path.join(os.path.dirname(__file__), '..'), '.')
class SPECIAL_TOKENS:
    PAD = '[PAD]'
    UNK = '[UNK]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    ALL = [PAD, UNK, SOS, EOS]


class Tokenizer(BaseModel):
    model: str = Field(default='bpe')
    vocab_size: int = Field(default=37120)
    min_freq: int = Field(default=2)


class Data(BaseModel, frozen=True):
    data_dir: str = Field(default=os.path.join(PROJECT_DIR, 'data', 'wmt14_en_de'))
    src_lang: str = Field(default='en')
    tgt_lang: str = Field(default='de')
    truncate: int = Field(default=156)
    tokenizer: Tokenizer = Field(default_factory=Tokenizer)
    batch_efficiency: float = Field(default=0.35)

    @computed_field(repr=False)
    @property
    def tag(self) -> str:
        return hashlib.md5(str(self.__repr__()).encode()).hexdigest()[:10]

    @computed_field(repr=False)
    @property
    def output_dir(self) -> str:
        return os.path.join(self.data_dir, self.tag)


class Reproduce(BaseModel):
    seed: int = Field(default=810975)


class Network(BaseModel):
    @computed_field(repr=False)
    @property
    def world_size(self) -> int:
        return int(os.environ.get('WORLD_SIZE', '1'))

    @computed_field(repr=False)
    @property
    def rank(self) -> int:
        return int(os.environ.get('RANK', '0'))
    
    @computed_field(repr=False)
    @property
    def local_rank(self) -> int:
        return int(os.environ.get('LOCAL_RANK', '0'))
    
    @computed_field(repr=False)
    @property
    def local_world_size(self) -> int:
        return int(os.environ.get('LOCAL_WORLD_SIZE', '1'))


class Model(BaseModel):
    arch: str = Field(default='transformer')
    d_model: int = Field(default=512)
    num_heads: int = Field(default=8)
    num_layers: int = Field(default=6)
    dim_feedforward: int = Field(default=2048)
    dropout: float = Field(default=0.1)


class AdamConfig(BaseModel):
    name: Literal['adam'] = Field(default='adam')
    lr: float = Field(default=0.0007)
    betas: Tuple[float, float] = Field(default=(0.9, 0.98))
    eps: float = Field(default=1e-9)

class AccumAdamConfig(BaseModel):
    name: Literal['accumadam'] = Field(default='accumadam')
    lr: float = Field(default=0.0007)
    betas: Tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-9)
    accum_iter: int = Field(default=8)

ALL_OPTIMIZERS = Union[AdamConfig, AccumAdamConfig]

class LRScheduler(BaseModel):
    type: str = Field(default='inverse_sqrt')
    warmup_steps: int = Field(default=4000)
    warmup_decay: float = Field(default=0.01)

class Log(BaseModel):
    log_freq: int = Field(default=250)
    wandb_on: bool = Field(default=False)
    wandb_project: str = Field(default='Reproduce-Transformer')
    checkpoint_freq: int = Field(default=2)
    
    @computed_field
    @property
    def slurm_id(self) -> str:
        return os.environ.get('SLURM_JOB_ID', '0')

    @computed_field
    @property
    def log_dir(self) -> str:
        return os.path.join(PROJECT_DIR, 'log', self.slurm_id)


class Train(BaseModel):
    backend: Literal['pytorchddp', 'decentdp'] = Field(default='pytorchddp')
    topology: Literal['complete', 'ring', 'alternating-exp-ring', 'exp'] = Field(default='complete')
    max_tokens_per_batch: int = Field(default=25000)
    label_smoothing: float = Field(default=0.1)
    checkpoint_dir: Optional[str] = None
    max_epochs: int = Field(default=20)
    use_amp: bool = Field(default=False)
    reproduce: Reproduce = Field(default_factory=Reproduce)
    network: Network = Field(default_factory=Network)
    model: Model = Field(default_factory=Model)
    optim: ALL_OPTIMIZERS = Field(default_factory=AdamConfig)
    lr_scheduler: LRScheduler = Field(default_factory=LRScheduler)
    log: Log = Field(default_factory=Log)

    @computed_field(repr=False)
    @property
    def max_tokens_per_local_batch(self) -> int:
        return self.max_tokens_per_batch // self.network.world_size

    @max_tokens_per_local_batch.setter
    def max_tokens_per_local_batch(self, value: int):
        self.max_tokens_per_batch = value * self.network.world_size


class Eval(BaseModel):
    exp_dir: str = Field(...)
    beam_size: int = Field(default=4)
    length_penalty: float = Field(default=0.6)
    tolerance: int = Field(default=50)


class Config(BaseModel):
    data: Data = Field(default_factory=Data)
    train: Optional[Train] = None
    eval: Optional[Eval] = None


def parse_config(load_data_cfg: bool = True,
                 load_train_cfg: bool = True,
                 load_eval_cfg: bool = True) -> Config:
    def _load_toml(config_dir):
        with open(config_dir, 'rb') as f:
            config = tomllib.load(f)
        return config

    parser = argparse.ArgumentParser()
    if load_eval_cfg:
        parser.add_argument('--eval-cfg', type=str, required=True)
        args = parser.parse_args()
        _eval_cfg = _load_toml(args.eval_cfg)
        args.data_cfg = os.path.join(_eval_cfg['exp_dir'], 'data_cfg.dump.toml')
        args.train_cfg = os.path.join(_eval_cfg['exp_dir'], 'train_cfg.dump.toml')
    else:
        if load_data_cfg:
            parser.add_argument('--data-cfg', type=str, required=True)
        if load_train_cfg:
            parser.add_argument('--train-cfg', type=str, required=True)
        args = parser.parse_args()
    cfg = {}
    if load_data_cfg:
        cfg['data'] = _load_toml(args.data_cfg)
    if load_train_cfg:
        cfg['train'] = _load_toml(args.train_cfg)
    if load_eval_cfg:
        cfg['eval'] = _load_toml(args.eval_cfg)

    return Config(**cfg)
