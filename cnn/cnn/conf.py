import os
import argparse
import tomllib
import hashlib
from typing import Optional, Tuple, Literal, Final, Union
from pydantic import BaseModel, Field, computed_field, ConfigDict

PROJECT_DIR: Final[str] = os.path.relpath(os.path.join(os.path.dirname(__file__), '..'), '.')

class Data(BaseModel):
    data_dir: str = Field(default='./data/Imagenet')
    sharded_data_dir: str = Field(default='./data/Imagenet-sharded')
    num_classes: int = Field(default=1000)

class Preprocess(BaseModel):
    preload_local: bool = Field(default=False)
    interpolation: str = Field(default='bilinear')
    train_crop_size: int = Field(default=224)
    val_image_size: int = Field(default=256)
    val_crop_size: int = Field(default=224)

class AdamConfig(BaseModel):
    name: Literal['adam'] = Field(default='adam')
    weight_decay: float = Field(default=1./32768)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    epsilon: float = Field(default=1e-8)

class SGDConfig(BaseModel):
    name: Literal['sgd'] = Field(default='sgd')
    weight_decay: float = Field(default=1./32768)
    momentum: float = Field(default=0.9)

class AdamWConfig(BaseModel):
    name: Literal['adamw'] = Field(default='adamw')
    weight_decay: float = Field(default=1./32768)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    epsilon: float = Field(default=1e-8)

class AccumAdamConfig(BaseModel):
    name: Literal['accumadam'] = Field(default='accumadam')
    weight_decay: float = Field(default=1./32768)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    epsilon: float = Field(default=1e-8)
    accum_iter: int = Field(default=4)

ALL_OPTIMS = Union[AdamConfig, SGDConfig, AccumAdamConfig]

class LRScheduler(BaseModel):
    name: str = Field(default='cosine')
    warmup_epochs: int = Field(default=5)
    warmup_decay: float = Field(default=0.01)

class Reproduce(BaseModel):
    seed: int = Field(default=810975)

class Log(BaseModel):
    log_freq: int = Field(default=100)
    wandb_on: bool = Field(default=True)
    wandb_project: str = Field(default='reproduce_resnet')
    checkpoint_freq: int = Field(default=45)

    @computed_field
    @property
    def slurm_id(self) -> str:
        return os.environ.get('SLURM_JOB_ID', '0')

    @computed_field
    @property
    def log_dir(self) -> str:
        return os.path.join(PROJECT_DIR, 'log', self.slurm_id)


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

    @computed_field(repr=False)
    @property
    def node_list(self) -> str:
        return os.environ.get('SLURM_NODELIST', 'localhost')


class Train(BaseModel):
    backend: Literal['pytorchddp', 'decentdp'] = Field(default='pytorchddp')
    topology: Literal['complete', 'alternating-exp-ring', 'ring', 'exp'] = Field(default='complete')
    batch_size: int = Field(default=1024)
    max_epochs: int = Field(default=90)
    lr: float = Field(default=0.001)
    label_smoothing: float = Field(default=0.1)
    grad_clip_norm: float = Field(default=0.0)
    checkpoint_dir: Optional[str] = Field(default=None)
    arch: str = Field(default='resnet50')
    use_amp: bool = Field(default=True)
    preprocess: Preprocess = Field(default_factory=Preprocess)
    optim: ALL_OPTIMS = Field(default_factory=AdamConfig, discriminator='name')
    lr_scheduler: LRScheduler = Field(default_factory=LRScheduler)
    reproduce: Reproduce = Field(default_factory=Reproduce)
    log: Log = Field(default_factory=Log)
    network: Network = Field(default_factory=Network)

    @computed_field(repr=False)
    @property
    def batch_size_per_local_batch(self) -> int:
        return self.batch_size // self.network.world_size


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    data: Data = Field(default_factory=Data)
    train: Train = Field(default_factory=Train)


def parse_config() -> Config:
    def _load_toml(config_dir):
        with open(config_dir, 'rb') as f:
            config = tomllib.load(f)
        return config

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-cfg', type=str, required=True)
    parser.add_argument('--train-cfg', type=str, required=True)
    args = parser.parse_args()
    cfg = {
        'data': _load_toml(args.data_cfg),
        'train': _load_toml(args.train_cfg)
    }
    return Config(**cfg) # type: ignore