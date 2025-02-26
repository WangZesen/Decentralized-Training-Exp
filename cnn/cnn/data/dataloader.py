import os
import torch
import torch.distributed as dist
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
from nvidia.dali.ops.readers import File
from nvidia.dali.ops.decoders import Image
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.ops import RandomResizedCrop, CropMirrorNormalize, Resize
from nvidia.dali.ops.random import CoinFlip
from cnn.conf import Config


class DaliImageNetTrainPipeline(Pipeline):
    def __init__(self,
                 batch_size: int,
                 data_dir: str,
                 interpolation: str,
                 crop: int,
                 seed: int,
                 dont_use_mmap: bool):
        super().__init__(batch_size,
                         num_threads=4,
                         device_id=0,
                         seed=seed,
                         set_affinity=True)
        interpolation_type = {
            "bicubic": types.DALIInterpType.INTERP_CUBIC,
            "bilinear": types.DALIInterpType.INTERP_LINEAR,
            "triangular": types.DALIInterpType.INTERP_TRIANGULAR,
        }[interpolation]

        if dist.is_initialized():
            shard_id = dist.get_rank()
            num_shards = dist.get_world_size()
        else:
            shard_id = 0
            num_shards = 1
        
        self.input = File(
            file_root=data_dir,
            read_ahead=True,
            shuffle_after_epoch=True,
            shard_id=shard_id,
            num_shards=num_shards,
            initial_fill=20000,
            seed=seed,
            dont_use_mmap=dont_use_mmap
        )

        self.decode = Image(
            device="mixed",
            output_type=types.DALIImageType.RGB,
            memory_stats=True,
        )

        self.res = RandomResizedCrop(
            device='gpu',
            size=(crop, crop),
            interp_type=interpolation_type,
            random_aspect_ratio=[0.75, 4.0/3.0],
            random_area=[0.08, 1.0],
            num_attempts=100,
            antialias=False,
            seed=seed,
        )
        self.coin = CoinFlip(probability=0.5)
        self.cmnp = CropMirrorNormalize(
            device='gpu',
            dtype=types.DALIDataType.FLOAT,
            output_layout='CHW',
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
    
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader") # type: ignore
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.cmnp(images, mirror=self.coin())
        return [images, self.labels]


class DaliImageNetValPipeline(Pipeline):
    def __init__(self,
                 batch_size: int,
                 data_dir: str,
                 interpolation: str,
                 resize: int,
                 crop: int,
                 dont_use_mmap: bool):
        super().__init__(batch_size, num_threads=2, device_id=0)
        interpolation_type = {
            "bicubic": types.DALIInterpType.INTERP_CUBIC,
            "bilinear": types.DALIInterpType.INTERP_LINEAR,
            "triangular": types.DALIInterpType.INTERP_TRIANGULAR,
        }[interpolation]

        if dist.is_initialized():
            shard_id = dist.get_rank()
            num_shards = dist.get_world_size()
        else:
            shard_id = 0
            num_shards = 1
        
        self.input = File(
            file_root=data_dir,
            random_shuffle=False,
            shard_id=shard_id,
            num_shards=num_shards,
            dont_use_mmap=dont_use_mmap,
            pad_last_batch=False,
        )

        self.decode = Image(
            device="mixed",
            output_type=types.DALIImageType.RGB,
            memory_stats=True,
        )
        self.res = Resize(
            device='gpu',
            resize_shorter=resize,
            interp_type=interpolation_type,
            antialias=False
        )
        self.cmnp = CropMirrorNormalize(
            device='gpu',
            dtype=types.DALIDataType.FLOAT,
            output_layout='CHW',
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader") # type: ignore
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.cmnp(images.gpu()) # type: ignore
        return [images, self.labels]


class DALIWrapper(object):
    @staticmethod
    def gen_wrapper(dalipipeline: DALIClassificationIterator):
        for data in dalipipeline:
            input = data[0]["data"].contiguous(memory_format=torch.contiguous_format)
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline):
        self.dalipipeline = dalipipeline

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline)


def get_dali_train_loader(cfg: Config):
    train_data_dir = os.path.join(cfg.data.data_dir, "train")
    pipe = DaliImageNetTrainPipeline(
        batch_size=cfg.train.batch_size_per_local_batch,
        data_dir=train_data_dir,
        interpolation=cfg.train.preprocess.interpolation,
        crop=cfg.train.preprocess.train_crop_size,
        seed=cfg.train.reproduce.seed,
        dont_use_mmap=not cfg.train.preprocess.preload_local
    )
    pipe.build()
    train_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return DALIWrapper(train_loader), int(pipe.epoch_size('Reader') / cfg.train.batch_size_per_local_batch / world_size) # type: ignore


def get_dali_valid_loader(cfg: Config):
    train_data_dir = os.path.join(cfg.data.data_dir, "val")
    pipe = DaliImageNetValPipeline(
        batch_size=cfg.train.batch_size_per_local_batch,
        data_dir=train_data_dir,
        interpolation=cfg.train.preprocess.interpolation,
        resize=cfg.train.preprocess.val_image_size,
        crop=cfg.train.preprocess.val_crop_size,
        dont_use_mmap=not cfg.train.preprocess.preload_local
    )
    pipe.build()
    valid_loader = DALIClassificationIterator(
        pipe,
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL
    )
    return DALIWrapper(valid_loader)

