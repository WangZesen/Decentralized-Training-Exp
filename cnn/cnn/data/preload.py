import os
import subprocess
import time
import torch.distributed as dist
from loguru import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..conf import Config

def preload_to_local(cfg: 'Config'):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    tmp_dir = os.environ['TMPDIR']
    num_shards = int(subprocess.check_output(f'ls {cfg.data.sharded_data_dir} | grep zip | wc -l', shell=True).decode().strip())

    if os.path.exists(os.path.join(tmp_dir, "train")):
        logger.debug(f'[rank {rank}] Preload Complete "data_dir" is changed to: {tmp_dir}')
        cfg.data.data_dir = tmp_dir
        return
    
    tmp_dir_capacity = int(subprocess.check_output(f"df {tmp_dir} | tail -n 1 | awk '{{print $4}}'", shell=True).decode().strip())
    tmp_dir_capacity_gb = tmp_dir_capacity

    shard_sizes = subprocess.check_output(f"du -s {cfg.data.sharded_data_dir}/*.zip | awk '{{print $1}}'", shell=True).decode().strip().split('\n')
    max_shard_size = max([int(size) for size in shard_sizes])
    
    shards_left = num_shards
    while shards_left > 0:
        num_jobs = min(shards_left, local_world_size)
        num_jobs = (min(num_shards - shards_left + num_jobs * 2, tmp_dir_capacity // max_shard_size) - (num_shards - shards_left)) // 2
        num_jobs = max(num_jobs, 1)
        for i in range(num_jobs):
            if local_rank == i:
                shard_index = num_shards - shards_left + i
                logger.debug(f'[rank {rank}] Preload shard {shard_index} to local storage')
                start_time = time.time()
                subprocess.check_call(f'cp {cfg.data.sharded_data_dir}/shard{shard_index}.zip {tmp_dir}', shell=True)
                logger.debug(f'[rank {rank}] Preload done. Elapsed time: {time.time() - start_time:.2f} s')
                logger.debug(f'[rank {rank}] Unzip shard {shard_index}')
                start_time = time.time()
                subprocess.check_call(f"mkdir {tmp_dir}/shard{shard_index}", shell=True)
                subprocess.check_call(f"unzip -q {tmp_dir}/shard{shard_index}.zip -d {tmp_dir}/shard{shard_index}", shell=True)
                subprocess.check_call(f"rm {tmp_dir}/shard{shard_index}.zip", shell=True)
                logger.debug(f'[rank {rank}] Unzip done. Elapsed time: {time.time() - start_time:.2f} s')
        shards_left -= num_jobs
        dist.barrier()

    if local_rank == 0:
        logger.debug(f'[rank {rank}] Rearrange data in local storage')
        start_time = time.time()
        local_folder = tmp_dir
        os.makedirs(os.path.join(local_folder, "train"))
        os.makedirs(os.path.join(local_folder, "val"))
        for i in range(num_shards):
            subprocess.call(f"mv {tmp_dir}/shard{i}/*/train/* {tmp_dir}/train/", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.call(f"mv {tmp_dir}/shard{i}/*/val/* {tmp_dir}/val/", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.debug(f'Done [rank {rank}]. Elapsed time: {time.time() - start_time:.2f} s')

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info(f'Preload Complete. "data_dir" is changed to: {tmp_dir}')

    cfg.data.data_dir = tmp_dir

