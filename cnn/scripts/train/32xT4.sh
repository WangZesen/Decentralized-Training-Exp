#!/usr/bin/bash
#SBATCH -J ImageCls
#SBATCH --nodes=4
#SBATCH --gpus-per-node=T4:8
#SBATCH -t 12:00:00
#SBATCH --switches=1
#SBATCH --gres=ptmpdir:1
#SBATCH -o log/%A/log.out
#SBATCH -e log/%A/err.out

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

mkdir -p log/$SLURM_JOB_ID
cp $2 log/$SLURM_JOB_ID/data_cfg.toml
cp $3 log/$SLURM_JOB_ID/train_cfg.toml

srun $1 \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv-backend=c10d \
    --rdzv-id=$RANDOM \
    --rdzv-endpoint=$head_node_ip:28052 \
    src/train.py --data-cfg log/$SLURM_JOB_ID/data_cfg.toml --train-cfg log/$SLURM_JOB_ID/train_cfg.toml

