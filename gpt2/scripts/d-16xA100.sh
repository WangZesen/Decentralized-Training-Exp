#!/usr/bin/bash
#SBATCH -J D-GPT2-Pretrain
#SBATCH --nodes=4
#SBATCH --gpus-per-node=A100:4
#SBATCH -t 35:00:00
#SBATCH --switches=1
#SBATCH -o log/%A/log.out
#SBATCH -e log/%A/err.out

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

mkdir -p log/$SLURM_JOB_ID

srun $1 --nnodes=4 --nproc_per_node=4 --rdzv-backend=c10d --rdzv-id=$RANDOM --rdzv-endpoint=$head_node_ip:28052 \
    -m gpt.decent_train --config $2
