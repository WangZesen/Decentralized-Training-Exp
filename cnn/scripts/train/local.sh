export LOGLEVEL=INFO
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export JOB_ID=$(date +%Y%m%d_%H%M%S)
export SLURM_JOB_ID=$JOB_ID
export NUM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o , | wc -l)+1))
export TMPDIR=/tmp

mkdir -p log/$JOB_ID
cp $2 log/$JOB_ID/data_cfg.toml
cp $3 log/$JOB_ID/train_cfg.toml

echo "Job ID: $JOB_ID"
echo "Log Directory: log/$JOB_ID"

$1 --standalone \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    -m cnn.train --data-cfg log/$JOB_ID/data_cfg.toml --train-cfg log/$JOB_ID/train_cfg.toml \
    > log/$JOB_ID/log.out 2> log/$JOB_ID/err.out
