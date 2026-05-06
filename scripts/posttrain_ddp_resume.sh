#!/bin/bash

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

POSTTRAIN_DIR="outputs/local_training/posttrain,load_file=outputs/local_training/pretrain/model_0009.pth,use_planning_decoder=True"
export LEAD_TRAINING_CONFIG="logdir=$POSTTRAIN_DIR \
load_file=$POSTTRAIN_DIR/model_0010.pth \
use_planning_decoder=true \
continue_failed_training=true"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    lead/training/train.py

