#!/bin/bash

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Single H100 only: the second GPU on this box is partitioned for MIG and is
# not usable for full DDP training. Do NOT hardcode CUDA_VISIBLE_DEVICES here —
# when launched via SLURM (srun --gres=gpu:1) the scheduler already masks the
# allocated device to index 0, and overriding it can point at the MIG slice.
nproc_per_node=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

# batch_size must be divisible by (num_datasets * nproc_per_node) — here 1*1 = 1.
export LEAD_TRAINING_CONFIG="model_type=adapt use_adapt_decoder=true use_planning_decoder=false use_carla_data=true use_navsim_data=false use_history_poses=true use_scheduled_sampling=true epochs=40 batch_size=128 load_file=outputs/ADAPT/B300/checkpoints/adapt/pretrain/model_0030.pth continue_failed_training=false use_radars=false radar_detection=false use_radar_detection=false"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    lead/training/train.py
