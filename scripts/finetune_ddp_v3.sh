#!/bin/bash

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use the 2 GPUs SLURM allocated (fresh --gres=gpu:2 gives indices 0,1).
export CUDA_VISIBLE_DEVICES=0,1
nproc_per_node=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

# batch_size must be divisible by (num_datasets * nproc_per_node) = 1*2 = 2.
# 32 → 16 per GPU, safe on H200 144GB. lr=1e-5 for finetuning.
# use_training_session_cache=false avoids the diskcache inode/slowdown spiral.
# load_file is the colleague's HF finetune checkpoint (has route + target_speed heads).
export LEAD_TRAINING_CONFIG="model_type=adapt use_adapt_decoder=true use_planning_decoder=false use_carla_data=true use_navsim_data=false use_history_poses=true use_training_session_cache=false epochs=5 batch_size=32 lr=1e-5 load_file=outputs/checkpoints/adapt/finetune/model_0001.pth continue_failed_training=false use_radars=false radar_detection=false use_radar_detection=false logdir=outputs/local_training/adapt_finetune_v3"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    lead/training/train.py
