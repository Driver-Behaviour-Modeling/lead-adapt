#!/bin/bash
#SBATCH --job-name=adapt_token_decoder
#SBATCH --partition=h100-full
#SBATCH --qos=professor-fullgpu-limited
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/%x_%j.log
# Fit the ADAPT token distribution used by the unruliness surprisal.
#
# Loads the navigation-conditioned radar checkpoint, re-initialises the token
# embedding and output projection for the body-frame codebook
# (kdisks_carla_body.pkl, 1234 codes), and trains only the AR transformer decoder
# with hard cross-entropy under pure teacher forcing. Everything else is frozen
# and kept in eval mode, so route/speed control is identical to the loaded model.
# The loaded checkpoint's config.json is the base config, so every changed key
# (codebook path, heading weight, vocab size) must be overridden here.
#
# bf16 has no GradScaler, so NaN steps are skipped in train.py and gradients are
# clipped; every epoch's checkpoint is kept so one can be picked by held-out NLL.
# v1 (6 epochs, no guard) went NaN at step 18,078 and lost all good checkpoints.
#
# Run from the repo root, either directly or queued:
#   bash scripts/posttrain_token_decoder.sh
#   sbatch scripts/posttrain_token_decoder.sh

if [ "${CONDA_DEFAULT_ENV:-}" != "lead" ]; then
    eval "$(conda shell.bash hook)"
    conda activate lead
fi

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Single H100 only: the second GPU on this box is partitioned for MIG.
nproc_per_node=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

RUN_NAME=${RUN_NAME:-adapt_token_decoder_v2}

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/$RUN_NAME description=$RUN_NAME model_type=adapt use_adapt_decoder=true use_planning_decoder=false use_carla_data=true use_navsim_data=false use_history_poses=true use_radars=true adapt_navigation_conditioning=true adapt_train_token_decoder_only=true adapt_waypoints_from_tokens=true use_scheduled_sampling=false kdisks_vocab_path=lead/adapt/codebooks/kdisks_carla_body.pkl kdisks_heading_weight=2.6719030517676834 kinematic_vocab_size=1234 epochs=3 batch_size=128 lr=1e-4 use_cosine_annealing_with_restarts=false grad_clip_norm=1.0 keep_all_epoch_checkpoints=true cuda_prefetch=true assigned_cpu_cores=32 prefetch_factor=4 wandb_project_name=lead_posttrain load_file=outputs/local_training/adapt_radar_navcond_v1_retry1/model_0014.pth continue_failed_training=false $EXTRA_CONFIG"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    lead/training/train.py
