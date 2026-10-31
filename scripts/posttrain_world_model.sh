#!/usr/bin/env bash
# Fresh post-training from the existing perception checkpoint, on one GPU.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export LEAD_PROJECT_ROOT="$PWD"
export SCRATCH="${SCRATCH:-/localstorage/home/f20221129/scratch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORLD_LOGDIR="${WORLD_LOGDIR:-outputs/local_training/adapt_world_v1}"
WORLD_CHECKPOINT="${WORLD_CHECKPOINT:-outputs/ADAPT/B300/checkpoints/adapt/pretrain/model_0030.pth}"
LEAD_PYTHON="${LEAD_PYTHON:-python}"
export LEAD_TRAINING_CONFIG="model_type=adapt use_adapt_decoder=true use_planning_decoder=false \
use_carla_data=true use_navsim_data=false use_waymo_e2e_data=false \
load_file=$WORLD_CHECKPOINT continue_failed_training=false freeze_backbone=false \
logdir=$WORLD_LOGDIR description=adapt_world_v1 wandb_project_name=lead_posttrain \
adapt_navigation_conditioning=true adapt_world_model=true \
world_num_steps=8 world_num_modes=3 world_step_seconds=0.25 \
use_radars=true radar_detection=true use_radar_detection=true detect_boxes=true \
use_history_poses=true num_history_poses=5 \
epochs=${WORLD_EPOCHS:-60} batch_size=${WORLD_BATCH_SIZE:-128} lr=${WORLD_LR:-1e-4} \
use_scheduled_sampling=true scheduled_sampling_min_prob=0.0 \
scheduled_sampling_start_epoch=3 scheduled_sampling_warmup_epochs=30 scheduled_sampling_max_prob=1.0 \
use_cosine_annealing_with_restarts=false compile=true compile_scope=backbone \
cuda_prefetch=true prefetch_factor=${WORLD_PREFETCH_FACTOR:-4} assigned_cpu_cores=${WORLD_WORKERS:-32} \
gpu_color_augmentation=false upsample_perspective_logits=false"

if [[ "${1:-}" == "--check-config" ]]; then
    "$LEAD_PYTHON" - <<'PY'
import json
from pathlib import Path
from lead.training.training_utils import initialize_config
from lead.data_loader.world_model_targets import validate_world_target_config
config = initialize_config()
validate_world_target_config(config)
assert Path(config.load_file).is_file(), config.load_file
keys = ['load_file', 'logdir', 'epochs', 'batch_size', 'lr', 'adapt_navigation_conditioning',
        'adapt_world_model', 'world_num_steps', 'world_num_modes', 'world_step_seconds',
        'world_max_speed', 'use_radars', 'use_history_poses', 'scheduled_sampling_min_prob',
        'use_cosine_annealing_with_restarts', 'assigned_cpu_cores', 'prefetch_factor',
        'training_session_cache_path']
print(json.dumps({k: getattr(config, k) for k in keys}, indent=2))
print('Configuration checked; no training started.')
PY
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo 'Usage: bash scripts/posttrain_world_model.sh [--check-config]' >&2
    exit 2
fi
if [[ -e "$WORLD_LOGDIR" ]]; then
    echo "Choose a fresh WORLD_LOGDIR; $WORLD_LOGDIR already exists." >&2
    exit 1
fi
"$LEAD_PYTHON" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit('Select exactly one allocated GPU with CUDA_VISIBLE_DEVICES before launching.')
print('Training GPU:', torch.cuda.get_device_name(0))
PY
mkdir -p "$(dirname -- "$WORLD_LOGDIR")"
"$LEAD_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    --max_restarts=0 -m lead.training.train 2>&1 | tee "${WORLD_LOGDIR}.console.log"
