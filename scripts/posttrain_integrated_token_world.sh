#!/usr/bin/env bash
# Fine-tune one combined model; use separate seeds/runs for ensemble members.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-/localstorage/home/f20221129/miniconda3/envs/lead/bin/python}
RUN_NAME=${RUN_NAME:-adapt_token_world_v1}
INIT_CHECKPOINT=${RESUME_CHECKPOINT:-${INIT_CHECKPOINT:-}}
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
    printf '%s\n' 'Set INIT_CHECKPOINT to the prepared weights, or RESUME_CHECKPOINT to a saved epoch.' >&2
    exit 1
fi
CONTINUE_FAILED_TRAINING=false
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then
    CONTINUE_FAILED_TRAINING=true
elif [[ -f outputs/local_training/$RUN_NAME/config.json && ${DRY_RUN:-0} != 1 ]]; then
    printf '%s\n' "Run $RUN_NAME already exists; choose a new RUN_NAME or set RESUME_CHECKPOINT." >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LEAD_TRAINING_CONFIG="load_file=$INIT_CHECKPOINT continue_failed_training=$CONTINUE_FAILED_TRAINING \
logdir=outputs/local_training/$RUN_NAME description=$RUN_NAME \
model_type=adapt use_adapt_decoder=true use_planning_decoder=false \
adapt_navigation_conditioning=true adapt_token_navigation_conditioning=true adapt_world_model=true \
adapt_history_free=true adapt_waypoints_from_tokens=true adapt_rolling_tokenization=true \
adapt_rolling_causal_first=true adapt_train_token_decoder_only=false \
use_scheduled_sampling=false use_soft_ce=false \
kdisks_vocab_path=lead/adapt/codebooks/kdisks_carla_body.pkl kinematic_vocab_size=1234 \
kdisks_heading_weight=2.6719030517676834 \
seed=${SEED:-0} epochs=${EPOCHS:-5} lr=${LR:-2e-5} batch_size=${BATCH_SIZE:-128} \
grad_clip_norm=1.0 keep_all_epoch_checkpoints=true freeze_backbone=false \
assigned_cpu_cores=${CPU_CORES:-32} prefetch_factor=${PREFETCH_FACTOR:-4} \
cuda_prefetch=true use_cosine_annealing_with_restarts=false ${EXTRA_CONFIG:-}"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%s\n' "$LEAD_TRAINING_CONFIG"
    exit 0
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node=1 --max_restarts=0 --module lead.training.train
