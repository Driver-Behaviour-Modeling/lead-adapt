#!/usr/bin/bash
# Official LEAD Bench2Drive evaluation for the ADAPT finetune checkpoint.
# Run from the repo root:  bash slurm/experiments/adapt_eval/b2d_0.sh
#
# EXPERIMENT_NAME = adapt_eval (parent dir), SCRIPT_NAME = b2d_0, SEED = 0.
# This sources the harness, points it at the checkpoint, and launches the
# official evaluate_bench2drive pipeline (generator + job pool + merge) inside
# a detached `screen` session named after EXPERIMENT_RUN_ID.

source slurm/init.sh

# The generated per-route scripts call random_free_port.sh / clean_carla.sh
# without a path, so scripts/ must be on PATH. init.sh doesn't add it.
export PATH="$LEAD_PROJECT_ROOT/scripts:$PATH"

export CHECKPOINT_DIR=outputs/checkpoints/adapt_finetune_v5

evaluate_bench2drive
