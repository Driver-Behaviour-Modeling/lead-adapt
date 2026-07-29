#!/usr/bin/bash
# Split the Bench2Drive routes into two halves and run TWO fully independent
# single-GPU evaluations (GPU 0 and GPU 1) in parallel, then aggregate all
# results with the OFFICIAL merge. If one GPU stalls it does not affect the
# other. This replaces the shared-scheduler launcher.
#
# Usage:  eval_b2d_split_2gpu.sh <checkpoint_dir> [route_glob]
# Env:    DEBUG=1|full, MAX_ATTEMPTS, ROUTE_WALLCLOCK, GPU_ADAPTERS (see one_gpu).

set -u
export LEAD_PROJECT_ROOT="${LEAD_PROJECT_ROOT:-/home/divyanshu/lead-adapt}"
cd "$LEAD_PROJECT_ROOT"

CHECKPOINT_DIR="${1:?usage: eval_b2d_split_2gpu.sh <checkpoint_dir> [route_glob]}"
ROUTE_GLOB="${2:-data/benchmark_routes/bench2drive/*.xml}"

# Single-instance guard.
LOCKFILE="/tmp/eval_b2d_split_2gpu.lock"
exec 9>"$LOCKFILE"
flock -n 9 || { echo "ERROR: another eval_b2d_split_2gpu.sh is running (lock $LOCKFILE)."; exit 1; }

CKPT_NAME=$(basename "$CHECKPOINT_DIR")
OUT="$LEAD_PROJECT_ROOT/outputs/evaluation/bench2drive/${CKPT_NAME}"
mkdir -p "$OUT/eval" "$OUT/logs"

# Deterministic split: even-indexed routes -> GPU0 list, odd -> GPU1 list.
# (Alternating keeps each half a representative mix of scenario types.)
mapfile -t ALL < <(ls -1 $ROUTE_GLOB 2>/dev/null | sort)
[ "${#ALL[@]}" -eq 0 ] && { echo "No routes matched $ROUTE_GLOB"; exit 1; }
L0="$OUT/routes_gpu0.txt"; L1="$OUT/routes_gpu1.txt"; : > "$L0"; : > "$L1"
for idx in "${!ALL[@]}"; do
    if [ $((idx % 2)) -eq 0 ]; then echo "${ALL[$idx]}" >> "$L0"; else echo "${ALL[$idx]}" >> "$L1"; fi
done
echo "Total routes: ${#ALL[@]}  -> GPU0: $(wc -l < "$L0")  GPU1: $(wc -l < "$L1")"
echo "Output: $OUT"

STAMP=$(date +%y%m%d_%H%M%S)
LOG0="$OUT/gpu0_${STAMP}.log"; LOG1="$OUT/gpu1_${STAMP}.log"

# Launch the two independent single-GPU runs.
DEBUG="${DEBUG:-0}" bash scripts/eval_b2d_one_gpu.sh "$CHECKPOINT_DIR" 0 "$L0" > "$LOG0" 2>&1 &
P0=$!
sleep 20   # stagger CARLA boots so the two don't hammer the driver simultaneously
DEBUG="${DEBUG:-0}" bash scripts/eval_b2d_one_gpu.sh "$CHECKPOINT_DIR" 1 "$L1" > "$LOG1" 2>&1 &
P1=$!

echo "GPU0 run pid $P0 -> $LOG0"
echo "GPU1 run pid $P1 -> $LOG1"
echo "Tail progress:  tail -f $LOG0 $LOG1"

wait "$P0"; echo "GPU0 run exited ($?)"
wait "$P1"; echo "GPU1 run exited ($?)"

echo "=================================================================="
echo "Both halves done. Aggregating with the OFFICIAL merge..."
export EVALUATION_DATASET=bench2drive   # merge_route_json reads this for the /220 formula
python3 slurm/evaluation/merge_route_json.py --folder "$OUT/eval"
cp -f "$OUT/eval/merged.json" "$OUT/merged.json" 2>/dev/null || true
echo "Merged: $OUT/merged.json"
echo "Read 'driving score' = sum(score_composed)/220 (official, valid for a full run)."
echo "'current_driving_score' = mean over routes with a result (use for partial runs)."
