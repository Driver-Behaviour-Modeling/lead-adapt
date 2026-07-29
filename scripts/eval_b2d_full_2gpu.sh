#!/usr/bin/bash
# Full Bench2Drive evaluation across 2 local GPUs (no SLURM), producing the
# OFFICIAL score via the harness's own merge_route_json aggregation.
#
# One CARLA + one leaderboard_evaluator run per route, two routes in flight at
# once (GPU 0 and GPU 1). Per-route result JSONs are written to the same
# <out>/eval/<route_id>.json layout the SLURM harness uses, then merged with
# slurm/evaluation/merge_route_json.py -> merged.json (the "95"-style number).
#
# Usage:
#   scripts/eval_b2d_full_2gpu.sh <checkpoint_dir> [route_glob]
# Example:
#   scripts/eval_b2d_full_2gpu.sh outputs/checkpoints/adapt_finetune_v5
#   scripts/eval_b2d_full_2gpu.sh outputs/checkpoints/adapt_finetune_v5 \
#       "data/benchmark_routes/bench2drive/23687.xml"   # single route smoke test

set -u

# --- Canonical repo root (verified location of the real files) ---
export LEAD_PROJECT_ROOT="${LEAD_PROJECT_ROOT:-/home/divyanshu/lead-adapt}"
cd "$LEAD_PROJECT_ROOT"

CHECKPOINT_DIR="${1:?usage: eval_b2d_full_2gpu.sh <checkpoint_dir> [route_glob]}"
ROUTE_GLOB="${2:-data/benchmark_routes/bench2drive/*.xml}"

# --- Single-instance guard: refuse to start if another copy is already running.
# The double-launch is what wrecked the previous run (4 CARLAs on 2 GPUs).
LOCKFILE="/tmp/eval_b2d_full_2gpu.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "ERROR: another eval_b2d_full_2gpu.sh is already running (lock: $LOCKFILE)."
    echo "       If you are sure none is, remove the lock and retry."
    exit 1
fi
trap 'pkill -9 -f CarlaUE4 2>/dev/null; rm -f "$LOCKFILE"' EXIT

# Conda env (matches slurm/evaluate.sh)
eval "$(conda shell.bash hook)"
conda activate "${CONDA_INTERPRETER:-lead}"

export PYTHONPATH="3rd_party/Bench2Drive/leaderboard:3rd_party/Bench2Drive/scenario_runner:$LEAD_PROJECT_ROOT/3rd_party/CARLA_0915/PythonAPI/carla:${PYTHONPATH:-}"
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export CARLA_ROOT="${CARLA_ROOT:-$LEAD_PROJECT_ROOT/3rd_party/CARLA_0915}"
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
export EVALUATION_DATASET=bench2drive
export PYTHONUNBUFFERED=1

# --- Debug visualizations -------------------------------------------------
# DEBUG=1        -> debug video only  (<route>_debug.mp4)
# DEBUG=full     -> debug + demo + input videos
# Or export LEAD_CLOSED_LOOP_CONFIG yourself for full control (wins over DEBUG).
# Videos are written to $OUT/<route_id>/. Rendering adds per-frame overhead and
# disk (~10-30 MB/route), so leave DEBUG unset for the fastest scoring-only run.
if [ -z "${LEAD_CLOSED_LOOP_CONFIG:-}" ]; then
    case "${DEBUG:-0}" in
        1)    export LEAD_CLOSED_LOOP_CONFIG="debug_mode=true" ;;
        full) export LEAD_CLOSED_LOOP_CONFIG="debug_mode=true produce_debug_video=true produce_demo_video=true produce_input_video=true" ;;
    esac
fi
[ -n "${LEAD_CLOSED_LOOP_CONFIG:-}" ] && echo "Debug config: LEAD_CLOSED_LOOP_CONFIG='$LEAD_CLOSED_LOOP_CONFIG'"

EVALUATOR=3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py
CKPT_NAME=$(basename "$CHECKPOINT_DIR")
OUT="$LEAD_PROJECT_ROOT/outputs/evaluation/bench2drive/${CKPT_NAME}"
export EVALUATION_OUTPUT_DIR="$OUT"
mkdir -p "$OUT/eval" "$OUT/logs"

echo "Checkpoint : $CHECKPOINT_DIR"
echo "Output dir : $OUT"
mapfile -t ROUTES < <(ls -1 $ROUTE_GLOB 2>/dev/null | sort)
echo "Routes     : ${#ROUTES[@]}"
[ "${#ROUTES[@]}" -eq 0 ] && { echo "No routes matched: $ROUTE_GLOB"; exit 1; }

free_port() {
    while true; do
        local port=$(( RANDOM % 50001 + 10000 ))
        if ! ss -lpn 2>/dev/null | grep -q ":$port "; then echo "$port"; return; fi
    done
}

# Hard wallclock ceiling per route (seconds). A route that exceeds this is
# SIGKILLed so a hung sim (Bench2Drive's signal-handler deadlock on the 120s
# sim-timeout) can never stall the pool. B2D routes are short; 900s is generous.
ROUTE_WALLCLOCK="${ROUTE_WALLCLOCK:-900}"
# How long to wait for CARLA's RPC port to accept connections before giving up.
CARLA_BOOT_TIMEOUT="${CARLA_BOOT_TIMEOUT:-240}"

# Map a logical render-GPU index (0,1,...) to a CARLA -graphicsadapter (Vulkan)
# index. Vulkan enumerates a software "llvmpipe" device among the real GPUs, so
# adapter indices are NOT 1:1 with physical GPUs. On this box:
#   adapter 0 = NVIDIA GPU0,  adapter 1 = llvmpipe (CPU!),  adapter 2 = NVIDIA GPU1
# Sending CARLA to the llvmpipe adapter wedges the render thread (60s timeout ->
# SIGSEGV), which is exactly what killed every GPU-1 route. Override GPU_ADAPTERS
# if your `vulkaninfo --summary` shows a different layout.
GPU_ADAPTERS="${GPU_ADAPTERS:-0 2}"
adapter_for_gpu() { echo "$GPU_ADAPTERS" | awk -v i="$1" '{print $(i+1)}'; }

# Retries per route (transient CARLA boot/segfault failures). The official
# harness retries up to 3; match that.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

# Poll until CARLA's world port ACCEPTS a TCP connection (listening alone is not
# enough — CARLA opens the port well before the RPC server is ready). Returns 0
# only once a real connect succeeds.
wait_for_carla() {
    local port=$1 deadline=$(( SECONDS + CARLA_BOOT_TIMEOUT ))
    while [ $SECONDS -lt $deadline ]; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&- 3<&- 2>/dev/null; sleep 3; return 0
        fi
        sleep 3
    done
    return 1
}

# Run ONE attempt of a route on a given GPU. Echoes progress into the caller's
# already-redirected log. Returns the evaluator rc (or 1 if CARLA never came up).
run_attempt() {
    local gpu=$1 route_file=$2 route_id=$3 attempt=$4
    local adapter; adapter=$(adapter_for_gpu "$gpu")
    local port tm_port; port=$(free_port); tm_port=$(free_port)
    local ckpt_json="$OUT/eval/${route_id}.json"

    echo "--- attempt $attempt: route $route_id GPU$gpu (vk adapter $adapter, port $port) $(date) ---"
    CUDA_VISIBLE_DEVICES=$gpu bash "$CARLA_ROOT/CarlaUE4.sh" \
        --world-port="$port" -nosound -graphicsadapter="$adapter" -RenderOffScreen &
    local carla_pid=$!

    local rc=1
    if ! wait_for_carla "$port"; then
        echo "!!! CARLA failed to accept on port $port within ${CARLA_BOOT_TIMEOUT}s"
    else
        echo "--- CARLA ready on $port; launching evaluator ---"
        rm -f "$ckpt_json"
        SAVE_PATH="$OUT/$route_id" BENCHMARK_ROUTE_ID="$route_id" \
        CUDA_VISIBLE_DEVICES=$gpu timeout -k 30 "$ROUTE_WALLCLOCK" python3 "$EVALUATOR" \
            --routes="$route_file" --track=SENSORS --checkpoint="$ckpt_json" \
            --agent=lead/inference/sensor_agent.py --agent-config="$CHECKPOINT_DIR" \
            --debug=0 --record=None --resume=False \
            --port="$port" --traffic-manager-port="$tm_port" --timeout=120 \
            --debug-checkpoint="$OUT/eval/debug_${route_id}.txt" \
            --traffic-manager-seed=0 --repetitions=1
        rc=$?
        [ $rc -eq 124 ] && echo "!!! route $route_id hit ${ROUTE_WALLCLOCK}s wallclock — killed"
    fi

    # Tear down THIS route's CARLA by its own pid + children; then wait for the
    # driver to release the GPU before the next attempt/route reuses it.
    kill -9 "$carla_pid" 2>/dev/null
    pkill -9 -P "$carla_pid" 2>/dev/null
    sleep 8
    return $rc
}

# A produced JSON counts as a real result only if it has a non-empty records
# array (setup crashes write an empty one — those should be retried).
json_ok() {
    python3 -c "import json,sys;d=json.load(open('$1'));sys.exit(0 if d['_checkpoint']['records'] else 1)" 2>/dev/null
}

# --- run ONE route fully on a given GPU, with retries (blocking) ---
run_route() {
    local gpu=$1 route_file=$2
    local route_id; route_id=$(basename "$route_file" .xml)
    local log="$OUT/logs/${route_id}_gpu${gpu}.log"
    {
        echo "=== route $route_id on GPU $gpu $(date) ==="
        local a rc
        for a in $(seq 1 "$MAX_ATTEMPTS"); do
            run_attempt "$gpu" "$route_file" "$route_id" "$a"; rc=$?
            if [ $rc -eq 0 ] && json_ok "$OUT/eval/${route_id}.json"; then
                echo "=== done $route_id rc=0 (attempt $a) $(date) ==="; return
            fi
            echo "--- attempt $a failed (rc=$rc); $([ $a -lt $MAX_ATTEMPTS ] && echo retrying || echo giving up) ---"
        done
        echo "=== FAILED $route_id after $MAX_ATTEMPTS attempts $(date) ==="
    } > "$log" 2>&1
}

# A route counts as DONE if its JSON exists with a non-empty records array
# (RESUME=0 forces a full re-run).
route_done() {
    local rid=$1
    [ "${RESUME:-1}" = "0" ] && return 1
    python3 - "$OUT/eval/${rid}.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    sys.exit(0 if d["_checkpoint"]["records"] else 1)
except Exception:
    sys.exit(1)
PY
}

# --- schedule 2 routes at a time (GPU 0 and GPU 1) ---
declare -A GPU_PID   # gpu -> bg pid
declare -A GPU_ROUTE
i=0
while [ $i -lt ${#ROUTES[@]} ] || [ ${#GPU_PID[@]} -gt 0 ]; do
    for gpu in 0 1; do
        # launch on a free GPU if routes remain
        if [ -z "${GPU_PID[$gpu]:-}" ] && [ $i -lt ${#ROUTES[@]} ]; then
            rf="${ROUTES[$i]}"; rid=$(basename "$rf" .xml)
            if route_done "$rid"; then
                echo "[skip]   route $rid already has a valid result ($((i+1))/${#ROUTES[@]})"
                i=$((i+1)); continue
            fi
            echo "[launch] GPU $gpu <- route $rid ($((i+1))/${#ROUTES[@]})"
            run_route "$gpu" "$rf" &
            GPU_PID[$gpu]=$!; GPU_ROUTE[$gpu]=$rid
            i=$((i+1))
        fi
    done
    # reap finished GPUs
    for gpu in 0 1; do
        pid="${GPU_PID[$gpu]:-}"
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            echo "[done]   GPU $gpu -- route ${GPU_ROUTE[$gpu]}"
            unset 'GPU_PID['"$gpu"']' 'GPU_ROUTE['"$gpu"']'
        fi
    done
    sleep 5
done

echo "=================================================================="
echo "All routes finished. Aggregating with the OFFICIAL merge..."
python3 slurm/evaluation/merge_route_json.py --folder "$OUT/eval"
cp -f "$OUT/eval/merged.json" "$OUT/merged.json" 2>/dev/null || true
echo "Merged result: $OUT/merged.json"
echo "Official B2D 'driving score' field = sum(score_composed)/220 (valid only for a full 220-route run)."
echo "For a partial run, read 'current_driving_score' (mean over the routes actually evaluated)."
