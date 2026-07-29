#!/usr/bin/bash
# Run a Bench2Drive evaluation for ONE GPU over an explicit list of route files.
# Fully self-contained and independent — two of these can run side by side, one
# per GPU, with no shared scheduling. Results go to the SAME shared eval/ dir so
# the official merge sees all routes from both halves.
#
# Usage:
#   eval_b2d_one_gpu.sh <checkpoint_dir> <gpu_index> <routes_list_file>
# where <routes_list_file> has one route .xml path per line.
#
# Env knobs:
#   DEBUG=1|full        debug videos (off by default)
#   MAX_ATTEMPTS=3      retries per route on transient CARLA failures
#   ROUTE_WALLCLOCK=900 hard per-route wallclock (s)
#   CARLA_BOOT_TIMEOUT=240
#   GPU_ADAPTERS="0 2"  logical GPU -> Vulkan -graphicsadapter map (see below)

set -u
export LEAD_PROJECT_ROOT="${LEAD_PROJECT_ROOT:-/home/divyanshu/lead-adapt}"
cd "$LEAD_PROJECT_ROOT"

CHECKPOINT_DIR="${1:?usage: eval_b2d_one_gpu.sh <checkpoint_dir> <gpu> <routes_list_file>}"
GPU="${2:?need gpu index}"
LIST="${3:?need routes list file}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_INTERPRETER:-lead}"

export PYTHONPATH="3rd_party/Bench2Drive/leaderboard:3rd_party/Bench2Drive/scenario_runner:$LEAD_PROJECT_ROOT/3rd_party/CARLA_0915/PythonAPI/carla:${PYTHONPATH:-}"
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export CARLA_ROOT="${CARLA_ROOT:-$LEAD_PROJECT_ROOT/3rd_party/CARLA_0915}"
export IS_BENCH2DRIVE=1 PLANNER_TYPE=only_traj EVALUATION_DATASET=bench2drive PYTHONUNBUFFERED=1

# Debug visualizations (off by default for a fast, stable scoring run).
if [ -z "${LEAD_CLOSED_LOOP_CONFIG:-}" ]; then
    case "${DEBUG:-0}" in
        1)    export LEAD_CLOSED_LOOP_CONFIG="debug_mode=true" ;;
        full) export LEAD_CLOSED_LOOP_CONFIG="debug_mode=true produce_debug_video=true produce_demo_video=true produce_input_video=true" ;;
    esac
fi

EVALUATOR=3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py
CKPT_NAME=$(basename "$CHECKPOINT_DIR")
OUT="$LEAD_PROJECT_ROOT/outputs/evaluation/bench2drive/${CKPT_NAME}"
export EVALUATION_OUTPUT_DIR="$OUT"
mkdir -p "$OUT/eval" "$OUT/logs"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
ROUTE_WALLCLOCK="${ROUTE_WALLCLOCK:-900}"
CARLA_BOOT_TIMEOUT="${CARLA_BOOT_TIMEOUT:-240}"
RESUME="${RESUME:-1}"

# Vulkan adapter map: adapter 1 is the llvmpipe CPU renderer on this box, so
# GPU1 must use adapter 2. `vulkaninfo --summary` confirms: 0=NVIDIA,1=llvmpipe,2=NVIDIA.
GPU_ADAPTERS="${GPU_ADAPTERS:-0 2}"
ADAPTER=$(echo "$GPU_ADAPTERS" | awk -v i="$GPU" '{print $(i+1)}')

echo "GPU $GPU (vk adapter $ADAPTER) | routes: $(wc -l < "$LIST") | debug='${LEAD_CLOSED_LOOP_CONFIG:-off}'"

free_port() { while true; do local p=$(( RANDOM%50001+10000 )); ss -ltn 2>/dev/null | grep -q ":$p " || { echo "$p"; return; }; done; }

wait_for_carla() {  # returns 0 only when a real TCP connect succeeds
    local port=$1 deadline=$(( SECONDS + CARLA_BOOT_TIMEOUT ))
    while [ $SECONDS -lt $deadline ]; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then exec 3>&- 3<&- 2>/dev/null; sleep 3; return 0; fi
        sleep 3
    done
    return 1
}

json_ok() { python3 -c "import json,sys;d=json.load(open('$1'));sys.exit(0 if d['_checkpoint']['records'] else 1)" 2>/dev/null; }

# Kill the CARLA engine bound to a given world-port. CARLA's CarlaUE4.sh wrapper
# launches CarlaUE4-Linux-Shipping which gets REPARENTED to init(1), so killing
# the wrapper pid leaks the engine (fills VRAM). Killing by the listening port
# reliably reaches the real engine regardless of reparenting.
kill_carla_on_port() {
    local port=$1 pids
    pids=$(ss -ltnp "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    # belt-and-suspenders: also match the world-port on the engine cmdline
    pkill -9 -f "CarlaUE4-Linux-Shipping.*world-port=$port" 2>/dev/null
}

run_attempt() {
    local route_file=$1 route_id=$2 attempt=$3
    local port tm_port; port=$(free_port); tm_port=$(free_port)
    local ckpt_json="$OUT/eval/${route_id}.json"
    echo "--- attempt $attempt: $route_id GPU$GPU adapter$ADAPTER port$port $(date) ---"
    CUDA_VISIBLE_DEVICES=$GPU bash "$CARLA_ROOT/CarlaUE4.sh" \
        --world-port="$port" -nosound -graphicsadapter="$ADAPTER" -RenderOffScreen &
    local carla_pid=$!
    local rc=1
    if ! wait_for_carla "$port"; then
        echo "!!! CARLA failed to accept on $port within ${CARLA_BOOT_TIMEOUT}s"
    else
        rm -f "$ckpt_json"
        SAVE_PATH="$OUT/$route_id" BENCHMARK_ROUTE_ID="$route_id" \
        CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 "$ROUTE_WALLCLOCK" python3 "$EVALUATOR" \
            --routes="$route_file" --track=SENSORS --checkpoint="$ckpt_json" \
            --agent=lead/inference/sensor_agent.py --agent-config="$CHECKPOINT_DIR" \
            --debug=0 --record=None --resume=False \
            --port="$port" --traffic-manager-port="$tm_port" --timeout=120 \
            --debug-checkpoint="$OUT/eval/debug_${route_id}.txt" \
            --traffic-manager-seed=0 --repetitions=1
        rc=$?
        [ $rc -eq 124 ] && echo "!!! $route_id hit ${ROUTE_WALLCLOCK}s wallclock"
    fi
    kill -9 "$carla_pid" 2>/dev/null; pkill -9 -P "$carla_pid" 2>/dev/null
    kill_carla_on_port "$port"   # the reparented engine survives the wrapper kill
    sleep 8
    return $rc
}

n=0; ok=0; total=$(wc -l < "$LIST")
while IFS= read -r route_file; do
    [ -z "$route_file" ] && continue
    n=$((n+1)); route_id=$(basename "$route_file" .xml)
    log="$OUT/logs/${route_id}_gpu${GPU}.log"
    if [ "$RESUME" = "1" ] && json_ok "$OUT/eval/${route_id}.json"; then
        echo "[skip] $route_id already done ($n/$total)"; ok=$((ok+1)); continue
    fi
    echo "[run ] GPU$GPU $route_id ($n/$total)"
    {
        echo "=== route $route_id on GPU $GPU $(date) ==="
        rc=1
        for a in $(seq 1 "$MAX_ATTEMPTS"); do
            run_attempt "$route_file" "$route_id" "$a"; rc=$?
            if [ $rc -eq 0 ] && json_ok "$OUT/eval/${route_id}.json"; then
                echo "=== done $route_id rc=0 (attempt $a) $(date) ==="; break
            fi
            echo "--- attempt $a failed (rc=$rc); $([ $a -lt $MAX_ATTEMPTS ] && echo retrying || echo giving up) ---"
        done
    } > "$log" 2>&1
    json_ok "$OUT/eval/${route_id}.json" && ok=$((ok+1))
done < "$LIST"

echo "GPU $GPU finished: $ok/$total routes have valid results."
