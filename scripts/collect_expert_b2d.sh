#!/usr/bin/bash
# Re-run the LEAD expert (DATAGEN=1, LEAD sensor rig) on benchmark routes, e.g. the
# 220 Bench2Drive eval routes, to get unseen normative drives for scoring.
# One CARLA per route with retries, like eval_b2d_one_gpu.sh; two of these can run
# side by side, one per GPU, on halves of the route list.
#
# Usage:
#   collect_expert_b2d.sh <gpu_index> <routes_list_file>
#   ls $PWD/data/benchmark_routes/bench2drive/*.xml > routes.txt   # all 220
#
# Output (a carla_root the training code and scripts/score_surprisal.py read):
#   $OUT/data/<ScenarioType>/<Town>_Rep0_<route>/   frames, metas, results.json
#   $OUT/results/<route_id>.json                   leaderboard record (expert DS)
#   $OUT/logs/<route_id>_gpu<k>.log
#
# Env knobs:
#   OUT=data/b2d_expert   MAX_ATTEMPTS=3   ROUTE_WALLCLOCK=1800   CARLA_BOOT_TIMEOUT=240
#   GPU_ADAPTERS="0 2"    logical GPU -> Vulkan -graphicsadapter map
#   TM_SEED=0             traffic-manager seed (the B2D evals use 0)

set -u
export LEAD_PROJECT_ROOT="${LEAD_PROJECT_ROOT:-/home/divyanshu/lead-adapt}"
cd "$LEAD_PROJECT_ROOT"

GPU="${1:?usage: collect_expert_b2d.sh <gpu> <routes_list_file>}"
LIST="${2:?need routes list file}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_INTERPRETER:-lead}"

export CARLA_ROOT="${CARLA_ROOT:-$LEAD_PROJECT_ROOT/3rd_party/CARLA_0915}"
export PYTHONPATH="3rd_party/leaderboard_autopilot:3rd_party/scenario_runner_autopilot:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
export SCENARIO_RUNNER_ROOT=3rd_party/scenario_runner_autopilot
export LEADERBOARD_ROOT=3rd_party/leaderboard_autopilot
export DEBUG_CHALLENGE=0 DATAGEN=1 PYTHONUNBUFFERED=1

OUT="${OUT:-$LEAD_PROJECT_ROOT/data/b2d_expert}"
mkdir -p "$OUT/data" "$OUT/results" "$OUT/logs" "$OUT/staging"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
ROUTE_WALLCLOCK="${ROUTE_WALLCLOCK:-1800}"
CARLA_BOOT_TIMEOUT="${CARLA_BOOT_TIMEOUT:-240}"
TM_SEED="${TM_SEED:-0}"
GPU_ADAPTERS="${GPU_ADAPTERS:-0 2}"
ADAPTER=$(echo "$GPU_ADAPTERS" | awk -v i="$GPU" '{print $(i+1)}')

echo "GPU $GPU (vk adapter $ADAPTER) | routes: $(wc -l < "$LIST") | out: $OUT"

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

kill_carla_on_port() {  # the engine is reparented to init, so kill it by its port
    local port=$1 pids
    pids=$(ss -ltnp "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    pkill -9 -f "CarlaUE4-Linux-Shipping.*world-port=$port" 2>/dev/null
}

run_attempt() {
    local route_file=$1 route_id=$2 attempt=$3
    local port tm_port; port=$(free_port); tm_port=$(free_port)
    local ckpt_json="$OUT/results/${route_id}.json"
    echo "--- attempt $attempt: $route_id GPU$GPU adapter$ADAPTER port$port $(date) ---"
    CUDA_VISIBLE_DEVICES=$GPU bash "$CARLA_ROOT/CarlaUE4.sh" \
        --world-port="$port" -nosound -graphicsadapter="$ADAPTER" -RenderOffScreen &
    local carla_pid=$!
    local rc=1
    if ! wait_for_carla "$port"; then
        echo "!!! CARLA failed to accept on $port within ${CARLA_BOOT_TIMEOUT}s"
    else
        # A failed attempt leaves a half-written route folder: start every attempt clean.
        rm -rf "$OUT/staging/$route_id" "$ckpt_json"
        SAVE_PATH="$OUT/staging/$route_id" TEAM_CONFIG="$route_file" \
        CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 "$ROUTE_WALLCLOCK" python3 -u \
            3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
            --routes="$route_file" --repetitions=1 --track=MAP --checkpoint="$ckpt_json" \
            --agent=lead/expert/expert.py --agent-config="$route_file" \
            --debug=0 --resume=0 --timeout=600 \
            --port="$port" --traffic-manager-port="$tm_port" --traffic-manager-seed="$TM_SEED"
        rc=$?
        [ $rc -eq 124 ] && echo "!!! $route_id hit ${ROUTE_WALLCLOCK}s wallclock"
    fi
    kill -9 "$carla_pid" 2>/dev/null; pkill -9 -P "$carla_pid" 2>/dev/null
    kill_carla_on_port "$port"
    sleep 8
    return $rc
}

n=0; ok=0; total=$(wc -l < "$LIST")
while IFS= read -r route_file; do
    [ -z "$route_file" ] && continue
    n=$((n+1)); route_id=$(basename "$route_file" .xml)
    scenario=$(python3 -c "import sys,xml.etree.ElementTree as ET;s=ET.parse(sys.argv[1]).find('route/scenarios/scenario');print(s.attrib['type'] if s is not None else 'noScenarios')" "$route_file")
    log="$OUT/logs/${route_id}_gpu${GPU}.log"
    if json_ok "$OUT/results/${route_id}.json" && [ ! -d "$OUT/staging/$route_id" ]; then
        echo "[skip] $route_id already collected ($n/$total)"; ok=$((ok+1)); continue
    fi
    echo "[run ] GPU$GPU $route_id $scenario ($n/$total)"
    {
        echo "=== route $route_id ($scenario) on GPU $GPU $(date) ==="
        for a in $(seq 1 "$MAX_ATTEMPTS"); do
            run_attempt "$route_file" "$route_id" "$a"; rc=$?
            # A record means the route ran to its end (with or without infractions);
            # the bucket builder later drops drives the expert did not finish cleanly.
            if json_ok "$OUT/results/${route_id}.json" && ls "$OUT/staging/$route_id"/*/results.json >/dev/null 2>&1; then
                mkdir -p "$OUT/data/$scenario"
                mv "$OUT/staging/$route_id"/* "$OUT/data/$scenario/" && rmdir "$OUT/staging/$route_id"
                echo "=== done $route_id rc=$rc (attempt $a) $(date) ==="; break
            fi
            echo "--- attempt $a failed (rc=$rc); $([ $a -lt $MAX_ATTEMPTS ] && echo retrying || echo giving up) ---"
        done
    } > "$log" 2>&1
    json_ok "$OUT/results/${route_id}.json" && [ ! -d "$OUT/staging/$route_id" ] && ok=$((ok+1))
done < "$LIST"

echo "GPU $GPU finished: $ok/$total routes collected."
