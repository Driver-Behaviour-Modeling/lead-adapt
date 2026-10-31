#!/usr/bin/env bash
# Capture once, fit on validation only, and evaluate two separate anomaly channels.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN=${PYTHON_BIN:-/localstorage/home/f20221129/miniconda3/envs/lead/bin/python}
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a directory containing one model and config.json}"
: "${DATA_ROOT:?Set DATA_ROOT to the recorded normal-data root}"
: "${PARTITIONS:?Set PARTITIONS to the frozen normal-drive manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new experiment directory}"
if [[ -e "$OUTPUT_DIR" ]]; then
    printf '%s\n' "Refusing to overwrite $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
unset LEAD_TRAINING_CONFIG
trap 'rc=$?; if ((rc != 0)); then printf "failed: %s\n" "$rc" > "$OUTPUT_DIR/status.txt"; fi' EXIT
cp "$PARTITIONS" "$OUTPUT_DIR/partitions.json"
printf '%s\n' capturing > "$OUTPUT_DIR/status.txt"
"$PYTHON_BIN" scripts/score_integrated_reference.py \
    --checkpoint-dirs "$CHECKPOINT_DIR" --data-root "$DATA_ROOT" \
    --output "$OUTPUT_DIR/raw" --batch-size "${BATCH_SIZE:-4}" --workers "${WORKERS:-4}" \
    --save-member-logits --save-world-futures --forecast-metrics
printf '%s\n' fitting > "$OUTPUT_DIR/status.txt"
"$PYTHON_BIN" scripts/fit_integrated_reference.py \
    --capture "$OUTPUT_DIR/raw" --partitions "$OUTPUT_DIR/partitions.json" \
    --output-dir "$OUTPUT_DIR/fitted" --device "${FIT_DEVICE:-cpu}"
printf '%s\n' calibrating > "$OUTPUT_DIR/status.txt"
for channel in score world_conflict_count; do
    "$PYTHON_BIN" scripts/calibrate_integrated_reference.py fit \
        --features "$OUTPUT_DIR/fitted/calibration" \
        --output "$OUTPUT_DIR/${channel}_calibration.json" --channel "$channel"
    "$PYTHON_BIN" scripts/calibrate_integrated_reference.py evaluate \
        --calibration "$OUTPUT_DIR/${channel}_calibration.json" \
        --features "$OUTPUT_DIR/fitted/drive_reference" \
        --output "$OUTPUT_DIR/${channel}_drive_reference.json"
    "$PYTHON_BIN" scripts/calibrate_integrated_reference.py evaluate \
        --calibration "$OUTPUT_DIR/${channel}_calibration.json" \
        --features "$OUTPUT_DIR/fitted/test" \
        --normal-drive-reference "$OUTPUT_DIR/${channel}_drive_reference.json" \
        --output "$OUTPUT_DIR/${channel}_test.json"
done
printf '%s\n' complete > "$OUTPUT_DIR/status.txt"
