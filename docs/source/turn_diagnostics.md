# Diagnosing failed turns

`scripts/diagnose_turns.py` prepares a controlled Bench2Drive subset using the
`adapt_posttrain_radar_ext2/model_0039.pth` checkpoint. The default routes are
14842, 3144, 3905, and 3737 (previous failed turns), plus 3904 and 25968
(previous successful turns). These labels describe the previous evaluation;
they do not assume a replay will produce the same result.

Use the project's `lead` Python environment from the repository root. The
default operation prepares files and checks Python imports, CUDA, and ffmpeg. It does
not start or connect to a simulator:

```bash
python -m scripts.diagnose_turns --routes 14842 4468
```

To execute a baseline comparison, add `--run`:

```bash
python -m scripts.diagnose_turns --run --routes 14842 4468 \
  --gpu 0 --graphics-adapter 0 --route-timeout 4500
```

The CUDA device and CARLA Vulkan adapter are separate indices. The adapter
mapping depends on the machine; do not reuse a different server's mapping
without checking it. CARLA 0.9.15 is expected at `3rd_party/CARLA_0915`; use
`--carla-root` or `--python` to select another installed simulator or interpreter.

Each invocation creates a fresh timestamped directory under
`outputs/diagnostics/turns/`. An explicit `--output PATH` must not already exist.
The runner copies each original per-route XML byte for byte, including weather,
scenario triggers, and waypoints. It copies the saved training configuration
and links exactly the selected checkpoint into a new directory so other model
files cannot accidentally form an ensemble. Original evaluation results and
checkpoint files are never replaced.

## One change at a time

The following variants differ from the baseline by exactly one behavior flag:

| Variant | Changed closed-loop option | Purpose |
|---|---|---|
| `baseline` | None | Reproduce the current navigation and history inputs. |
| `history_aligned` | `adapt_history_mode=training_aligned` | Compare history construction against the training convention. |
| `localized_navigation` | `navigation_position_source=planner` | Use the route planner's position source for target-point localization. |
| `training_pop_distance` | `navigation_pop_distance_mode=training` | Use the saved training target-point pop distance. |

For example:

```bash
python -m scripts.diagnose_turns --run --routes 14842 4468 \
  --variants baseline history_aligned \
  --gpu 0 --graphics-adapter 0 --route-timeout 4500
```

Every variant sets `record_driving_diagnostics=true` and disables debug video.
The three behavior controls are set explicitly, including their legacy values.
The runner rejects ambient `LEAD_TRAINING_CONFIG`, `LEAD_OPEN_LOOP_CONFIG`,
`LEAD_CLOSED_LOOP_CONFIG`, and `LEAD_EXPERT_CONFIG` overrides so they cannot
silently alter a comparison. This also keeps camera-head and augmentation
options at the values saved with the selected checkpoint.

Each route/variant has one behavioral attempt with one benchmark repetition;
failures are recorded without retrying. Every attempt starts a separate CARLA
process with unused local ports and terminates only its own process groups.
Startup and evaluation wall-clock limits default to 240 and 900 seconds.
Choose the evaluation limit from measured simulator throughput and allow time
for an official blocked/timeout outcome. The examples use 4500 seconds on this
server. Route 4468 is a previously successful left turn in Town12/weather 25
that took 16.15 simulated seconds in the supplied evaluation; route 3904 took
109.85 seconds. These prior durations inform pilot selection, not expectations
of the replay's outcome. A wall-clock cutoff is incomplete evidence, not an
official driving failure or success.
`--seed` defaults to zero and controls Python, NumPy, Torch, and TrafficManager.
These seeds help comparisons but do not guarantee deterministic CARLA replays.
`--cpu-threads` defaults to 4. It sets OMP, MKL, OpenBLAS, Numba, vecLib, and
NumExpr limits before Python imports and explicitly sets both Torch thread
pools. Large unbounded pools can make a small inference batch much slower.
The manifest records the limits and observed Torch pool sizes. Set
`--cpu-threads 0` to retain inherited library defaults. Keep these settings
identical across the baseline and each behavior variant.

Use `--snapshot-steps 40 41 42` to capture selected model forwards for offline
comparisons. Selections must be unique nonnegative agent step numbers. The
manifest records the selection; files are published under
`agent/model_snapshots/<step>.pth` with a SHA-256/size index in `index.jsonl`.
The runner verifies that every requested step present in the driving trace has
a valid indexed snapshot. It reports all missing selections, including steps
that were never observed because a route ended early or initialization did not
run a model forward. Partial capture is explicitly marked incomplete; a missing
snapshot at an observed step or an invalid file hash is an output error.
An execution error or missing diagnostic output stops the experiment so a code
or simulator problem is not repeated across the remaining cases. Ordinary
driving failures with valid evaluation and diagnostic output continue normally.

## Inspecting the outputs

`manifest.json` records source/configuration/checkpoint/codebook SHA-256 hashes,
the original route locations, exact variant settings, the seed, and preflight
results. `<variant>/<route>/` contains:

- `agent/driving_diagnostics.jsonl`: per-tick navigation inputs, history,
  predictions, and executed controls from the instrumented agent.
- `result.json`: the official evaluator's route outcome and infractions.
- `attempt.json`: commands, execution status, timing, and diagnostic-file size.
- `carla.log` and `evaluator.log`: simulator and evaluator output.

`summary.json` collects completed attempts, including failed ones. A successful
runner exit means each attempted case produced a route record and diagnostic
output with at least one valid step; metadata alone does not qualify. An agent
initialization/runtime crash is an execution error even if the evaluator exits
zero. Successful runner completion does not mean the vehicle completed every
route successfully.
This subset is for diagnosing a mechanism, not estimating the full 220-route
score. A promising change still needs broader evaluation with successful-route
regression checks.

Analyze a completed trace with:

```bash
python -m scripts.analyze_driving_diagnostics \
  --trace outputs/diagnostics/turns/<run>/baseline/14842/agent/driving_diagnostics.jsonl \
  --output outputs/diagnostics/turns/<run>/baseline/14842/analysis --plot
```

The analyzer writes a summary, per-step CSV, candidate moments, and a selected
step record. `--plot` adds trajectory and timing plots; `--step N` selects a
specific recorded decision. Candidate flags identify differences to inspect,
not a causal verdict. Sparse navigation targets are not lane boundaries or an
exact reference trajectory. Planned-path overlays use observed filtered
localization and compass in the GPS-derived navigation frame and are labeled as
estimates. That frame's origin can differ from the simulator's true world
origin; comparing it directly with `offline_ground_truth` would require origin
calibration. The reported localization gap compares noisy and filtered
positions within the same navigation frame. Parsing issues and
truncated records are reported rather than silently treated as valid driving.

## Input conventions and trace timing

History selection and navigation target construction now use shared helpers in
`lead/common/history_features.py` and `lead/common/navigation_features.py`.
Legacy training and inference values are preserved by default. With five poses,
spacing five and a 20 Hz simulator, the existing training ages are
`[25, 20, 15, 10, 5]` ticks; legacy online ages are `[20, 15, 10, 5, 0]`.
`training_aligned` changes only the online selection to the former window.
Startup padding is marked explicitly; unavailable samples are never described
as actual observations at the requested timestamp.

Navigation localization and waypoint popping are separate experiments. The
`planner` position option uses the same filtered/noisy position source as online
route progression. It does not reproduce privileged training localization.
The `training` pop-distance option uses the checkpoint's distance and disables
the adaptive 5/4 m selection. Neither option accesses ground-truth vehicle pose
for policy input. Offline ground-truth pose is recorded after control selection
solely for analysis.

`timestamp_seconds` is the evaluator's elapsed game time. `sensor_frames` are
global CARLA frame IDs, not elapsed seconds. Historical timestamps are inferred
from queue ages and the configured tick frequency. The initialization tick has
no model prediction, so step records begin at agent step 1. `controller_control`
is captured before stopping/creeping heuristics; `executed_control` includes
those heuristics and initial braking. Radar outputs are retained separately
for each model because the existing ensemble object does not aggregate them.

## Sensor-cache compatibility

Persistent and session sensor keys now include a versioned fingerprint of the
preprocessing that creates their contents, including radar enablement and label
settings. This is separate from the recently added metadata cache. Run settings
such as optimizer, compilation, and post-cache color augmentation are excluded.
Persistent writes use atomic replacement so concurrent readers cannot consume
partially written entries.

Old sensor caches remain on disk but are not reused. New entries populate
`cache/<scenario>/<route>/sensor-v2/<fingerprint>/<view>/<frame>.pkl` lazily.
Expect extra storage and a slower first training pass while this cache fills.
Source recordings are assumed immutable within a dataset root; replacing those
files still requires `force_rebuild_data_cache=true` or a new dataset root.

Use the original camera-head and augmentation settings for the first score
comparisons. The diagnostic variants should each change one input convention;
they are not claimed improvements until evaluated.

## Initial radar_ext2 pilot (2026-09-09)

With `model_0039.pth`, one completed replay per route and history variant gave:

| Route | Legacy history score | Training-aligned history score |
|---|---:|---:|
| 14842 | 36.34, route deviation | 36.34, route deviation |
| 4468 | 100, completed | 60, completed with one vehicle collision |

Actual input traces confirm the history windows changed as intended. The
checkpoint, saved configuration, codebooks, and source hashes matched across
cases. This pilot does not support enabling alignment by default; retain
`adapt_history_mode=legacy`. It does not estimate a new 220-route score.

## Replaying captured navigation inputs

Selected online frames can retain all pre-forward sensor/history tensors and
raw planning outputs in `agent/model_snapshots/<step>.pth`, with a SHA-256
`index.jsonl`. Capture remains opt-in through the runner's `--snapshot-steps`
argument. It records the actual navigation inputs and two isolated alternatives:
`localized_navigation` changes target-point localization; `training_pop_distance`
changes target-point popping. Each alternative uses the observed localization
history and planner progression from this rollout. It does not reconstruct the
vehicle trajectory that another policy would previously have produced.

After the capture completes, run the full frozen model on the exact saved inputs:

Use the same installed `lead` Python environment as capture, from the repository
root. On this server its interpreter is
`/localstorage/home/f20221129/miniconda3/envs/lead/bin/python`. The module command
below resolves `lead` from the repository; no additional CARLA/leaderboard
`PYTHONPATH` or running simulator is required for replay. The tool sets
`LEAD_PROJECT_ROOT` internally while constructing the model. The environment
must already have the project's model dependencies installed, as for capture.

```bash
python -m scripts.replay_navigation_snapshots \
  --manifest outputs/diagnostics/turns/<capture-run>/manifest.json \
  --snapshots 'outputs/diagnostics/turns/<capture-run>/baseline/14842/agent/model_snapshots/*.pth' \
  --output outputs/diagnostics/turns/<capture-run>/replay_14842 \
  --device cuda:0 --cpu-threads 4
```

The output directory must be new. The tool requires the run's staged, single
checkpoint and configuration, validates source/configuration/codebook/checkpoint
and snapshot-index hashes, and loads weights strictly. Effective configuration,
Torch/CUDA/cuDNN versions, device type/capability, thread counts, autocast and
numeric backend settings must match the capture. Use the capture's CUDA device
mapping and unset ambient `LEAD_*_CONFIG` overrides. CPU threads default to four;
both captured Torch thread pools must match the selected value.

The unchanged full forward must first reproduce every captured raw planning
output. Default `--atol 0 --rtol 0` requires bitwise equality, including dtype
and shape. A mismatch writes the measured maximum/mean errors and stops before
running alternatives. Tolerances are never increased automatically. If a
nonzero tolerance is deliberately supplied after reviewing the discrepancy,
its value and any nonexact acceptance remain explicit in the report.
`--continue-on-mismatch` saves each rejected frame and checks the remaining
snapshots independently. It never runs alternatives for a rejected baseline,
and the overall status/exit code still indicates failure if any frame fails.
Identity or input/weight mutation errors always stop the batch.

Each accepted replay runs both real navigation alternatives through the whole
model and repeats the original baseline afterward. RNG state is restored before
each forward. Every preserved sensory/history input, including fields unknown
to the analyzer, and all model parameters/buffers are hashed before and after.
A restored-baseline failure withholds alternative results.

`summary.json` records validation status. Each selected-step JSON contains raw
route/AR paths, headings, speed logits, genuine `OpenLoopInference` ensemble
decoding into probabilities and speed, and deltas from the reproduced baseline.
The genuine decoder applies the captured brake threshold and optional speed
factor. These reports describe input sensitivity at one observed state; they
do not estimate route score, interacting traffic, or PID controls from a reset
controller. Sparse navigation cues are not an exact reference trajectory.

There is also an explicit `--decoder-features` diagnostic mode. It invokes the
actual `net.adapt_decoder` with the captured BEV/radar tensors and its captured
pre-decoder RNG state. It has the same baseline/restoration checks, but labels
`mode=decoder_features` and `full_sensor_replay_status=not_run_in_this_invocation`.
This can investigate a full-model parity discrepancy after its report is
reviewed; it neither reconstructs sensor feature extraction nor establishes
full-model parity. The tool never falls back to this mode automatically. Keep
its output in a separate directory beside the full-model attempt.

The September 9 navigation pilot captured 33 frames each from routes 14842 and
4468. Their baseline scores remained 36.34 and 100. Full sensor replay failed
bitwise equivalence, while the explicit captured-feature decoder replay passed
all 66 baseline/restoration checks. Neither navigation option restored a
leftward path at the failed route's captured critical frames. Keep the legacy
navigation defaults for this checkpoint; these frozen-state results do not
estimate the score of an alternative rollout.
