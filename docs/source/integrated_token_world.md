# Navigation, vehicle futures, and the normative token reference

This implementation starts from main (`61c250a`), retaining the history-free
decoder and body-frame vocabulary. It incorporates the vehicle predictor from
`3f05125`, adds dedicated navigation modulation to the token context, and provides
scoring components based on `DriverBehaviorModeling-2.pdf` and its revision
`DriverBehaviorModeling-3.pdf`. The PDFs specify the scoring method; they do not
specify a vehicle-world-model anomaly detector. The interaction channel below is
an extension proposed here.

## Architecture

```text
camera/LiDAR/radar + current state -> perception -> scene context
navigation points + commands -> shared navigation encoder -> token-context FiLM
radar queries -> vehicle futures -> confidence-weighted future attention
                    |                            |
                    +----------------------------+ -> enriched context
                                                        |
                         learned START -> AR motion-token distribution
                                                        |
                                               integrated ego trajectory
```

The route/speed heads remain available as auxiliary training tasks and a control
baseline. The primary comparison drives with the token trajectory. This lets
navigation and world information affect the distribution that is both scored
and driven, rather than improving only the separate route head. Navigation
commands must express the externally supplied route intent before the action;
they must not be inferred retrospectively from the driver's maneuver.

Both new residuals start at identity. Old token-model parameters, including its
1,234-code vocabulary, remain intact. Actor futures contain predicted positions,
interval velocities and mode/vehicle/detection confidence only. Future labels
enter supervision, never perception or the prediction memory. The world model
still forecasts other traffic without conditioning on candidate ego actions;
it is not a counterfactual interactive simulator.

## Token targets and the PDFs

Both PDFs define the step token as the nearest primitive to the observed motion
over one 0.25-second interval. Main's rolling tokenizer instead optimizes the
entire two-second sequence; its first label can depend on later observed poses.
`adapt_rolling_causal_first=true` fixes its first token to the nearest observed
step while retaining sequence recovery for subsequent tokens. This is opt-in
because it changes targets and requires retraining and fresh calibration.

The step scorer uses the physical nearest token and requires the new target
configuration. It retains hard cross-entropy and pure teacher forcing. No
trajectory-MLP loss is added to the categorical probabilities. The future block
tokens after the first retain a rolling representation, so treating their joint
likelihood as the PDFs' sequence of independently quantized physical steps is
not yet justified. `ensemble_block_log_likelihood` implements the correct mixture
algebra, but a complete physical-step block-scoring experiment remains to be
specified and validated.

## Two complementary anomaly channels

1. **Ego normativity:** the revised PDF's temperature-scaled and geometrically
   smoothed distribution. `lead/scoring/distributions.py` evaluates the kernel in
   log space, averages member probabilities, computes total/within-member entropy
   and their difference, and reports `s = -log P(u) - A`. Smoothing and temperature
   change scoring only. They do not silently alter the control policy.
2. **Traffic interaction:** compare the observed timed ego path against all
   predicted vehicle modes. `world_interactions.py` returns an expected number
   of conflicting actors and a soft version, using oriented separating axes.
   These are geometric proxies with assumed vehicle dimensions; stationary
   orientation is unknown and uses an enclosing circle. They are not probability
   densities, calibrated collision probabilities, or evidence of causal blame.
   Keep this channel separate and calibrate its false-alarm rate on normal
   drives before combining verdicts.

The old vehicle head uses a best-mode regression objective. Neither its minimum
trajectory error nor its mode softmax supplies a calibrated density of observed
actor futures. Unexpected actor motion and unsafe ego action are distinct
questions. A future probabilistic vehicle model and temporal association would
be needed for a principled actor-surprisal channel.

Three actor modes are not three epistemic ensemble members. The revision's
uncertainty decomposition needs independently trained model seeds. One member
works as a baseline, with epistemic uncertainty exactly zero, but cannot test
the revision's OOD-masking claim. Even independent ensembles can agree wrongly;
the uncertainty decomposition does not guarantee OOD detection.

## Calibration and evidence required

Use disjoint **training**, **normal validation**, **normal calibration**, **normal
drive-reference**, and **evaluation** drives. Fit temperatures and smoothing by
NLL on validation drives only. Fit uncertainty-bin thresholds on calibration
drives. `SceneCalibration` uses only pre-action uncertainty to choose a bin,
explicitly pools sparse bins, rejects evaluation/calibration drive overlap, and
reports upper-tail probabilities, excess above threshold, drive U, and the
low-confidence fraction. Whole-drive reference percentiles require a separate
normal drive set. Within-drive dependence means per-step exchangeability and
binwise coverage must be checked, not assumed. Do not restore the Gaussian
drive-Z assumption that failed on Stage 1.

The capture command does not fit parameters or manage partitions. The separate
`calibrate_integrated_reference.py` command fits thresholds and evaluates held-out
binwise flags, verifies checkpoint/scoring-parameter identity, and rejects drive
overlap. Use `--channel world_conflict_count` to calibrate the interaction proxy
as its own channel. Combining channel flags needs its own false-alarm budget.

Before claiming improvement:

- Fine-tune the combined checkpoint and measure vehicle forecast error: its
  imported predictor was trained on another backbone's features.
- Evaluate token control against e26 on identical routes/seeds, including all
  five sequential-lane-change routes. Report all route attempts and missing
  routes, rather than selecting successful reruns.
- Ablate token navigation and future fusion individually, with matched training
  if attributing architectural gains. A frozen-checkpoint switch is diagnostic.
- Refit score parameters on normal validation drives, then evaluate false flags
  per uncertainty bin and scenario on unseen normal drives.
- Test graded and sustained anomalies, rare legal maneuvers, and OOD masking.
  Offline motion perturbations do not replace closed-loop BADCAR evaluation.

The previous 91.47 world-model driving score used different weights and control
heads. It is evidence motivating the combination, not this model's score.

## Initialization and verification

The prepared initialization is at
`/localstorage/home/f20221129/lead-adapt/outputs/checkpoints/integrated_token_world_init`.
It combines history-free epoch 26's 1,006 tensors with 22 world-predictor tensors
from `adapt_world_v1/model_0059.pth`. `initialization.json` records their hashes.
It imports neither the older ego decoder/codebook nor trained future-fusion
weights. The new fusion and token-navigation residuals start at identity.

Recreate in a new output folder:

```bash
PYTHONPATH=. python scripts/prepare_integrated_checkpoint.py \
  --ego-checkpoint /path/to/adapt_fresh_histfree_v1/model_0026.pth \
  --world-checkpoint /path/to/adapt_world_v1/model_0059.pth \
  --output /path/to/new_initialization
```

`scripts/verify_integrated_model.py` checks strict loading, exact full-model
inference identity against e26, finite real-data losses/gradients, and gradients
reaching both new context branches. It performs no optimizer update and does
not establish driving performance.

From this checkout, the first fine-tuning command is:

```bash
CUDA_VISIBLE_DEVICES=GPU-4ec20303-3c97-fc1c-9499-e5539efe3717 \
INIT_CHECKPOINT=/localstorage/home/f20221129/lead-adapt/outputs/checkpoints/integrated_token_world_init/model_init.pth \
RUN_NAME=adapt_token_world_v1 SEED=0 EPOCHS=5 LR=2e-5 \
bash scripts/posttrain_integrated_token_world.sh
```

Five epochs at 2e-5 are an initial experiment, not a proven optimum. The script
does not start automatically. Use `DRY_RUN=1` to inspect the configuration.
Independent ensemble runs use distinct seeds and run names.

The first experiment was launched on 2026-10-24 (02:15 IST) after selection of
the e26 initialization. It runs in `lead-adapt-integrated` on branch
`integrated-token-world`, using the full H100, batch 128, 32 data workers and
prefetch factor 4. Image logging is every 1,000 steps; scalar logging is every
step. The detached launcher does not depend on the chat or terminal remaining
open. Its log is `lead-adapt/outputs/adapt_token_world_v1.log` and its W&B run is
`divyanshu_div-penn/lead_posttrain/1pf5b1wz`.

`lead-adapt/outputs/local_training/adapt_token_world_v1` contains the effective
training config, `launch.json`, initialization hashes, a source snapshot and a
SHA-256 manifest. No optimizer state is inherited from e26. The batch-128 probe
used 55.8 GiB peak allocated GPU memory and about 0.68 seconds per step after
warmup; this excludes normal training augmentation and visualization overhead.
Driving and held-out anomaly performance have not yet been measured.

For an interrupted run, resume from its latest **completed epoch** with
`RESUME_CHECKPOINT=/absolute/path/to/model_XXXX.pth`, keeping the same `RUN_NAME`,
`EPOCHS=5` and `LR=2e-5` in the command above. The script restores the saved
optimizer/scheduler state. Work since that epoch checkpoint is repeated.

After training, use a one-model directory per checkpoint, then capture scores:

```bash
PYTHONPATH=. python scripts/score_integrated_reference.py \
  --checkpoint-dirs /path/to/member0 /path/to/member1 \
  --temperatures 1.25 1.25 --epsilon 0.01 --sigma 0.1 \
  --data-root /path/to/normal_validation_drives \
  --output outputs/surprisal/integrated_validation --save-member-logits
```

The example temperatures/smoothing values are placeholders to fit on validation
data, not calibrated settings. Outputs contain raw score components and world
interaction features; metadata explicitly marks them as uncalibrated.

### Frozen normal-drive protocol

The initial drive manifest is
`lead-adapt/outputs/diagnostics/integrated_token_world/normal_drive_partitions.json`.
It selects 197 expert recordings that completed without recorded infractions
except minimum-speed warnings. Roles are assigned by scenario and route group;
repeated recordings of the same route cannot cross roles:

| Role | Drives | Scorable frames |
| --- | ---: | ---: |
| Parameter validation | 50 | 3,932 |
| Threshold calibration | 49 | 3,969 |
| Whole-drive reference | 49 | 3,607 |
| Test | 49 | 3,574 |

The underlying 200-drive cache also contains two tick-runtime failures and one
drive with a lane departure, all `VehicleOpensDoorTwoWays`. Their 1,637 frames
are captured for provenance but excluded from all four roles. This leaves no
clean recording of that scenario in this protocol. The older Stage-1 analysis
used these drives; the new split is not a pristine benchmark. Recording paths
are disjoint from training; physical-road overlap is not excluded. The strict
infraction filter is a data-selection rule, not proof that every selected action
is normative or that every excluded lane departure was inappropriate.

`scripts/partition_integrated_drives.py` recreates the manifest in a new file.
`scripts/evaluate_integrated_reference.sh` captures raw logits, vocabulary,
observed ego poses, predicted actor futures and current-assignment forecast
errors. It fits bounded temperatures and geometric smoothing by validation NLL
with `fit_integrated_reference.py`, then rescores each frozen role. This is a
coordinate search with a documented grid, not a global-optimum claim. Thresholds
and whole-drive percentiles use the other roles. The calibration command rejects
parameter-fitting/evaluation overlap and reports scenario flag rates and a
whole-drive bootstrap interval.

The first complete initialization baseline is
`lead-adapt/outputs/surprisal/integrated_init_baseline_v1`. Its ego outputs are
initially identical to e26; its imported world head has not adapted to e26's
backbone. Validation selected temperature 1.3188, epsilon 0.01 and sigma 0.25;
validation NLL fell from 1.73585 to 1.66507. On the 49 held-out test drives:

| Channel | Normal-step flag rate | Drive-bootstrap 95% interval |
| --- | ---: | ---: |
| Ego normativity | 8.23% | 5.44–11.96% |
| Predicted traffic conflict | 7.50% | 4.94–10.65% |

The nominal target is 5%; the rates do not establish adequate calibration.
Scenario rates vary substantially: ego flags reach 51% on the held-out
`InvadingTurn` recording and conflict flags 61% on `EnterActorFlow`. These are
diagnostic observations, not grounds to refit parameters on test data. One model
has epistemic uncertainty exactly zero. The old head has minADE 1.91 m,
most-likely ADE 2.59 m and most-likely FDE 4.45 m over 16,270 supervised actor
appearances (17,731 current vehicle labels). Actors repeat across frames; missing
future labels and actors beyond the query limit are not evaluated.

`scripts/evaluate_integrated_perturbations.py` evaluates graded acceleration,
braking and positive/negative-y curvature perturbations. Zero perturbation
preserves the recorded motion, braking does not reverse the vehicle, and replay
windows never cross recording gaps. It verifies that reconstructed unmodified
ego scores and interaction evidence match the saved captures. Only held-out
actions are changed; normal window references come from the separate drive
reference set. Output is `perturbations.json` in the baseline directory.

The ego reference is sensitive to graded swerves and strong braking. The
interaction channel responds to predicted traffic clearance: braking often
reduces it, as can swerving away from traffic. Thus this channel measures a
specific kind of predicted interaction risk, rather than all behavioral
deviations. Overlapping replay windows are not reacting closed-loop rollouts,
BADCAR results, AR block likelihoods, or independent observations. Real anomaly
and OOD-masking evaluations remain outstanding.

### Queued post-training checks

`scripts/after_integrated_training.py` is running with the original trainer's
PID and process start time. It waits for that exact process to terminate and
requires the final completed epoch's model, optimizer, scheduler, scaler and
skip-count files. It does not restart training. Its live state is
`lead-adapt/outputs/diagnostics/integrated_token_world/posttrain_evaluation/status.json`.

Once training finishes, it stages exactly one checkpoint for reference scoring
on the GPU slice and runs driving diagnostics on the full GPU. Both e26 and the
combined model use beam width 8 and waypoint control for steering, throttle and
braking, on routes 17655, 17569, 17563, 17635 and 17598 with seed 0. The combined
model also gets separate frozen world-fusion and navigation ablations. Each
route/variant gets one attempt, preserving infrastructure errors. A real import
and configuration preflight passed for e26; no simulator was started during that
check. Five-route results will not establish a 220-route improvement.

Fit and test calibration on separate normal-drive captures:

```bash
PYTHONPATH=. python scripts/calibrate_integrated_reference.py fit \
  --features outputs/surprisal/normal_calibration \
  --output outputs/surprisal/ego_calibration.json
PYTHONPATH=. python scripts/calibrate_integrated_reference.py evaluate \
  --calibration outputs/surprisal/ego_calibration.json \
  --features outputs/surprisal/normal_test \
  --output outputs/surprisal/ego_test.json
```

For drive percentiles, first evaluate a separate normal-drive reference capture
and pass that JSON with `--normal-drive-reference` during test evaluation. The
tool checks drive-set separation; it cannot prove that supplied drives are
normal or that they were absent from training and parameter fitting.
