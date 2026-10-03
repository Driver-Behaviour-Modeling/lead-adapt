# lead-adapt status

State of the ADAPT decoder in `lead-adapt`, and what has been changed so that its token distribution can serve as the reference model for the unruliness surprisal (`paper/unruliness_surprisal.tex`). Last updated 2026-09-29. The fixes are on `main` (based on `a2a208d`) and are uncommitted. The `world-model` branch (radar vehicle world model, `3f05125`) is separate and not part of the reference model.

## Decisions

- **The paper is about the scoring method.** The tokenizer is not a contribution (k-disks follows SMART/AutoVLA).
- **Reference model: the fixed lead-adapt.** All experiments run on it.
- **Dataset (BADCAR):** a PDM-Lite expert with rule-based anomalies of several types injected at random. Types and severity λ are still to be specified.
- **What the score needs from the model** (see `paper/unruliness_surprisal.tex`): a proper conditional distribution P(u_t | u_<t, c_t) over motion tokens. So:
  - hard cross-entropy (a proper scoring rule);
  - pure teacher forcing (autoregressive conditionals, not per-step marginals);
  - a normative-only codebook;
  - identical tokenization in codebook construction, training and scoring.
- **Decoder input slots stay token embeddings** (not navsim's `pose_proj` of continuous deltas). The model is then exactly P(u_t | u_<t, c_t), inputs are always in-distribution, and block-score sampling feeds back tokens exactly as in training.
- **Training scope (first run): token decoder only.** Start from `adapt_radar_navcond_v1_retry1/model_0014.pth` (88.9 DS). Freeze everything except the AR transformer decoder and output projection, and keep the frozen modules in eval mode. Route/speed control, and so the driving score, are identical to the loaded model.
- **Future work: full fine-tune.** Also try training the whole model with the token objective (plus the driving losses) and re-evaluate driving. The frozen-context run is the first, cleanest reference, not the last word.
- **Trajectory MLP is not trained in the reference model.** Under pure teacher forcing it would read hidden states conditioned on ground-truth future tokens (a leak) and push the decoder toward copying its input. Its losses are off, and `pred_future_waypoints` is produced by integrating the argmax token rollout instead.
- **Codebook:** tolerance τ = 0.05, `min_cluster_size` = 25, chosen from the sweep below.
- **Scoring and calibration code is out of scope for now.** The model is trained and evaluated like any lead/ADAPT model. Scoring comes after the anomaly dataset.

## What changed (on `main`, uncommitted)

| Item | Change | Files |
|---|---|---|
| B1 delta frame | `_compute_deltas` produces per-step body-frame deltas `R(ψ_k)ᵀ(p_{k+1} − p_k)`, matching the codebook extractor. Legacy codebooks (no `frame` tag) keep the unrotated differences, so old checkpoints reproduce exactly. | `lead/kdisks/kdisks_codebook.py` |
| Codebook guards | Refuse a codebook whose recorded heading weight differs from `kdisks_heading_weight`. The decoder refuses `kinematic_vocab_size` ≠ number of codes. | `kdisks_codebook.py`, `lead/adapt/adapt_decoder.py` |
| B2 history timing | Training history ages are [20, 15, 10, 5, 0] ticks, ending at the current pose. This equals the default `legacy` runtime window. Every history delta and the first future delta span 0.25 s. `training_aligned` ([25…5]) is kept only to evaluate older checkpoints. | `lead/common/history_features.py`, `carla_dataset.py`, `sensor_agent.py` (comments), `tests/test_history_features.py` |
| B3 codebook | Clustering ported from navsim: minimum cluster size, reserved stationary token, plausibility bounds rescaled to 0.25 s steps, no single-sample padding, perplexity and `frame='body'` recorded. Heading weight = CARLA ego half-diagonal √(2.451² + 1.064²) = 2.672 m/rad. | `lead/kdisks/kdisks_clustering.py`, `scripts/build_kdisks_carla.py` |
| Delta extraction | Parallel, and restricted to the routes the training buckets use (failed or unfinished routes are not normative). Writes `lead/adapt/data/carla_deltas_valid_routes.npy` (1,099,472 deltas from 8,718 of 9,715 routes). Global poses were checked against the dataloader's ego-frame futures: they agree to ~1 mm. | `scripts/extract_lead_carla_deltas.py` |
| New codebook | `lead/adapt/codebooks/kdisks_carla_body.pkl`: 1,234 codes, 26,057 samples below `min_cluster_size` (they snap to the nearest code), stationary token holds 331,874 samples (~30%). Old `kdisks_carla.pkl` and `carla_deltas.npy` untouched. | — |
| Config | Defaults now point at the new codebook (`kinematic_vocab_size = 1234`, heading weight 2.672). New flags `adapt_train_token_decoder_only` (freeze everything but the token decoder, CE-only loss weights, refuses scheduled sampling) and `adapt_waypoints_from_tokens`. | `lead/training/config_training.py` |
| Freezing | Applied after checkpoint load. Frozen modules are put back in eval mode every epoch, so BatchNorm statistics don't drift. `AdaptDecoder.training` is set on its own (non-recursively), because `forward` uses that flag to choose teacher forcing. The first smoke run missed this: the decoder rolled out and trained CE against its own argmax (loss ≈ 0.005, accuracy 1.0). | `lead/training/training_utils.py`, `lead/training/train.py` |
| Waypoints from tokens | `integrate_body_frame_deltas` (inverse of `_compute_deltas`) integrates the argmax rollout from the current pose. The trajectory MLP is frozen when this is on. | `adapt_decoder.py` |
| Launch script | `scripts/posttrain_token_decoder.sh`: 15 epochs, lr 1e-4, single cosine cycle, batch 128. | new |
| Tests | Extractor deltas equal model deltas on a turn through ±π; integration inverts deltas; token-rollout waypoints. The full suite passes (373, excluding the untracked world-model tests, which need the other branch). | `tests/test_kdisks_tokenization.py` |

> **The loaded checkpoint's `config.json` is the base config for training.** Keys not overridden in `LEAD_TRAINING_CONFIG` are inherited from the checkpoint, not from the code defaults. The launch script therefore sets the codebook path, heading weight and vocab size explicitly. The same applies to any future run started from an old checkpoint.

> **Codebook centroids are a registered buffer and are saved in checkpoints.** Loading a checkpoint built with a different codebook of the *same* size would silently replace the centroids from the pickle. Here the sizes differ (4096 vs 1234), so they are dropped and re-read.

### Codebook sweep

Clustered on 90% of the deltas; residual r measured on the held-out 10%. "r > 2τ" is the share of *normal* steps outside the codebook's guaranteed cover (the residual's false-alarm floor).

| τ | n_min | codes | codes with <100 samples | median r | r p99 | r > 2τ | resolution |
|---|---|---|---|---|---|---|---|
| 0.03 | 10 | 3078 | 2402 | 0.013 | 0.114 | 1.44% | 0.12 m/s |
| 0.03 | 25 | 1561 | 704 | 0.013 | 0.346 | 3.17% | 0.12 m/s |
| **0.05** | **25** | **1167** | **481** | **0.021** | **0.139** | **1.02%** | **0.2 m/s, 0.075 rad/s** |
| 0.05 | 10 | 1937 | 1397 | 0.020 | 0.071 | 0.45% | 0.2 m/s |
| 0.08 | 25 | 736 | 261 | 0.032 | 0.093 | 0.38% | 0.32 m/s |

Other configurations (n_min = 50) were strictly worse on cover. Pose glitches (|Δh| > 0.5 rad or reverse > 0.5 m per step) make up about 0.01% of deltas.

## Current run and follow-ups

### Runs at a glance (AR-decoding view)

Token decoding at inference is the same for every token run:
- The prefix is 4 true history tokens (ages 20→15→10→5→0 ticks).
- Then 8 steps: logits → **argmax** token → its embedding is fed back → next step.
- The 8 argmax centroids (body-frame deltas) are integrated from the current pose into 8 waypoints at 0.25 s spacing, re-planned at every control step.

| Run | Start | What trains | AR decoder training | Codebook | Driving trajectory | B2D DS |
|---|---|---|---|---|---|---|
| navcond (`adapt_radar_navcond_v1_retry1`) | chain: pretrain → 60 → 40 → 15 epochs | everything | hard CE + MLP trajectory L1/Fréchet, **scheduled sampling to 1.0 (argmax feedback)** | old: 4096 codes, current-frame deltas | route + speed heads (non-AR queries) | 88.9 |
| `navcond_waypoint` | same checkpoint | — | — | — | **trajectory MLP** on the AR hidden states of the greedy rollout (MLP trained on rollout states via SS) | 84.9 |
| v2 | navcond `model_0014` | token decoder only (rest frozen, eval mode) | hard CE, **pure teacher forcing**, 3 ep, lr 1e-4 | new: 1234 codes, body frame | integrated argmax token rollout | 62.6 |
| v3 | v2 `model_0002` | token decoder only | same, +3 ep, lr 5e-5 | new | integrated argmax rollout | 64.4 |
| `full_v1` (waypoint) | v3 `model_0002` | everything but the unused MLP | CE + all perception/route/speed losses, pure TF, 5 ep, lr 2e-5 | new | integrated argmax rollout | 67.1 |
| `full_v1` (route) | same checkpoint | — | — | — | route + speed heads | 88.8 |
| `adapt_fresh_posttrain_v1` (running) | LEAD pretrain `model_0030` (perception only) | everything; decoder, route/speed, navigation from scratch | CE + all losses, pure TF, 30 ep, lr 1e-4, cuDNN SDPA off | new | — (not yet evaluated) | — |

**Decoding options (uncommitted, inference only).**
- `adapt_decoding = greedy | beam | expected` (default `greedy`) and `adapt_beam_width` (default 8). Set per eval with `LEAD_TRAINING_CONFIG="adapt_decoding=beam"`; the agent applies it over the checkpoint's `config.json`.
- **Beam** returns the most likely 8-token sequence under P. This is the paper-consistent planner: the car drives the mode of the same density the score uses.
- **Expected** is a per-step mean, which averages modes, so it is an ablation only.
- Tests: width 1 equals greedy, and an exhaustive width finds the brute-force most likely sequence.
- `scripts/eval_open_loop_rollout.py` (Step A) reports the per-horizon error of the GT-token floor, teacher forcing, each decoding and a reference checkpoint, plus token accuracy and teacher-forced NLL.

Training token CE (train set, last epoch): v2 1.13, v3 1.05, `full_v1` 0.99, fresh 0.92 at epoch 20.

**v1 failed:** `adapt_token_decoder_v1` (launched 2026-09-29 04:28, `EXTRA_CONFIG="epochs=6"`, wandb `wjavav2a`) ran all 45,408 steps but the token CE went NaN at step 18,078 (epoch 2) and stayed NaN. Up to then it was healthy: CE 1.58 → 1.18 mean over epochs 0–1 (≈ 1.0 at the end of epoch 1), token accuracy up to 70%; the loss was steady at ≈ 1.0 right up to the NaN, so it was a single bad step, not a slow blow-up. The only surviving checkpoint (`model_0005.pth`) has NaN in all 113 token-decoder and output-projection tensors (frozen modules are fine). Do not evaluate it.

Causes:
- bf16 mixed precision disables the GradScaler (`need_grad_scaler` is fp16-only), so `scaler.step` never checks for inf/NaN and the NaN update went straight into AdamW. `gradient_steps_skipped` only ever counted fp16 scale backoffs (it read 0).
- No gradient clipping.
- `epoch_checkpoints_keep` is `[]` in leaderboard mode, so each epoch deleted the previous checkpoint, including the good epoch-1 one.

Fixes (uncommitted): `train.py` unscales, computes the global grad norm (logged as `debug/grad_norm`), clips to `grad_clip_norm`, and skips the step when the norm is non-finite and no GradScaler is active. New config keys `grad_clip_norm` (default None) and `keep_all_epoch_checkpoints` (default false). The launch script now sets `grad_clip_norm=1.0 keep_all_epoch_checkpoints=true epochs=3`, `RUN_NAME` defaults to `adapt_token_decoder_v2`, and it has an `#SBATCH` header (`h100-full`, QOS `professor-fullgpu-limited`, the full H100 NVL).

**v2 and v3 (done, healthy):**

| Run | Start | Epochs, lr | Token CE per epoch | Token accuracy | Skipped non-finite steps |
|---|---|---|---|---|---|
| `adapt_token_decoder_v2` | navcond `model_0014` | 3, 1e-4 | 1.55 → 1.19 → 1.13 | 0.57 → 0.64 | 0 |
| `adapt_token_decoder_v3` | v2 `model_0002` | 3, 5e-5 | 1.11 → 1.07 → 1.05 | 0.64 → 0.65 | 91 (guard worked) |

All `model_0002.pth` weights are finite. **Latest: v3 `model_0002.pth`**. v2 `model_0002` is on Hugging Face (`Divyanshudiv/ADAPT`, model repo, `adapt_token_decoder_v2/`). The waypoint-modality eval of v2 (`outputs/eval_v2_e2.log`) stopped after 1 of 220 routes. Clipping at 1.0 is active on most steps (mean pre-clip grad norm ≈ 1.1).

**Full fine-tune (next):** start from v3 `model_0002`, set `adapt_train_token_decoder_only=false`, 5 epochs, lr 2e-5 (the navcond post-train lr), keeping grad clipping, pure teacher forcing and waypoints from tokens. A dry run confirmed 73.9 M of 76.0 M parameters trainable (all but the unused trajectory MLP). All losses are weighted equally (token CE ≈ 0.08 of the total, like every perception and route/speed term). The run name is `adapt_token_full_v1`.

**`adapt_token_full_v1` (done):** token CE 1.065 → 0.991, accuracy 0.668, route and speed losses improving. The final `model_0004` is finite and uploaded to HF. But skipped steps rose every epoch (178, 196, 793, 1004, 1015; 13% of the last epoch).

**Root cause of skipped steps (fixed, uncommitted):** the **cuDNN attention backend in torch 2.5.0** returns non-finite gradients from the ADAPT decoder's bf16 attention backward.
- Every skipped step had a finite loss.
- Per-loss attribution at the final weights: 44/300 steps non-finite, all from `loss_kinematic_token`, then propagating into the backbone and radar.
- On 8 bad batches with a fixed dropout seed, re-running reproduced the failure each time. Each became finite with cuDNN SDPA excluded and with math SDPA.
- Fix: `torch.backends.cuda.enable_cudnn_sdp(False)` in `training_utils.initialize_torch` (off by default from torch 2.5.1). With it: 0/300 non-finite, 0.949 s vs 0.877 s per step (≈ 6% more useful updates per hour, since 13–15% of cuDNN steps were discarded).
- The token-only runs saw little of this (v2: 0, v3: 91) because only the decoder received gradients. The checkpoints already trained are valid; skipped steps only drop updates.

**Held-out data caveat:** every run so far (navcond, v1–v3) trained on the full post-train buckets, *including* Town13. Town13 is therefore not held out. A fair NLL comparison between v3 and the full fine-tune needs data neither model saw (for example BADCAR's normal drives), or both retrained with `hold_out_town13_routes=true`.

**Closed-loop eval: v2 driving from its tokens** (Bench2Drive-220, waypoint modality for steer/throttle/brake, run on the ADAPT machine, 2026-10-01). The first attempt failed because `CARLA_ROOT` pointed at a missing `carla_0.9.16`.

| | v2, token rollout (waypoint mode) | navcond, route + target speed |
|---|---|---|
| Driving score | 61.7 (217/220 routes; current 62.6) | 88.9 |
| Success rate | 29.5% | 70.9% |
| Route completion / penalty | 85.3 / 0.73 | 92.4 / 0.96 |
| Layout / vehicle / pedestrian collisions | 74 / 102 / 5 | 1 / 12 / 0 |
| Outside lanes / blocked / TickRuntime | 61 / 22 / 18 | 7 / 4 / 3 |
| Route deviation / red light | 38 / 5 | 34 / 4 |

Navigation intent and rules are about as good as navcond (route deviation and red lights barely change). The loss is in tracking and interaction: collisions, lane departures, and turns, parking exits and control loss at 21–35 DS. **Confound:** waypoint mode uses the old TransFuser controller (one aim point and PID; speed from the distance between the 0.5 s and 1 s waypoints), not LEAD's route controller. Control experiment: navcond in waypoint mode. Then v3 and `adapt_token_full_v1` in waypoint mode.

**Closed-loop eval round 2 (ADAPT machine, Bench2Drive-220, 2026-10-03):**

| Run | Driver | n | DS (cur) | SR % | RC | Penalty | Layout / vehicle collisions | Outside lanes | Route dev |
|---|---|---|---|---|---|---|---|---|---|
| navcond (other box) | route + speed heads | 220 | 88.9 | 70.9 | 92.4 | 0.96 | 1 / 12 | 7 | 34 |
| `full_v1_route` | route + speed heads | 220 | **88.8** | **72.3** | 92.5 | 0.96 | 1 / 15 | 1 | 34 |
| `navcond_waypoint` | old MLP waypoints | 219 | 84.9 | 62.3 | 90.9 | 0.93 | 2 / 19 | 9 | 36 |
| v2 | token rollout | 217 | 62.6 | 29.5 | 85.3 | 0.73 | 74 / 102 | 61 | 38 |
| v3 | token rollout | 217 | 64.4 | 31.8 | 86.6 | 0.74 | 62 / 95 | 48 | 40 |
| `full_v1_waypoint` | token rollout | 220 | 67.1 | 30.9 | 87.9 | 0.77 | 52 / 80 | 57 | 36 |

- **The full fine-tune keeps the driving policy:** 88.8 vs 88.9 DS (within eval noise), with slightly higher success.
- **The waypoint controller costs only ~4 DS** (navcond 88.9 → 84.9). It is **not** the main reason the token rollout is weak, so that confound is largely ruled out.
- **The token rollout trails the MLP waypoints by ~18 DS under the same controller.** The loss is collisions (layout, vehicle) and lane departures. Route deviation and red lights are on par.
- **Rollout driving improves as token CE falls** (1.13 → 1.05 → 0.99 gives 62.6 → 64.4 → 67.1).
- **Remaining suspects:** exposure bias (pure teacher forcing vs the scheduled-sampling-trained MLP head), greedy argmax decoding, and quantization drift integrated over 8 steps.

**Earlier: v2 launch command.** `adapt_token_decoder_v2` was launched directly on the full H100 from the repo root: `CUDA_VISIBLE_DEVICES=GPU-4ec20303-3c97-fc1c-9499-e5539efe3717 nohup bash scripts/posttrain_token_decoder.sh > outputs/adapt_token_decoder_v2.log 2>&1 &`. Pin the GPU by UUID: `CUDA_VISIBLE_DEVICES=1` selects a 10 GiB MIG slice, not the second H100 (log `outputs/adapt_token_decoder_v2.log`, checkpoints in `outputs/local_training/adapt_token_decoder_v2/`, 3 × 7,568 steps ≈ 3.5 h).

**After it finishes:**
1. **Held-out token NLL per checkpoint.** Measure it on the Town13 routes (a normal held-out loss, not the calibration experiments). Use it to pick the checkpoint and to judge whether 6 epochs was enough. The trainer has no validation loop, so this script still has to be written.
2. **Drive with the token distribution.** Run the closed-loop eval with `LEAD_CLOSED_LOOP_CONFIG="steer_modality=waypoint throttle_modality=waypoint brake_modality=waypoint"`, so the car follows the argmax token rollout (`adapt_waypoints_from_tokens`). The default modalities reproduce navcond exactly (≈ 88.9 DS) and prove nothing new.
   - Driving tests the distribution's *mode*, not its calibration.
   - Pure teacher forcing may cost some rollout quality (exposure bias). If the car drives poorly, that points to exposure bias, not necessarily a bad density.
3. **Full fine-tune** (listed under Decisions). Train the whole model with token cross-entropy plus the driving losses. This lets the context adapt to token prediction and removes the route/speed heads' old history-window mismatch (trained on [25…5], run on [20…0]). Then re-evaluate driving in both modalities.
4. **Paper wording.** In the default configuration the scored token head shares the network with, but is not, the head that drives. State the claim as "a token distribution trained on normative driving", unless (2) or (3) shows the token rollout drives competently.
5. **Clean-up:** delete `outputs/local_training/smoke_token_decoder` and `smoke_nanguard` (smoke-test output) and `adapt_token_decoder_v1` (NaN weights).

## Remaining items and known limitations

- **Samples without a full future.** `mixed_data_collate_fn` fills missing keys with zeros, and drops a key for the whole batch if shapes differ. Token CE is now skipped (zero) for a batch without ground-truth future tokens, instead of training against the decoder's own argmax as before. A *single* sample missing its future inside a batch would still be zero-filled and tokenized as stationary. In post-train buckets (8 frames skipped at each end) this was measured not to occur (0 of 5,038 frames on 40 routes). Pretraining buckets (`skip_last = 1`) could hit it.
- **Trajectory MLP leak in other configs.** Any run with pure teacher forcing *and* the MLP trajectory losses on has the leak. That is the reason for `adapt_waypoints_from_tokens`.
- **Context leakage (for the paper).** The context holds speed, acceleration, 5 history poses and 4 history tokens. Driver-induced unfamiliar contexts can raise H and forgive an anomaly. Ablate the history length later.
- **Residual blind spot.** A token is a single first-order step (speed, yaw rate). Hard braking or jerk from a normal speed stays in-vocabulary, so only the surprisal of the transition can see it.
- **The `training_aligned` name** now means "the window checkpoints before this change were trained on". It was kept to avoid breaking diagnostics scripts.

## navsim ADAPT vs lead-adapt (reference)

| Area | navsim ADAPT | lead-adapt now |
|---|---|---|
| Delta frame | body frame | body frame (new codebooks) |
| Clustering | min size, stationary token, bounds | same, bounds for 0.25 s |
| Heading weight | 2.831 (nuPlan ego) | 2.672 (CARLA ego) |
| History | includes current pose | includes current pose |
| Decoder input | `pose_proj` of continuous deltas | token embeddings (decision above) |
| Token loss | CE weight 0; loss on softmax expectation | hard CE only |
| Scheduled sampling | ramps to 1.0 | off (pure teacher forcing) |
| Trajectory | integrated softmax expectation | integrated argmax rollout (metric only) |

> **The navsim champion's softmax is not a probability distribution over motions.** With CE weight 0 and the loss on the softmax expectation, any weights with the right mean are equally good, so ℓ and H would be meaningless. Do not port that recipe to the logits used for scoring.

## Open questions

- BADCAR details: anomaly types, λ, ramped vs abrupt onset, sensor suite and rate, and how close PDM-Lite contexts are to the LEAD training distribution.
- Results of the full fine-tune variant (future work above).
