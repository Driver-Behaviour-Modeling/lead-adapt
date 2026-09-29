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

**v1 failed:** `adapt_token_decoder_v1` (launched 2026-09-29 04:28, `EXTRA_CONFIG="epochs=6"`, wandb `wjavav2a`) ran all 45,408 steps but the token CE went NaN at step 18,078 (epoch 2) and stayed NaN. Up to then it was healthy: CE 1.58 → 1.18 mean over epochs 0–1 (≈ 1.0 at the end of epoch 1), token accuracy up to 70%; the loss was steady at ≈ 1.0 right up to the NaN, so it was a single bad step, not a slow blow-up. The only surviving checkpoint (`model_0005.pth`) has NaN in all 113 token-decoder and output-projection tensors (frozen modules are fine). Do not evaluate it.

Causes:
- bf16 mixed precision disables the GradScaler (`need_grad_scaler` is fp16-only), so `scaler.step` never checks for inf/NaN and the NaN update went straight into AdamW. `gradient_steps_skipped` only ever counted fp16 scale backoffs (it read 0).
- No gradient clipping.
- `epoch_checkpoints_keep` is `[]` in leaderboard mode, so each epoch deleted the previous checkpoint, including the good epoch-1 one.

Fixes (uncommitted): `train.py` unscales, computes the global grad norm (logged as `debug/grad_norm`), clips to `grad_clip_norm`, and skips the step when the norm is non-finite and no GradScaler is active. New config keys `grad_clip_norm` (default None) and `keep_all_epoch_checkpoints` (default false). The launch script now sets `grad_clip_norm=1.0 keep_all_epoch_checkpoints=true epochs=3`, `RUN_NAME` defaults to `adapt_token_decoder_v2`, and it has an `#SBATCH` header (`h100-full`, QOS `professor-fullgpu-limited`, the full H100 NVL).

**Next run:** `adapt_token_decoder_v2`, launched directly on the full H100 from the repo root: `CUDA_VISIBLE_DEVICES=GPU-4ec20303-3c97-fc1c-9499-e5539efe3717 nohup bash scripts/posttrain_token_decoder.sh > outputs/adapt_token_decoder_v2.log 2>&1 &`. Pin the GPU by UUID: `CUDA_VISIBLE_DEVICES=1` selects a 10 GiB MIG slice, not the second H100 (log `outputs/adapt_token_decoder_v2.log`, checkpoints in `outputs/local_training/adapt_token_decoder_v2/`, 3 × 7,568 steps ≈ 3.5 h).

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
