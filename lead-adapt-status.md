# lead-adapt status

State of the ADAPT reference model in `lead-adapt`: what it is, what has been tried, and what is running. The model is the reference distribution for the unruliness surprisal (`analysis/unruliness.tex`). Last updated **2026-10-11**. The previous, chronologically appended version of this file is in git history (commit `36c947a`).

## Summary

- **Model.** LEAD (TransFuser v6) perception and context. Its planning decoder is replaced by route + target-speed query heads (these drive by default) plus an autoregressive (AR) head over a 1,234-code motion vocabulary. The AR head is the scoring density, and it also drives in waypoint mode.
- **Route-mode driving matches the earlier chain.** Bench2Drive-220: `full_v1` 88.8 DS, fresh post-train 88.2 DS (75.5% success, 0 route deviations).
- **The token distribution drives poorly on its own.** Best is `full_v1` at 70–71 DS in waypoint mode, against 84.9 for the old MLP waypoints. The fresh post-train manages only 53 DS.
- **Cause: exposure bias in the 8-step chain.** Open loop, teacher-forced and rollout errors agree at 0.25 s, then diverge (fresh: 0.21 m vs 0.80 m at 2 s).
- **Remedies tried so far:**
  - better decoding (beam, expected): +3–4 DS;
  - short scheduled-sampling fine-tunes: no gain, and scheduled sampling alone degrades the density.
- **Current direction (agreed 2026-10-07):** the score does not need token-by-token history.
  - Condition on scene + current state only.
  - Use LEAD's sensor perturbation as recovery labels.
  - If the AR chain still compounds, move to one token = one whole 2 s trajectory (VADv2-style), keeping a discrete distribution.
- **The history-free decoder fixed token driving:** `adapt_fresh_histfree_v1` e26 drives from its token rollout (beam 8) at **87.3 DS** (217/220 routes), against 53.1 for the fresh run and 70.3 for `full_v1`, on par with the route heads. Why it works (history removal vs rolling tokenization vs recovery labels) is not isolated yet.
- **Bug fixed 2026-10-08:** PyTorch 2.5.0's cuDNN attention also produces **NaN logits at inference** for short decoder sequences (history-free rollouts: ~5% NaN, read as "stop"). Fixed in the model constructor, so it covers training, analysis and the driving agent.
- **Next: calibration of the score** on normative driving, in stages: training frames, then unseen LEAD-expert drives on the 220 Bench2Drive routes, then BADCAR (see "Calibration plan").

## Ongoing experiments

### `adapt_fresh_histfree_v1` (this machine, full H100)

- **Setup:** fresh post-train from LEAD pretrain `model_0030`, identical to `adapt_fresh_posttrain_v1` (30 epochs, lr 1e-4, whole model) except:
  - `adapt_history_free=true`:
    - a learned start token replaces the 4 history tokens;
    - history poses leave the encoder context (for the route/speed heads too);
    - the future starts at the ego origin. Under sensor perturbation (50% of frames; median 0.4 m / 3.5°, up to ~1 m / 11°) the origin is the virtual car, so the targets lead back to the expert path: the same DART-like recovery signal LEAD's route head gets.
  - `adapt_rolling_tokenization=true` (`adapt_rolling_beam_width=32`):
    - Targets are the codebook-step sequence whose reconstruction is closest to the ground-truth poses: summed K-disks pose distance, found by beam search over code sequences.
    - A step-by-step greedy match never recovers from an offset start, because turning towards the path costs heading error at that step. The sequence match recovers within ~1 s at 8–12 m/s, and only partly at 4 m/s.
- **Interrupted** 2026-10-10 13:58 by a machine reboot (SIGHUP at shutdown; new kernel, back up 14:04), at epoch 27, 88% through. Epoch 26 has optimizer, scheduler and scaler state, so it resumes exactly with `continue_failed_training=true load_file=.../model_0026.pth`.
- **Launched** 2026-10-07 23:15 from commit `36c947a`. 73,879,141 trainable parameters. 0 skipped steps so far. About 2.25 h per epoch, so it finishes around **2026-10-10 evening**.
- **Log** `outputs/adapt_fresh_histfree_v1.log`; checkpoints in `outputs/local_training/adapt_fresh_histfree_v1/` (every epoch kept).

**Step A at epoch 9** (2,048 fixed post-train frames, augmentation off). Position error in metres; the fresh run's matched epoch 9 is the baseline:

| | t+1 (0.25 s) | t+4 (1 s) | t+8 (2 s) | ADE | Greedy ÷ TF at 2 s |
|---|---|---|---|---|---|
| fresh e9: teacher forced | 0.048 | 0.146 | 0.272 | 0.161 | |
| fresh e9: greedy / beam / expected | 0.048 / 0.052 / 0.044 | 0.338 / 0.344 / 0.304 | 1.007 / 0.974 / 0.929 | 0.457 / 0.452 / 0.416 | 3.70 |
| **histfree e9: teacher forced** | 0.055 | 0.169 | 0.312 | 0.185 | |
| **histfree e9: greedy / beam / expected** | 0.055 / 0.056 / 0.049 | 0.285 / 0.289 / 0.256 | **0.876 / 0.860 / 0.799** | 0.399 / 0.397 / 0.360 | **2.81** |
| GT-token floor (fresh: independent / histfree: rolling) | 0.015 / 0.018 | 0.046 / 0.016 | 0.099 / **0.017** | 0.054 / 0.017 | |

**Reading:**
- **Rollout:** 13–14% lower error at 2 s, and the compounding ratio falls from 3.7 to 2.8.
- **Teacher forcing:** slightly worse (0.31 vs 0.27 m). Without past motion, the model has less to go on.
- **The quantization floor nearly vanishes** with sequence tokenization (0.10 → 0.02 m at 2 s).
- **Not comparable across runs:** token accuracy and NLL, because the targets are defined differently.
- **The rollout still compounds** (0.055 → 0.88 m), as predicted: the 8-step chain inside each plan is still autoregressive. Judge the final epoch before concluding.

### History-free epoch 26, closed loop: final beam result (D151, 2026-10-11)

**`histfree_e26_beam`, full run** (token rollout drives; waypoint mode, beam 8): 217/220 routes, **DS 86.1 official / 87.3 on evaluated routes**, SR 67.7%, RC 93.5, penalty 0.94.
- Infractions: layout 1, vehicle 19, pedestrian 5, outside lanes 6, route deviation 17, red light 4, blocked 10, TickRuntime 7.
- Versus the token-driving runs: fresh e27 51.4 / 53.1, `full_v1` 70.0 / 70.3.
- Versus navcond's MLP waypoints: 84.5 / 84.9.
- Versus route heads: fresh 88.2, `full_v1` 88.8.
- **The token distribution now drives within ~1–2 DS of LEAD-style route control.**

`histfree_e26_route`: only 54/220 routes produced results (86.6 on those). The other 166 failed with CARLA's `LowLevelFatalError` (an Unreal Engine crash, not the model). Not rerun.

**Decision (2026-10-11): route-mode evals are no longer a priority.** The paper's model is the token distribution, and it drives on its own (waypoint mode, beam). Route mode is only a reference.

Provisional notes from the partial run:

`histfree_e26_beam`: **the token distribution drives** (waypoint mode for steer, throttle and brake; beam 8; settings verified in the evaluator processes). After **135 of 220 routes**:
- **mean DS 89.4**, RC 94.5;
- 119 completed, 6 route deviations, 4 blocked, 6 TickRuntime;
- infractions: layout 1, vehicle 11, outside lanes 4, route_dev 11, blocked 4.

For comparison, full runs in the same mode: fresh e27 53.1, `full_v1` 70.3, navcond MLP waypoints 84.9; route heads 88.2–88.9.
- If it holds over all 220 routes, history-free conditioning + LEAD recovery labels close the token-driving gap entirely.
- Open-loop error at 2 s (epoch 9: 0.88 m) overstated the closed-loop harm: the car re-plans every 0.25 s, and the recovery labels teach correction.
- **Same-route check** (the 136 routes finished; DS of earlier runs on exactly these routes vs histfree's 89.5):

| Run on the same routes | DS | Mode |
|---|---|---|
| fresh e27 | 56.3 | token rollout, beam |
| `full_v1` | 75.1 / 74.4 | token rollout, beam / expected |
| navcond | 85.3 | MLP waypoints |
| fresh e27 | 90.9 | route heads |
| `full_v1` | 89.5 | route heads |

- So the subset is slightly easier than the full 220 (+2–4 DS for every run), but the ranking holds. **Token driving: +33 over fresh, +14 over `full_v1`, +4 over navcond's MLP, and on par with the route heads.**
- Still provisional until all 220 routes finish. `histfree_e26_route` is queued next.

### D151 (ADAPT machine) closed-loop evals, queued by the user

- All three are done: `sstf_route` and `sstf_beam` (scheduled-sampling arm 2) and `ss_beam` (arm 1).

| Run | Who drives | n | DS (cur) | SR % | RC | Layout / vehicle / pedestrian collisions | Outside lanes | Route dev | Blocked |
|---|---|---|---|---|---|---|---|---|---|
| fresh e27 (reference) | route + speed heads | 220 | 88.2 | 75.5 | 96.1 | 10 / 29 / 3 | 7 | 0 | 7 |
| `sstf_route` (arm 2 e1) | route + speed heads | 220 | 87.8 | 73.6 | 95.8 | 7 / 32 / 2 | 5 | 1 | 8 |
| fresh e27 (reference) | token rollout, beam 8 | 213 | 53.1 | 17.3 | 77.4 | 54 / 100 / 4 | 69 | 34 | 42 |
| `sstf_beam` (arm 2 e1) | token rollout, beam 8 | 213 | 56.4 | 20.5 | 77.3 | 51 / 81 / 7 | 73 | 33 | 42 |
| `ss_beam` (arm 1 e2, SS only) | token rollout, beam 8 | 203 | 49.9 | 19.1 | 73.0 | 64 / 81 / 3 | 63 | 22 | 42 |

- **Route mode:** unchanged within noise (87.8 vs 88.2), as expected; the arms changed little outside the decoder.
- **Token rollout:** +3.3 DS (fewer vehicle collisions), matching Step A's 4–9% rollout gain. Still far below `full_v1` (70.3), and the 42 blocked routes are unchanged.
- **Conclusion:** a short scheduled-sampling fine-tune is not the fix, confirmed in closed loop.
- **Arm 1 (scheduled sampling only)** is worse than the fresh run (49.9 vs 53.1, 203/220 routes), consistent with its degraded conditionals.

## Results so far

### Closed loop, Bench2Drive-220

| Run (checkpoint) | Who drives | Decoding | n | DS (on evaluated routes) | SR % | Layout / vehicle collisions | Outside lanes | Route dev | Blocked |
|---|---|---|---|---|---|---|---|---|---|
| navcond (`adapt_radar_navcond_v1_retry1` e14) | route + speed heads | — | 220 | 88.9 | 70.9 | 1 / 12 | 7 | 34 | 4 |
| `full_v1` e4 | route + speed heads | — | 220 | 88.8 | 72.3 | 1 / 15 | 1 | 34 | 3 |
| fresh post-train e27 | route + speed heads | — | 220 | 88.2 | **75.5** | 10 / 29 | 7 | **0** | 7 |
| navcond | trajectory MLP | (AR rollout) | 219 | 84.9 | 62.3 | 2 / 19 | 9 | 36 | 5 |
| v2 e2 | token rollout | greedy | 217 | 62.6 | 29.5 | 74 / 102 | 61 | 38 | 22 |
| v3 e2 | token rollout | greedy | 217 | 64.4 | 31.8 | 62 / 95 | 48 | 40 | 17 |
| `full_v1` e4 | token rollout | greedy | 220 | 67.1 | 30.9 | 52 / 80 | 57 | 36 | 16 |
| `full_v1` e4 | token rollout | beam 8 | 219 | 70.3 | 37.7 | 55 / 73 | 41 | 32 | 17 |
| `full_v1` e4 | token rollout | expected | 220 | 71.3 | 40.5 | 47 / 74 | 38 | 32 | 16 |
| fresh post-train e27 | token rollout | beam 8 | 213 | 53.1 | 17.3 | 54 / 100 | 69 | 34 | **42** |

- **Waypoint-mode controller:** the old TransFuser waypoint controller. It costs only ~4 DS (navcond 88.9 → 84.9), so it is not the main reason the token rollout is weak.
- **Noise:** each scenario type has 5 routes, so per-scenario numbers are noisy. Expect about ±1–2 DS run to run.
- **Fresh's 0 route deviations:** its route/speed heads and navigation conditioning were trained from scratch on the corrected history window. Not isolated by an ablation.
- **Fresh's many blocked routes in token mode:** suspected collapse onto the stationary token when its own prefix drifts. Not verified.

### Open loop (Step A), all checkpoints (2 s position error, m)

| Checkpoint | Decoder history | Teacher forced | Greedy | Beam | Expected | Greedy ÷ TF | TF NLL per step |
|---|---|---|---|---|---|---|---|
| v2 e2 | navcond decoder (scheduled sampling) + 3 TF epochs | 0.275 | 0.506 | — | — | 1.84 | 0.95–1.18 |
| v3 e2 | + 3 TF epochs | 0.246 | 0.424 | — | — | 1.72 | 0.87–1.13 |
| `full_v1` e4 | + 5 full-model TF epochs | 0.226 | 0.389 | 0.363 | 0.340 | 1.72 | 0.84–1.06 |
| fresh e29 | from scratch, TF only | 0.211 | 0.799 | 0.722 | 0.747 | 3.79 | 0.68–0.87 |
| SS arm 1 (fresh + 3 ep SS) | | **0.355** | 0.801 | 0.808 | 0.609 | 2.26 | **1.00–1.23** |
| SS arm 2 (arm 1 + 2 ep TF) | | 0.206 | 0.729 | 0.693 | 0.680 | 3.54 | 0.68–0.85 |
| navcond MLP (reference) | | — | 0.212 (flat 0.15–0.22) | | | | |

- **Exposure bias is the main cause.** The rollout equals teacher forcing at 0.25 s, then diverges, while rollout token accuracy decays (fresh: 0.71 → 0.49).
- **`full_v1`'s better rollout is inherited.** 91% of its decoder (6 transformer layers, 6.34 M parameters) comes from navcond's decoder, after ~115 epochs of scheduled sampling. Its layers moved ~35% in later fine-tuning, while the fresh decoder is unrelated to navcond (distance ≈ √2).
- **Short scheduled sampling does not reproduce it.** Arm 1 leaves the greedy rollout unchanged and degrades the conditionals (+0.3 nats). The final teacher-forced stage restores them (arm 2), but the rollout gain is only 4–9%.
- **Decoding** (beam, expected) and **quantization** are minor contributors.

### Training notes

- **v1** went NaN at step 18,078:
  - bf16 has no GradScaler check and there was no gradient clipping;
  - previous epochs' checkpoints were deleted.

  Fixed with gradient clipping (1.0), a non-finite-step guard and `keep_all_epoch_checkpoints`.
- **Skipped steps in `full_v1`** (up to 13% of steps per epoch) came from the cuDNN attention backend in PyTorch 2.5.0 returning non-finite gradients. Fixed by disabling it. Since then: 0 skipped steps in all runs.
- **The same backend also gives NaN logits in the forward pass** for short sequences (found 2026-10-08). The disable has moved to the model constructor.
- **Speed (single H100):** about 1.0 s/step for full-model teacher forcing, 1.16–1.3 s/step under scheduled sampling, and ~0.5 s/step for token-decoder-only runs. 7,568 steps per epoch.

## Decisions and their reasons

1. **The paper is about the scoring method.** BADCAR is PDM-Lite plus random rule-based anomalies; the details are still to come.
2. **The score needs a proper density:**
   - hard cross-entropy (a proper scoring rule);
   - a normative-only vocabulary;
   - identical tokenization in vocabulary building, training and scoring.

   Not acceptable as the final stage:
   - soft CE (its entropy has a σ floor);
   - navsim's CE-off expectation training (the softmax isn't a density);
   - scheduled sampling: Huszár (2015) proves its objective improper; at full replacement it converges to per-position marginals and ignores the prefix.
3. **Condition on scene + current state, not on past motion** (2026-10-07).
   - The per-step score only uses the first token after the history, so it never needed the chain.
   - Past-motion conditioning causes both:
     - the copycat / causal-confusion failure (ChauffeurNet; Wen et al. 2020);
     - prefix contamination of anomalous drivers' scores.
   - LEAD itself feeds no past positions in CARLA mode.
4. **Discrete distributions stay:** categorical over a vocabulary, closed-form entropy, any multimodal shape.
5. **Step and window scores are both hypotheses;** keep whichever works.
6. **Waypoint mode (the token distribution drives) is the test of "the distribution itself drives competently".** Route-mode scores only show that the shared context is good.

## Calibration plan (2026-10-11)

**Goal.** Before scoring anomalies, show that on normative driving the excess surprisal s = ℓ − H is centred and correctly scaled. Every check below is computed by `scripts/score_surprisal.py` on whole drives, frame by frame (4 Hz, so one token = one frame step), under the history-free checkpoint.

**What the script computes.**
- **Step score:** the first token given scene + current state: ℓ, H, varentropy ν², s, z = s/ν.
- **Window score:** the whole 2 s token sequence. ℓ comes from teacher forcing; H and ν² from 32 ancestral samples per frame.
- **Summaries:** split by speed bin, entropy decile and scenario type. Per-frame values are saved to `outputs/surprisal/<name>.npz`.

| Check | A calibrated reference gives | A failure means |
|---|---|---|
| mean s: overall, per speed bin, per entropy decile, per scenario | ≈ 0 everywhere | > 0: the model is overconfident or wrong there, so normal driving looks unruly. < 0: it is too diffuse, which forgives anomalies |
| Var s ÷ mean ν² (and std z) | ≈ 1 | the z scale (and any per-step threshold) is off |
| slope of s on H (ℓ on H) | ≈ 0 (≈ 1) | the bias grows with uncertainty |
| drive-level Z = S/√V across drives | mean 0, std 1, \|Z\| > 2 for ≈ 5% | a drive-level flag would fire on normal drives |
| lag-1 autocorrelation of s within a drive | ≈ 0 | steps are not a martingale, so Z's variance is understated |
| coverage P(m < α), m = probability mass of codes more likely than the observed one | ≈ α | predictive sets are too narrow (< α) or too wide |
| top-1 ECE | small | top-1 confidence is miscalibrated |
| R² of H on speed and \|yaw rate\| while moving | well below 1 | the entropy is just kinematics and the scene adds nothing |
| first-step quantization residual | ≈ 0 | tokens don't describe the motion (rolling tokens should be exact) |

- **Drive Z is the strictest test.** A per-step bias b shifts Z by about b·√N/ν, so b = 0.08 nats over a 140-step drive already gives Z ≈ 1.
- **The fix for a small uniform bias** is the per-step allowance η from `unruliness.tex`, fitted on held-out normal drives. It is not a retraining.

**Stages.**

0. **Training frames** (this machine; `--num_routes 200`, about 21k frames). Sanity check only: a model that fails here is miscalibrated in-sample.
   - **Result, `histfree_e29`, 200 random training routes (21,080 frames; `outputs/surprisal/stage0_histfree_e29_train200.*`):**

     | Check | Step score | Window score (2 s) |
     |---|---|---|
     | mean s | **+0.035** (ℓ 1.258, H 1.223; perplexity 3.5) | −0.047 (ℓ 10.09, H 10.14) |
     | Var s ÷ mean ν² | 1.15 (std z 1.04) | 1.10 |
     | mean s by speed: stopped / 0.1–5 / 5–10 / > 10 m/s | 0.000 / +0.03 / +0.05 / +0.10 | +0.18 / −0.25 / −0.13 / −0.01 |
     | mean s by entropy decile | −0.01 … +0.12 (deciles 2–6) … −0.11 (top) | +0.2 (low H) … −1.05 (top) |
     | slope s on H | −0.03 | |
     | randomized-PIT coverage at α 0.5 / 0.8 / 0.9 / 0.95 | **0.505 / 0.796 / 0.897 / 0.946**, PIT deciles flat (0.095–0.103) | |
     | top-1 accuracy / ECE | 0.58 / 0.018 | |
     | R² of H on speed and \|yaw rate\| (moving) | 0.22 | |
     | first-step residual: median / > 2τ | 0.017 / 1.4% | |
     | drive Z (200 drives, median 84 steps) | **mean +0.37, std 1.44, \|Z\| > 2 for 11.5%** | |
     | autocorrelation of s at lag 1 / 2 / 4 / 8 | 0.07 / 0.03 / 0.03 / 0.04 | |

   - **Per step, the score is calibrated in-sample:** mean s ≈ 0 across speeds, entropy levels and scenario types (all within ±0.26), the PIT is uniform and ECE is small. The scene, not kinematics, drives the entropy (R² 0.22).
   - **Per drive, it is not.** Var(S)/E(V) = 2.29, because s has a small autocorrelation that persists to lag 8: some drives are consistently a little more surprising than others. A drive-level flag therefore cannot assume the martingale variance V. It needs an empirical null (a variance factor and η fitted on held-out normal drives) or a block bootstrap. Stage 1 decides whether this is a property of the score or an in-sample artefact.
   - **The window score** is centred overall but tilts with entropy (−1.05 in the top decile). Its entropy is a 32-sample estimate, so part of that tilt may be estimator noise; check with more samples before reading it.
   - The coverage check uses the randomized PIT m + U·p(u). Without the randomization it reads 0.74 at α = 0.5, an artefact of peaked discrete distributions.
1. **Unseen normative drives: the LEAD expert re-run on the 220 Bench2Drive eval routes.**
   - These routes are not in the training data. There are 63 route-id collisions (16 in the same town), but the start positions differ by kilometres.
   - **Nothing ready-made fits:**
     - The fork's `slurm/evaluate_expert.sh` (→ `evaluate_scripts_generator.py --expert`) targets Bench2Drive, but it calls `3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_local.py`, which does not exist. It also never sets `DATAGEN`, so it would only score the expert, and it needs SLURM.
     - Upstream LEAD v1.5 (`python -m lead --expert`) writes the py123d format, which this fork does not read.
   - **New `scripts/collect_expert_b2d.sh <gpu> <routes.txt>`:**
     - same settings as `scripts/eval_expert.sh`: `leaderboard_autopilot` + `scenario_runner_autopilot`, `DATAGEN=1`, LEAD sensor rig, traffic-manager seed 0;
     - one CARLA per route with retries, like `eval_b2d_one_gpu.sh`;
     - writes `data/b2d_expert/data/<Scenario>/<route>/`, a data root the training code reads.
     - **Not run yet.** Try one route first, check its time and disk use, then split the 220 across both D151 GPUs.
   - **Scenario coverage:** all 42 Bench2Drive scenario types exist in the autopilot scenario runner. Six have no training folder of that name: `Vanilla*`, `SequentialLaneChange` and `YieldToEmergencyVehicle`. Report them separately as out-of-distribution scenes.
   - **After collection:**
     1. Count the clean drives. The bucket builder keeps only drives the expert finished without infractions (minimum-speed and minor lane infractions excepted). Record the expert's DS from `data/b2d_expert/results/`.
     2. Copy the data root to this machine, or run scoring on D151:
        ```
        python scripts/score_surprisal.py --checkpoint_dir outputs/checkpoints/histfree_e29 --data_root data/b2d_expert
        ```
        The first call builds `data/b2d_expert/buckets/`.
     3. Compare with Stage 0. The difference is the generalisation part of the bias.
     4. Fit η on half of the drives and test drive Z on the other half.
2. **Optional: a second normative driver.** Think2Drive's Bench2Drive dataset is log-replayable from its annotations, rendered with LEAD's rig. It would test whether a competent but different driving style scores s ≈ 0. A replay script is not written yet.
3. **BADCAR normal drives (PDM-Lite):** the baseline that anomalies are injected into. Needed for η and the false-alarm rate. Open: whether the sensor rig and rate match LEAD's.
4. **BADCAR anomalies:** detection AUROC per anomaly type, step vs window score, time to detection.

## Next steps

1. **Calibration**, as above: read Stage 0, then collect and score Stage 1.
2. **Ablations, if useful:** history-free without rolling tokenization; rolling tokenization with history. These attribute the 87.3 DS.
3. **Paper wording:** the scored head now also drives competitively (87.3 DS in waypoint mode), so the claim can be "the distribution that drives is the distribution that scores".
4. **Fallback only if the score needs it: the whole-trajectory token.** One categorical over K = 4,096 trajectories, each 2 s.
   - **Vocabulary:** k-means over expert 2 s windows.
     - Held-out nearest-template error at 2 s (moving windows): 0.24 m. For comparison, K = 8,192 gives 0.21 m, and VADv2-style farthest-point sampling 0.43 m.
   - **Head:** logit_k = q(c_t) · E(template_k); hard CE to the nearest template, in the virtual-car frame.
   - **Conditioning:** history-free, as now.
   - **Scores:**
     - window score: exact;
     - step score: via the first-step marginal, or a small first-step head on the 1,234-code vocabulary.
   - **Precedent:** VADv2 (ICLR 2026) uses exactly this, with hard CE and argmax driving, and reaches 76.2 DS on Bench2Drive.

## Literature consulted (2026-10-07)

| Work | Relevance |
|---|---|
| Huszár 2015, *How (not) to train your generative model* | scheduled sampling is improper; at full replacement it converges to per-position marginals |
| Bansal et al. 2018, ChauffeurNet | "past motion dropout" against history copying; perturbation for recovery |
| Wen et al. 2020, *Fighting Copycat Agents* | the copycat problem with observation histories |
| Laskey et al. 2017, DART | noise injected into the expert during collection gives correct recovery labels |
| He et al., *Exposure bias vs self-recovery* | exposure bias is milder in language models; worse for smooth integrated signals |
| Zhang et al. 2025, CAT-K (CVPR oral; repo at `~/catk`) | closed-loop fine-tuning with recovery targets; ego-planning optimum K = 2–3 |
| Philion et al. 2024, Trajeglish | noisy tokenization; little gain |
| Chen et al., VADv2 | trajectory vocabulary + hard CE, 76.2 DS on Bench2Drive |
| Li et al. 2024, Hydra-MDP | trajectory vocabulary with soft L2 targets (smoothed density) |
| Chai et al. 2019, MultiPath | anchor trajectories + Gaussian mixture, closed-form likelihood |

## Code (all on `main`)

| Item | What | Where |
|---|---|---|
| Body-frame deltas | `_compute_deltas` matches the extractor; legacy codebooks keep the old behaviour | `lead/kdisks/kdisks_codebook.py` |
| History window | training history ends at the current pose (ages 20…0), equal to the default runtime window | `lead/common/history_features.py` |
| Codebook | `kdisks_carla_body.pkl`: 1,234 codes, τ 0.05, n_min 25, stationary token reserved, heading weight 2.672 (CARLA ego half-diagonal); built from valid routes only | `scripts/extract_lead_carla_deltas.py`, `scripts/build_kdisks_carla.py` |
| Waypoints from tokens | integrate the decoded tokens from the start pose; the trajectory MLP is unused | `adapt_waypoints_from_tokens` |
| Decoding | `adapt_decoding = greedy / beam / expected`, `adapt_beam_width` (inference only) | `lead/adapt/adapt_decoder.py` |
| History-free decoder | `adapt_history_free`: start token, no history poses, origin start | same |
| Sequence tokenization | `adapt_rolling_tokenization`, `adapt_rolling_beam_width` | same |
| Token-decoder-only training | `adapt_train_token_decoder_only` (everything else frozen and in eval mode); combinable with scheduled sampling | `lead/training/training_utils.py`, `train.py` |
| Stability | `grad_clip_norm`, non-finite-step guard, `keep_all_epoch_checkpoints`, cuDNN attention disabled in the model constructor | `train.py`, `lead/adapt/adapt.py` |
| Open-loop analysis (Step A) | per-horizon error for the GT-token floor, teacher forcing, each decoding and a reference checkpoint; token accuracy and NLL | `scripts/eval_open_loop_rollout.py` |
| Launch script | `scripts/posttrain_token_decoder.sh` (+ `RUN_NAME`, `EXTRA_CONFIG`) | |
| Surprisal calibration | step and window excess surprisal over whole drives, with the calibration checks of the plan; `--data_root` scores any CARLA data root | `scripts/score_surprisal.py` |
| Expert collection on benchmark routes | LEAD expert with `DATAGEN=1`, one CARLA per route with retries, local (no SLURM) | `scripts/collect_expert_b2d.sh` |

**Uncommitted (2026-10-08):**
- the cuDNN disable has moved to `lead/adapt/adapt.py`;
- the rolling tokenizer returns contiguous tokens.

## Operational notes

- **Config inheritance.** A run started from a checkpoint inherits that checkpoint's `config.json`; only `LEAD_TRAINING_CONFIG` overrides it. The launch script sets the codebook path, heading weight and vocab size explicitly. Duplicate keys: the last one wins.
- **GPU.** Pin the full H100 by UUID: `CUDA_VISIBLE_DEVICES=GPU-4ec20303-3c97-fc1c-9499-e5539efe3717`. Index 1 is a 10 GiB MIG slice. Do not use the slices.
- **One model per checkpoint folder.** The agent ensembles every `model*.pth` in a folder, so evaluate through one-model folders (symlinks). Results are keyed by folder name, and routes with results are skipped on rerun.
- **D151:**
  - `export CARLA_ROOT=/home/adapt/adapt/ADAPTDataset/carla` (the default points at a missing `carla_0.9.16`);
  - Vulkan adapters `0 2`;
  - waypoint mode via `LEAD_CLOSED_LOOP_CONFIG="steer_modality=waypoint throttle_modality=waypoint brake_modality=waypoint"`;
  - decoding via `LEAD_TRAINING_CONFIG="adapt_decoding=beam adapt_beam_width=8"`.
- **Hugging Face** (`Divyanshudiv/ADAPT`, model repo): navcond e14, v2 e2, v3 e2, `full_v1` e4, fresh e27 and e29, `adapt_ss_full_v1` e2, `adapt_ss_full_tf_v1` e1.
- **Running Step A during training** shares the GPU: it slows training ~2× while it runs and needs ~7 GB (training uses up to 87 GB). Run one at a time.
- **Commits:** no Claude author or co-author attribution lines.

## Known limitations and open questions

- **No held-out split:** every run trains on all post-train towns, Town13 included. Fair NLL comparisons need data no model saw, such as BADCAR normal drives.
- **First-order tokens:** a token is one 0.25 s step (speed, yaw rate). Hard braking from a normal speed stays in-vocabulary, so the quantization residual can't see it; only transition surprisal can.
- **Driver-induced out-of-distribution states remain through the current state:**
  - Sustained speeding pushes speed outside the training data, and LEAD's perturbation only covers lateral and yaw offsets.
  - An out-of-distribution state can raise H and so forgive an anomaly.
- **Samples without a full future:** they are zero-filled by the collate function. This does not occur in post-train buckets (8 frames skipped at each end).
- **Open:** BADCAR details (anomaly types, λ, ramped vs abrupt onset, sensor suite and rate); how close PDM-Lite contexts are to LEAD's training distribution.
