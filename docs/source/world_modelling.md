# Vehicle future prediction for ADAPT

`adapt_world_model=true` adds an object-centric future predictor to ADAPT while
retaining navigation conditioning and the existing ego prediction heads. The
initial implementation forecasts radar-observed vehicles; it is not yet an
interactive simulator of how traffic reacts to candidate ego actions.

```text
RGB + LiDAR history + radar + ego status
                 │
          shared perception
                 │
          radar actor queries ── current actor state/validity
                 │
        multimodal future predictor
                 │
       predicted positions + interval velocities
          + mode/vehicle/detection confidence
                 │
         one token per actor/mode
                 │
       residual future cross-attention
                 │
      existing scene/history context
             ┌───┴─────────────┐
   navigation-conditioned     AR ego trajectory
      route/speed planner       prediction
```

## Predictions and planning

The default predictor emits three futures for each of 20 radar queries, with
8 points at 0.25-second intervals (0.25–2 seconds). Positions and interval
velocities use the fixed current-ego frame. Positions are anchored at each
predicted current actor position. Vector speeds are bounded by a separate
`world_max_speed=40` m/s, independent of the ego/detector speed normalization.
The model currently predicts position and interval velocity, not heading or
future actor existence.

A trajectory MLP encodes each complete hypothesis into one token: 60 tokens
rather than 480 timestep tokens. Its inputs are predicted states and
probabilities; raw actor features cannot bypass forecasting into this memory.
The hypotheses remain separate instead of averaging incompatible paths.

`FutureContextFusion` uses the product of mode probability, predicted vehicle
probability and detector validity as attention confidence. A null token keeps
empty scenes well-defined. The attention output projection starts at zero, so
adding this branch preserves the loaded ego model's initial outputs. Its
context feeds both the route/speed queries and AR trajectory branch; the two
heads retain their existing architectures.

Predicted actor futures are available in `Prediction.pred_actor_futures`,
including `future_positions [B,Q,M,T,2]`, `future_velocities`, `mode_logits`,
`vehicle_logits`, `memory`, and `memory_confidence`. Inference does not need any
new sensor, privileged actor state, track ID or future-label input.

## Labels and losses

The detector's existing current-state Hungarian assignment is reused for future
supervision. Future endpoints never determine actor identity. The data loader
uses one shared radar selection function, preserving current detection labels
and aligning future labels with precisely the same selected boxes.

The recorder stores current pose plus up to 40 future simulator ticks, matched
by actor ID. Missing observations are omitted without timestamps. Consequently
only complete 41-pose recordings can establish the timestamps of the cached
8-point horizon. All incomplete/extrapolated futures are masked out. They are
not treated as vehicle disappearance. Current vehicle classification remains
supervised even when future annotations are unavailable. An 81-frame audit
found complete tracks for 403/448 radar-visible cars; this is a small coverage
sample, not a dataset-wide estimate.

Targets are derived from existing cached boxes, augmented futures and validity
counts, without new raw-file reads or a cache rebuild:

- `world_future_positions [B,Q,8,2]`: metric positions;
- `world_future_mask [B,Q,8]`: annotation availability;
- `world_vehicle_mask [B,Q]`: VEHICLE/SPECIAL/PARKING class membership.

The cached target path requires the complete recorded horizon; changing its
length or spacing requires a new timestamp-aware label representation.

The losses added to the existing loss normalization are:

| Loss | Default relative weight | Meaning |
|---|---:|---|
| `world_loss_position` | 1.0 | Masked L1 position loss for the mode with lowest ADE, normalized by `world_position_scale=10` m |
| `world_loss_mode` | 0.1 | Classification of that best mode |
| `world_loss_vehicle` | 0.1 | Current vehicle classification, including negative nonvehicle/padded slots |

Planner losses can also backpropagate through predicted future states once the
residual connection opens. Ground-truth actor futures are used only in loss
computation, in both training and evaluation code.

Track `metric/world_min_ade`, `world_mode_ade` (highest-probability mode),
`world_min_fde`, and `world_supervised_actors`. Low best-mode error alone does
not establish calibrated probabilities or improved ego driving. Closed-loop
route completion, infractions and turn performance remain the deciding metrics.

## Fresh post-training from perception pretraining

The selected starting checkpoint is
`outputs/ADAPT/B300/checkpoints/adapt/pretrain/model_0030.pth`. It contains the
backbone, radar and auxiliary perception heads, with no ego planner. All 736
stored tensors load exactly into the combined model, with no shape mismatches
or unexpected keys. Missing keys are exclusively `adapt_decoder.*` and
`world_model.*`. Navigation conditioning is retained as an architecture, with
fresh weights in this run. The previously fine-tuned checkpoint is unchanged.

From the repository root in the `lead` conda environment:

```bash
# Inspect settings without starting a run or allocating the model.
bash scripts/posttrain_world_model.sh --check-config

# For a direct run on this server, select its full H100.
# Inside a scheduler allocation, retain the allocated CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=GPU-4ec20303-3c97-fc1c-9499-e5539efe3717
bash scripts/posttrain_world_model.sh
```

The preset follows the historical fresh radar post-training budget: 60 epochs,
learning rate `1e-4`, batch 128, and all model weights trainable. It uses a fresh
single cosine decay, backbone compilation, CUDA prefetch, 32 workers and worker
prefetch factor 4. The sensor cache uses the existing scratch location. The
alternative GPU augmentation and auxiliary-head computation remain disabled.

Unlike the `radar_ext2` fine-tune, scheduled sampling starts at **0**: three
epochs of teacher forcing, then a 30-epoch ramp to 1. The new ego planner has not
been trained yet, so inheriting the fine-tune's floor of 1 would be inappropriate.
No optimizer state is resumed. The new architecture/config must accompany its
checkpoint during subsequent evaluation.

Override run settings explicitly, for example:

```bash
WORLD_LOGDIR=outputs/local_training/adapt_world_v1_40ep \
WORLD_EPOCHS=40 WORLD_WORKERS=32 WORLD_PREFETCH_FACTOR=4 \
bash scripts/posttrain_world_model.sh
```

Other supported settings are `WORLD_LR`, `WORLD_BATCH_SIZE`, `WORLD_CHECKPOINT`,
`LEAD_PYTHON` and `SCRATCH`. The script expects exactly one visible allocated GPU
and a fresh output directory; it does not launch a simulator. Normal interrupted
run resumption should use that run's saved config/checkpoint and the existing
trainer resume workflow, rather than this fresh-training launcher. The existing
trainer retains only the latest checkpoint.

## Validation and limits

Focused checks cover label timing/augmentation, matching identity, empty/masked
scenes, gradient flow, CPU/CUDA BF16, strict checkpoint compatibility, preserved
zero-initialized outputs, and forward independence from future labels. A real
CARLA batch also passed two complete forward/loss/backward/optimizer steps with
BF16 and CUDA prefetch, including finite gradients and changed predictor weights.
That smoke check used no external logging or checkpoint writes.

This version has no persistent actor tracking state, predicts only the selected
radar-observed vehicles, and does not forecast unseen/new actors or pedestrians.
It uses temporal evidence already present in stacked LiDAR and scene features.
Modes are per-actor hypotheses, not a jointly consistent future scene. The next
extension toward interactive world modelling would condition actor responses on
candidate ego actions; it is not part of this implementation. No driving-score
improvement is claimed before training and evaluation.
