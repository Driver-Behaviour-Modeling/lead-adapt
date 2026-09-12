# Navigation-conditioned ADAPT planning

Set `adapt_navigation_conditioning=true` for a new CARLA ADAPT training run.
The model jointly encodes ordered previous/current/next target points and the
current/next commands. This representation initializes the route/speed planning
queries and modulates them before every planning decoder layer.

The implementation uses a small MLP rather than additional sensor-attention
layers. Its input includes normalized target coordinates, segment offsets,
physical segment directions and lengths, ego-to-target distances, and both
one-hot command vectors. Segment directions are computed in metres before
anisotropic normalization. Coincident targets use zero directions and explicit
validity bits, covering terminal routes and distant-target duplication.

For each planning layer the residual conditioning is:

```text
navigation = NavigationEncoder(observed_targets, current_command, next_command)
queries = learned_queries + query_projection(navigation)
for each planning layer:
    queries = queries + scale(navigation) * LayerNorm(queries) + shift(navigation)
    queries = planning_layer(queries, sensor_and_history_context)
```

Each layer learns its own scale/shift projection. The current radar_ext2 shape
(256-dimensional embeddings, six planning layers, six command classes) adds
930,304 trainable parameters. The original sensor memory, planning-layer weights,
route/speed output interfaces, and AR trajectory branch retain their structure.
This implements dedicated navigation conditioning; it does not implement shared
plan tokens, path-conditioned speed, or world modelling.

The query projection and per-layer scale/shift projections start at zero. This
preserves the loaded planner's initial function while new connections learn from
the existing route/speed losses. On the first update, these projections receive
gradients; the joint encoder starts receiving nonzero gradients once they open.
Turning the option on for an old checkpoint alone is not a trained improvement.

Use the existing training entry point and a fresh output directory. For example,
from the repository root in the `lead` environment:

```bash
LEAD_TRAINING_CONFIG="model_type=adapt use_adapt_decoder=true load_file=outputs/local_training/adapt_posttrain_radar_ext2/model_0039.pth continue_failed_training=false adapt_navigation_conditioning=true logdir=outputs/local_training/adapt_radar_navcond_v1" \
torchrun --standalone --nproc_per_node=1 -m lead.training.train
```

The loader inherits the saved radar_ext2 configuration and applies these
overrides. Select the allocated CUDA device through your usual environment and
choose the fine-tuning learning rate/epoch budget before launching; the example
otherwise inherits those values from the saved configuration. Use a new logdir
for each experiment. `continue_failed_training=false` warm-starts model weights
with fresh optimizer/scheduler state. The only newly missing model keys should
begin with `adapt_decoder._navigation_`; other missing/unexpected or
shape-mismatched keys need investigation.

Old configurations default to `adapt_navigation_conditioning=false`, instantiate
no new weights, and retain strict checkpoint loading. New training configurations
save the enabled flag; use their accompanying config at inference and normal
strict loading. This is an architecture option in `LEAD_TRAINING_CONFIG`, not a
runtime switch in `LEAD_CLOSED_LOOP_CONFIG`.

Both training and online inference already supply the required navigation
fields. Missing fields fail explicitly; the option requires previous/current/next
targets and discrete commands and is currently intended for CARLA. No future
ground-truth trajectory enters the new encoder. Sensor-cache preprocessing and
navigation frame/pop-distance conventions are unchanged.

The tests cover physical geometry, duplicate targets, checkpoint round trips,
identity warm starts in training/inference, gradient flow under the actual
route/speed losses, next-command sensitivity after connections open, absence of
future-label dependence, and compilation of the navigation encoder.
