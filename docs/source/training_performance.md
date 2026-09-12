# Training performance options

The first infrastructure update keeps the existing CARLA data and checkpoint
format. It adds working compilation, bounded metadata caching, optional cheaper
perspective heads, optional CUDA batch prefetch and optional device-side image
augmentation. It does not migrate the dataset to py123d or replace the trainer.

## Defaults and compatibility

| Setting | Default | Effect |
| --- | --- | --- |
| `compile` | `true` | Compile the module actually used by training; `false` now works as an override |
| `compile_scope` | `backbone` | Compile the sensor backbone; `model` selects the complete model/DDP wrapper |
| `compile_backend` | `inductor` | PyTorch compiler backend |
| `compile_mode` | `default` | `max-autotune` spends more startup time selecting kernels |
| `compile_fullgraph` | `false` | Permit graph breaks around unsupported Python code |
| `compile_dynamic` | `false` | Specialize to the configured tensor shapes |
| `metadata_cache_max_entries` | `128` | Bound metadata RAM use per worker; zero disables the RAM tier |
| `upsample_perspective_logits` | `false` | Move the last semantic/depth convolution block before final upsampling |
| `upsample_mode` | `bilinear` | `nearest` selects cheaper interpolation in perspective heads |
| `cuda_prefetch` | `false` | Upload the next batch on a separate CUDA stream |
| `gpu_color_augmentation` | `false` | Use the alternative upstream batched image augmentation recipe |

Compilation now runs rather than discarding a `torch.compile` return value. It
uses `Module.compile` in place, so model parameter names, optimizer parameters,
DDP access and checkpoint keys remain stable. Start with backbone compilation;
full-model compilation depends on the selected heads and Python control flow.
Compilation startup belongs in a separate warmup measurement, not steady-state
step throughput. Use `compile=false` for an eager comparison or short debug run.
Backbone runtime type checks remain active in eager execution and are bypassed
during compiler capture; tensor operations and compiler guards still apply.

The default perspective heads reproduce the previous computation exactly. All
four upsampling combinations retain the same parameter shapes and names, and
the radar_ext2 semantic/depth checkpoint weights load strictly. Enabling the
cheaper computation changes auxiliary predictions and gradients, so measure
driving quality after training with it.

The metadata cache reads its session entries, populates a bounded worker LRU,
invalidates entries when the source file changes and returns independent mutable
records. `force_rebuild_data_cache=true` bypasses both metadata cache tiers.
Existing sensor caches remain readable. The new metadata namespace ignores old
path-only entries. Initial cache population costs extra work; repeated access
benefits from skipping LZMA decompression. The entry limit bounds record count,
not an exact number of bytes.

GPU augmentation is CARLA-only and disables the CPU image augmentation in the
CARLA loader. It preserves image shape/dtype but is **a different stochastic
recipe**: fixed operation ordering, 5% dropout and different parameter sampling.
Leave it disabled for experiments intended to preserve the existing imgaug
distribution. It runs before normalization and outside the compiled model.
`use_color_aug=false` now explicitly disables either augmentation path.

## Example training overrides

Use a fresh output directory when comparing a new computation to an existing run:

```bash
LEAD_TRAINING_CONFIG="model_type=adapt load_file=outputs/local_training/adapt_posttrain_radar_ext2/model_0039.pth continue_failed_training=false logdir=outputs/local_training/adapt_radar_speed compile=true compile_scope=backbone upsample_perspective_logits=true upsample_mode=nearest cuda_prefetch=true prefetch_factor=2" \
torchrun --standalone --nproc_per_node=2 -m lead.training.train
```

This initializes from the checkpoint; it is not an exact optimizer-state resume.
For an actual resume, use the existing `continue_failed_training=true` workflow
and keep the training computation unchanged. No existing checkpoint is converted.

Lowering `prefetch_factor` from 16 to 2 reduces queued host batches. Whether it
improves speed depends on the data-loader bottleneck; it is not a universal
speedup. Pinned memory, persistent workers, mixed precision and channels-last
were already present. Data-loader construction now also supports zero workers
for profiling and debugging.

## Short benchmarks

From the repository root, select one GPU and benchmark actual samples with a
checkpoint. The command performs forward/loss/backward/AdamW steps in memory;
it never writes trained weights or starts W&B. Ensure `LEAD_TRAINING_CONFIG` is
unset so overrides cannot silently change the recorded experiment.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_training \
  --config outputs/local_training/adapt_posttrain_radar_ext2/config.json \
  --checkpoint outputs/local_training/adapt_posttrain_radar_ext2/model_0039.pth \
  --batch-size 32 --workers 4 --warmup 3 --steps 20 \
  --output outputs/performance/eager.json
```

Run a second process with `--compile --logits-upsample --nearest --prefetch`
and a different output path. Use `--gpu-augmentation` separately when comparing
the alternative augmentation recipe. The default probe disables image and
sensor-view augmentation and uses the selected epoch's loss/sampling schedule.
It initializes a fresh optimizer rather than resuming optimizer state.
Fresh processes isolate allocator and in-process cache state, but persistent
compiler caches can still shorten subsequent warmups.

Reports include setup/warmup time, data waiting, forward, losses, backward,
optimizer, sample throughput and peak allocated GPU memory during measured
steps. Per-step synchronization makes these diagnostic timings, not a claim of
maximum asynchronous training throughput. Loader/enqueue timing is CPU time;
CUDA events measure the device timeline, including launch gaps. Augmentation
and prefetched transfers contribute to wall time but are outside the forward
event. Benchmark the production batch size and enough steps before estimating
epoch duration.

### Measurements on this server

Measured with `radar_ext2/model_0039.pth`, PyTorch 2.5.0, BF16, one H100 NVL,
batch size 32, four workers, three warmup steps and ten measured steps. The
checkpoint directory is `outputs/local_training/adapt_posttrain_radar_ext2`.
Both image and sensor-view augmentation were disabled for these comparisons.
All variants use the new metadata cache, so the table does not measure that
cache's benefit. The complete model, losses, backward and optimizer run in each
probe; logging, scheduler updates and distributed communication are excluded.

| Variant | Mean step | Samples/sec | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| Eager, original heads | 432.7 ms | 73.9 | 18.47 GiB |
| Compiled backbone, original heads | 374.2 ms | 85.5 | 15.23 GiB |
| Eager, logits-first heads with nearest interpolation | 356.3 ms | 89.8 | 15.51 GiB |
| Compiled backbone + cheaper heads + CUDA prefetch | 329.1 ms | 97.2 | 12.56 GiB |

The combined probe used 24% less step time and 32% less peak allocated memory
than the eager baseline. Its three warmup steps took 113.8 seconds, versus 25.6
seconds for the baseline; these are observed warmup costs, not guaranteed cold
compiler timings. CUDA prefetch was not isolated in this comparison, so the
table does not establish its individual benefit. At batch size four, the eager
heads-plus-prefetch probe was slower than baseline (293 vs 275 ms). Measure at
the intended batch size before choosing settings. These short probes do not
establish multi-GPU epoch speed or driving-score changes.

Raw step timings and a baseline operator trace are in
`/localstorage/home/f20221129/analysis/adapt_performance/`. The separate metadata
probe is `../analysis/metadata_cache_benchmark.json` relative to the repository:
128 real records took 86.6 ms through the previous read/write path, 12.2 ms with
a warm worker cache, and 19.0 ms with a warm session cache. First-time population
took 158.6 ms; these are metadata-only timings, not whole-loader throughput.

`--trace` runs an additional diagnostic step and writes a Chrome/Perfetto trace
plus an operator table. It labels backbone, radar, ADAPT and auxiliary forward
components. This step is excluded from timed averages and memory reporting;
profiling hooks can cause extra compilation when compilation is enabled.

For the isolated perspective heads:

```bash
python scripts/benchmark_perspective_decoder.py --help
```

Regression checks:

```bash
python -m pytest -q tests
```

The suite covers metadata invalidation/mutation isolation, default perspective
output and gradient parity, strict checkpoint shapes, actual compiler use,
two-process DDP gradient synchronization, CUDA stream/buffer lifetimes and
augmentation behavior. Visualization checks compare CPU/CUDA radar and waypoint
rendering, since prefetched batches now reach those paths on the GPU.
CUDA-specific checks skip on machines without CUDA.

The actual `Trainer.train()` also completed three steps with compilation,
prefetch, the cheaper heads and GPU augmentation enabled, using the trainer's
warnings-as-errors policy. Losses and parameters stayed finite and backbone
weights updated. This smoke check replaced external logging and config writes
with local stubs and did not save checkpoints; its report is
`/localstorage/home/f20221129/analysis/adapt_performance/trainer_smoke.json`.
