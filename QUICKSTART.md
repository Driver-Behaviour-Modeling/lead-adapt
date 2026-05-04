### 2.3. Training

Before training, build the data cache. This preprocesses the raw routes into an optimized format for the data loader, significantly speeding up training:

```bash
python scripts/build_cache.py
```

**Perception pretraining.** Logs and checkpoints are saved to `outputs/local_training/pretrain`:

```bash
# Single GPU
python3 lead/training/train.py \
  logdir=outputs/local_training/pretrain

# Distributed Data Parallel
bash scripts/pretrain_ddp.sh
```

**Planning post-training.** Logs and checkpoints are saved to `outputs/local_training/posttrain`:

```bash
# Single GPU
python3 lead/training/train.py \
  logdir=outputs/local_training/posttrain \
  load_file=outputs/local_training/pretrain/model_0009.pth \
  use_planning_decoder=true

# Distributed Data Parallel
bash scripts/posttrain_ddp.sh
```

> [!TIP]
> 1. For distributed training on SLURM, see the [SLURM training docs](https://ln2697.github.io/lead/docs/slurm_training.html).
> 2. For a complete workflow (pretrain → posttrain → eval), see this [example](slurm/experiments/001_example).
> 3. For detailed documentation, see the [training guide](https://ln2697.github.io/lead/docs/carla_training.html).

### 2.4. Benchmarking

With CARLA running, evaluate on any benchmark via **Python**:

```bash
python -m lead \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes <ROUTE_FILE> \
  [--bench2drive]
```

<div align="center">

| Benchmark   | Route file                                               | Extra flag      |
| :---------- | :------------------------------------------------------- | :-------------- |
| Bench2Drive | `data/benchmark_routes/bench2drive/23687.xml`            | `--bench2drive` |
| Longest6 v2 | `data/benchmark_routes/longest6/00.xml`                  | -               |
| Town13      | `data/benchmark_routes/Town13/0.xml`                     | -               |
| Fail2Drive  | `data/benchmark_routes/fail2drive/Base_Animals_0075.xml` | `--fail2drive`  |

</div>

Or via **bash**:

```bash
bash scripts/eval_bench2drive.sh   # Bench2Drive
bash scripts/eval_longest6.sh      # Longest6 v2
bash scripts/eval_town13.sh        # Town13
bash scripts/eval_fail2drive.sh    # Fail2Drive (requires CARLA_F2D)
```