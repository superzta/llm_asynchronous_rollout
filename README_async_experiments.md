# Async Experiment Suite (Controlled Grid + Analysis)

This layer builds on the existing minimal learning sync/async baselines and adds:

- repeatable experiment sweeps
- structured run directories
- merged run tables
- aggregation + plotting scripts

The goal is systematic comparison, not algorithm redesign.

## What This Adds

- `scripts/run_experiments.sh`
- `src/run_experiment_grid.py`
- `scripts/analyze_experiments.sh`
- `src/analyze_experiments.py`

plus stronger per-run summary metrics from the sync/async runners.

## One-Command Sweep

Run a small default sweep (PSC-friendly):

```bash
bash scripts/run_experiments.sh
```

Default grid:

- mode: `sync_train`, `async_train`, `async_areal_style`
- staleness_k (async): `0,1,2,4`
- seeds: `0,1,2`
- update_batch_size: `4`
- queue_maxsize (async): `64`
- epochs: `2`
- backend: `tiny_policy`
- lr: `0.1`
- AReaL worker topology defaults:
  - `num_rollout_workers=1`
  - `num_trainer_workers=1`
  - `rollout_devices=cpu`
  - `trainer_devices=cpu`

Override any dimension with CLI args, for example:

```bash
bash scripts/run_experiments.sh \
  --seeds 10,11,12 \
  --lr-values 0.05,0.1 \
  --queue-maxsizes 32,64 \
  --modes sync_train,async_train,async_areal_style \
  --num-rollout-workers-values 1,2 \
  --num-trainer-workers-values 1 \
  --rollout-devices cpu,cpu \
  --trainer-devices cpu \
  --producer-delay-sec 0.01 \
  --learner-delay-sec 0.02
```

## Result Layout

Each run goes to:

- `results/experiments/<timestamp>/<run_name>/summary.json`
- `results/experiments/<timestamp>/<run_name>/results.jsonl`
- `results/experiments/<timestamp>/<run_name>/config.json`
- `results/experiments/<timestamp>/<run_name>/command.txt`
- `results/experiments/<timestamp>/<run_name>/stdout.log`
- `results/experiments/<timestamp>/<run_name>/stderr.log`

Merged tables are also written:

- `results/experiments/<timestamp>/merged_summary.json`
- `results/experiments/<timestamp>/merged_summary.csv`

## Analyze + Plot

```bash
bash scripts/analyze_experiments.sh results/experiments/<timestamp>
```

This writes:

- `analysis_per_run.json`
- `analysis_per_run.csv`
- `analysis_aggregated.json`
- `analysis_aggregated.csv`
- plots under `plots/`

Aggregates are keyed by:

- `mode`
- `staleness_k`
- `seed`

with mean/std for:

- `avg_reward`
- `pass_rate`
- `wall_clock_sec`
- `update_count`
- `dropped_stale_count`
- `mean_staleness`
- `max_staleness`
- `avg_tokens_per_sec`
- plus extended run metrics (`accepted_fraction`, `dropped_fraction`, etc.)

## Generated Plots

Under `results/experiments/<timestamp>/plots/`:

- reward vs staleness bound
- pass rate vs staleness bound
- dropped stale count vs staleness bound
- wall-clock vs mode
- updates completed vs mode
- queue depth traces for representative async runs

Matplotlib is the only plotting dependency.

## Interpreting Sync vs Async vs K

- Lower `staleness_k` usually reduces stale updates but can increase dropped samples.
- Higher `staleness_k` can improve acceptance but may increase policy lag effects.
- Compare all three modes:
  - `sync_train` (synchronous baseline)
  - `async_train` (threaded async baseline)
  - `async_areal_style` (multiprocess AReaL-style topology)
- Compare:
  - reward/pass metrics (quality)
  - dropped/accepted fractions (data efficiency)
  - wall-clock + updates/sec (systems efficiency)

## Current Limitations (vs Full PPO/Slime)

- Still minimal policy-gradient style updates (not full PPO objectives/value modeling).
- Single-process/threaded control loop; no distributed optimizer stack.
- Tiny benchmark and lightweight model choices prioritize robust experimentation over scale.
- Slime integration remains scaffold-level; this suite is the controlled standalone path for now.
