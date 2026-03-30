# AReaL-Style Async Runner (Lightweight)

This adds a new multiprocessing async path that is closer to an AReaL-style systems topology while staying PSC-friendly.

It does **not** replace existing minimal sync/async runners.

## New Components

- `src/run_async_areal_style.py`: CLI entrypoint
- `src/areal_runtime.py`: process orchestration
- `src/areal_parameter_service.py`: serialized update commits + global policy version
- `src/areal_rollout_worker.py`: rollout workers with interrupt polling
- `src/areal_trainer_worker.py`: trainer workers consuming replay batches
- `src/areal_controller.py`: dispatch, reward evaluation, staleness filtering, replay batching

## Architecture

- **Parameter Service**
  - owns global trainable state and `policy_version`
  - applies trainer updates serially
  - increments `policy_version` only after a real committed update
- **Rollout Controller**
  - dispatches prompts/tasks to rollout workers
  - evaluates reward using existing shared reward harness
  - computes `staleness = current_global_policy_version - sample.policy_version`
  - drops stale samples when `staleness > K`
  - stores accepted samples in replay buffer and dispatches training batches
- **Replay Buffer**
  - in-controller accepted sample list/batches
- **Rollout Workers**
  - local backend replicas on assigned devices
  - poll for version changes and interrupt at safe chunk boundaries
  - reload latest trainable state when interrupted
- **Trainer Workers**
  - local backend replicas on assigned devices
  - consume training batches
  - send updates to parameter service for serialized commit

## Interruptible Rollout Behavior

Interrupts are real and version-driven:

1. Trainer commit updates state in parameter service and increments global `policy_version`.
2. Rollout workers poll `policy_version` during chunked generation.
3. If newer version is observed, worker interrupts current sample at safe boundary, marks it interrupted, reloads latest state, and moves on.

Safe boundary in this implementation:

- `tiny_policy`: chunk boundary before final emit (fully interruptible for this toy backend)
- `hf_trainable`: chunk-level fallback via backend helper; true token-level interruption is not implemented in this step

## Device Split and Worker Assignment

- `--rollout-devices` and `--trainer-devices` accept comma-separated lists.
- If list is shorter than worker count, assignment is round-robin.
- If CUDA is unavailable, runtime falls back to CPU for that worker.

PSC intended default:

- 1 rollout worker on `cuda:0`
- 1 trainer worker on `cuda:1`

## Scripts

- CPU smoke test:
  - `bash scripts/run_async_areal_style.sh`
- PSC 2-GPU topology:
  - `CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_async_areal_style_psc_2gpu.sh`
- Interruptibility stress test:
  - `bash scripts/run_async_areal_style_stress.sh`

## Output Files

Default:

- `results/async_areal_results.jsonl`
- `results/async_areal_summary.json`

Stress:

- `results/async_areal_stress_results.jsonl`
- `results/async_areal_stress_summary.json`

Summary includes:

- worker counts/devices
- update count / final policy version
- accepted/dropped fractions
- staleness stats
- reward/pass/wall-clock/tokens-per-sec
- queue depth and replay buffer traces
- per-rollout interrupt/generated/version stats
- per-trainer update stats
- interrupted sample count
- config block

## Limitations vs Full AReaL / Full PPO

- No full PPO objective, value head, GAE, or distributed optimizer.
- No heavyweight distributed infra (Ray/DeepSpeed/etc).
- Parameter service commits entire trainable state per update in this lightweight prototype.
- `hf_trainable` interruption is chunk-level fallback, not token-level preemption.
