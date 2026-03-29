# Async Training Next Step (Minimal Real Learning)

This update moves the project from rollout-only validation to a **real learning signal** while staying lightweight for PSC.

## What Changed vs Rollout-Only Baseline

- `src/model_backends.py` now includes trainable backends:
  - `tiny_policy` (portable pure-Python REINFORCE baseline)
  - `hf_trainable` (tiny Hugging Face causal LM backend with token logprobs + optimizer updates)
- `src/run_sync_baseline.py` now performs actual policy updates (no `NoOpTrainer`)
- `src/run_async_baseline.py` now has a real learner loop consuming a train queue and updating the model
- `policy_version` increases **only when a real optimizer update happens**
- Added per-step training metrics and update statistics in summaries

## Sync Learning Flow

1. Build prompt from the shared benchmark/task loader.
2. Generate response from current policy.
3. Evaluate reward with shared coding reward harness.
4. Accumulate samples into update batches.
5. Run a minimal policy-gradient update and increment `policy_version` only on successful update.

## Async Learning Flow

1. Producer generates responses and tags each sample with current `policy_version`.
2. Consumer evaluates reward and applies bounded staleness filter (`staleness > K` drops sample).
3. Accepted samples are pushed to learner queue.
4. Learner updates policy in batches and increments `policy_version` only on real updates.

This preserves the systems question (sync vs async under staleness constraints) while introducing actual learning dynamics.

## Backend Notes

- `tiny_policy`:
  - always available (no torch/transformers needed)
  - real REINFORCE updates over a tiny categorical policy
  - recommended for guaranteed end-to-end pipeline runs
- `hf_trainable`:
  - requires `torch` + `transformers`
  - supports generation, generated-token logprobs, and policy-gradient optimizer steps
  - use tiny models such as `sshleifer/tiny-gpt2`

## Limitations vs Full PPO

- Uses minimal policy-gradient updates, not full PPO clipping/value/KL objectives
- No value head or GAE yet
- No distributed optimizer/runtime stack
- Async learner is single-process/threaded for simplicity

The code is intentionally structured so PPO components can be swapped in later where updates happen.

## Reproducibility Controls

Both sync and async support:

- `--seed`
- deterministic dataset order (no task shuffle by default)
- `--lr`
- `--update-batch-size`
- `--epochs`
- `--staleness-k` (async)
- `--max-new-tokens`

## PSC Commands

Run minimal real-learning sync:

```bash
bash scripts/run_sync_train.sh
```

Run minimal real-learning async:

```bash
bash scripts/run_async_train.sh
```

Outputs:

- `results/sync_train_results.jsonl`
- `results/sync_train_summary.json`
- `results/async_train_results.jsonl`
- `results/async_train_summary.json`

Optional tiny HF run (if `torch` + `transformers` are installed):

```bash
python3 -m src.run_sync_baseline \
  --backend hf_trainable \
  --hf-model-id sshleifer/tiny-gpt2 \
  --lr 1e-5 \
  --epochs 1
```
