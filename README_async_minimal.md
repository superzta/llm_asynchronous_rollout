# Minimal Sync vs Async Coding Experiment (PSC)

This repository contains a tiny, robust experiment for comparing:

- synchronous rollout + reward evaluation
- asynchronous producer/consumer rollout + reward evaluation with bounded staleness

The implementation is intentionally lightweight for PSC V100 runs and avoids heavy H100-oriented dependencies (FlashAttention/Apex/Transformer Engine requirements are not introduced here).

## File Structure

- `data/tiny_coding_train.jsonl`: tiny train benchmark (12 deterministic coding tasks)
- `data/tiny_coding_eval.jsonl`: tiny eval benchmark (6 deterministic coding tasks)
- `src/coding_task.py`: task schema, JSONL loader, prompt builder
- `src/coding_reward.py`: code extraction + sandboxed subprocess test runner + CLI
- `src/model_backends.py`: swappable model interface (`dummy`, `hf` stub)
- `src/staleness.py`: staleness computation and bounded acceptance helper
- `src/metrics.py`: run summaries + JSON/JSONL writing helpers
- `src/run_sync_baseline.py`: minimal synchronous baseline
- `src/run_async_baseline.py`: minimal asynchronous baseline with staleness filtering
- `scripts/run_sync.sh`: runnable sync entrypoint
- `scripts/run_async.sh`: runnable async entrypoint
- `tests/test_coding_reward.py`: unit tests for extraction + timeout + syntax errors
- `slime_examples/psc_async_project/fully_async_rollout.py`: tiny Slime-shaped hook (lightweight adapter)
- `slime_examples/psc_async_project/run_async_tiny.sh`: fallback runner to standalone async baseline

## Environment Assumptions (PSC)

- Python environment already available
- PyTorch available for future model/training work (not required for dummy baseline)
- No requirement for FlashAttention/Apex/Transformer Engine
- Run commands from repo root

## Run Sync Baseline

```bash
bash scripts/run_sync.sh
```

Outputs:

- `results/sync_results.jsonl`
- `results/sync_summary.json`

Each row includes `task_id`, response length, reward, latency, and tokens/sec.

## Run Async Baseline

```bash
bash scripts/run_async.sh
```

Outputs:

- `results/async_results.jsonl`
- `results/async_summary.json`

Rows include policy metadata:

- `policy_version`
- `enqueue_time`
- `dequeue_time`
- `current_policy_version`
- `staleness`
- `dropped` (when `staleness > K`)

Default staleness bound is `K=1`.

## Reward Harness CLI

`src/coding_reward.py` can be used standalone:

```bash
python3 -m src.coding_reward --input sample.json
```

Expected `sample.json` format:

```json
{
  "response": "```python\\ndef f(x):\\n    return x + 1\\n```",
  "task": {
    "task_id": "example",
    "prompt": "Write f(x)",
    "reference_tests": ["assert f(1) == 2"],
    "starter_code": "def f(x):\\n    pass\\n"
  }
}
```

## Unit Tests

```bash
python3 -m unittest tests/test_coding_reward.py
```

## Slime Integration Note

`slime_examples/psc_async_project/fully_async_rollout.py` provides a tiny adapter with the expected function shape and lightweight policy metadata tagging. If your Slime version needs deeper hooks, use the standalone baseline scripts as the primary path and treat this adapter as a minimal scaffold.

## Known Limitations

- This is rollout + reward only (no true PPO optimization loop yet)
- Async policy version increments are simulated by a periodic updater thread
- Dummy backend returns canned solutions; HF backend is a lightweight stub for future use
- Sandboxing is timeout-based and minimal (good for tiny deterministic tasks, not untrusted arbitrary code hardening)

## Path to Full PPO Later

To evolve this into trainable PPO while preserving experiment parity:

1. Keep `src/coding_task.py` and `src/coding_reward.py` unchanged for both modes.
2. Replace `NoOpTrainer` in sync with PPO update steps on collected trajectories.
3. Replace async `train_buffer` placeholder with a minibatch learner that increments policy version after optimizer steps.
4. Keep staleness filtering in async and match logging fields across sync/async.
5. Continue writing JSONL rows and summary JSON from `src/metrics.py` for apples-to-apples analysis.
