"""
Tiny Slime-shaped async rollout hook for PSC experiments.

This file intentionally avoids heavy dependencies and advanced optimizations.
If this adapter is not compatible with your current Slime commit, use the
standalone scripts in `scripts/run_sync.sh` and `scripts/run_async.sh`.
"""

import time


def _maybe_set(sample, key: str, value) -> None:
    try:
        setattr(sample, key, value)
    except Exception:
        pass


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode is not implemented in this tiny adapter.")

    target_batch = int(getattr(args, "rollout_batch_size", 1))
    groups = data_buffer.get_samples(target_batch)
    now = time.time()

    # Lightweight metadata tagging to mirror the standalone async baseline.
    policy_version = int(getattr(args, "policy_version", 0))
    for group in groups:
        for sample in group:
            _maybe_set(sample, "policy_version", policy_version)
            _maybe_set(sample, "enqueue_time", now)
            _maybe_set(sample, "dequeue_time", now)

    return groups
