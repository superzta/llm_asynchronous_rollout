import queue
import time

from src.coding_reward import evaluate_response


def _dispatch_training_batch(trainer_queues, trainer_rr_idx, batch_id, batch_rows):
    trainer_id = trainer_rr_idx % max(len(trainer_queues), 1)
    train_samples = []
    sample_ids = []
    for row in batch_rows:
        sample_ids.append(int(row["sample_id"]))
        train_samples.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "response": row["response"],
                "reward": float(row["reward"]),
                "metadata": dict(row.get("metadata", {})),
            }
        )
    trainer_queues[trainer_id].put(
        {
            "batch_id": int(batch_id),
            "sample_ids": sample_ids,
            "train_samples": train_samples,
        }
    )
    return trainer_id


def run_controller(
    args,
    tasks,
    rollout_task_queues,
    rollout_sample_queue,
    trainer_queues,
    parameter_commit_queue,
    policy_version_value,
):
    """
    Main controller loop:
    - dispatch rollout prompts
    - evaluate reward
    - bounded staleness filtering
    - replay buffering
    - dispatch train batches
    - consume parameter service commits
    """
    all_rows = []
    accepted_rows = []
    dropped_rows = []
    row_by_sample_id = {}
    update_history = []
    replay_buffer = []
    queue_depth_trace = []
    replay_buffer_size_trace = []
    reward_by_step = []
    pass_rate_by_step = []
    tokens_per_sec_by_step = []

    num_total_seen = 0
    num_accepted = 0
    num_dropped = 0
    interrupted_samples = 0
    trainer_rr_idx = 0
    batch_id = 0
    in_flight_batches = 0
    update_count = 0
    next_sample_id = 0

    # Dispatch rollout tasks round-robin.
    for step_idx, task in enumerate(tasks, start=1):
        prompt = task["prompt"]
        queue_idx = (step_idx - 1) % len(rollout_task_queues)
        rollout_task_queues[queue_idx].put(
            {
                "sample_id": next_sample_id,
                "task": task["task"],
                "prompt": prompt,
                "enqueue_time": time.time(),
                "step": step_idx,
            }
        )
        next_sample_id += 1
        if args.producer_delay_sec > 0:
            time.sleep(args.producer_delay_sec)

    total_tasks = len(tasks)
    for q in rollout_task_queues:
        q.put(None)

    start_time = time.time()
    last_trace_time = 0.0
    replay_tail_dispatched = False

    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed - last_trace_time >= args.queue_trace_interval_sec:
            trainer_depth = 0
            for tq in trainer_queues:
                trainer_depth += tq.qsize()
            queue_depth_trace.append(
                {
                    "t_sec": elapsed,
                    "rollout_queue_depth": rollout_sample_queue.qsize(),
                    "trainer_queue_depth": trainer_depth,
                }
            )
            replay_buffer_size_trace.append({"t_sec": elapsed, "replay_buffer_size": len(replay_buffer)})
            last_trace_time = elapsed

        # Handle rollout sample output.
        if num_total_seen < total_tasks:
            try:
                sample = rollout_sample_queue.get(timeout=0.05)
            except queue.Empty:
                sample = None
            if sample is not None:
                num_total_seen += 1
                current_global_version = int(policy_version_value.value)
                interrupted = bool(sample.get("interrupted", False))
                if interrupted:
                    interrupted_samples += 1
                    row = {
                        "sample_id": int(sample["sample_id"]),
                        "task_id": sample["task_id"],
                        "prompt": sample["prompt"],
                        "rollout_worker_id": int(sample["rollout_worker_id"]),
                        "rollout_device": sample.get("rollout_device", "cpu"),
                        "trainer_worker_id": None,
                        "policy_version": int(sample["policy_version"]),
                        "current_global_policy_version": current_global_version,
                        "staleness": max(0, current_global_version - int(sample["policy_version"])),
                        "interrupted": True,
                        "dropped": True,
                        "reward": 0.0,
                        "pass": False,
                        "num_passed": 0,
                        "total_tests": len(sample["task"]["reference_tests"]),
                        "response": "",
                        "response_length": 0,
                        "tokens_per_sec": float(sample.get("tokens_per_sec", 0.0)),
                        "enqueue_time": float(sample.get("enqueue_time", 0.0)),
                        "dequeue_time": float(sample.get("dequeue_time", 0.0)),
                        "error_type": "interrupted_before_emit",
                        "step": int(sample.get("step", 0)),
                    }
                    num_dropped += 1
                    dropped_rows.append(row)
                    all_rows.append(row)
                    continue

                reward_info = evaluate_response(
                    response_text=sample["response"],
                    task=sample["task"],
                    timeout_sec=args.reward_timeout_sec,
                )
                staleness = max(0, current_global_version - int(sample["policy_version"]))
                dropped = staleness > args.staleness_k
                row = {
                    "sample_id": int(sample["sample_id"]),
                    "task_id": sample["task_id"],
                    "prompt": sample["prompt"],
                    "rollout_worker_id": int(sample["rollout_worker_id"]),
                    "rollout_device": sample.get("rollout_device", "cpu"),
                    "trainer_worker_id": None,
                    "policy_version": int(sample["policy_version"]),
                    "current_global_policy_version": current_global_version,
                    "staleness": int(staleness),
                    "interrupted": False,
                    "dropped": bool(dropped),
                    "reward": float(reward_info["reward"]) if not dropped else 0.0,
                    "pass": bool(reward_info["pass"]) if not dropped else False,
                    "num_passed": int(reward_info["num_passed"]) if not dropped else 0,
                    "total_tests": int(reward_info["total_tests"]),
                    "response": sample["response"],
                    "response_length": int(sample["response_length"]),
                    "tokens_per_sec": float(sample.get("tokens_per_sec", 0.0)),
                    "enqueue_time": float(sample.get("enqueue_time", 0.0)),
                    "dequeue_time": float(sample.get("dequeue_time", 0.0)),
                    "error_type": "dropped_stale" if dropped else reward_info.get("error_type"),
                    "step": int(sample.get("step", 0)),
                    "metadata": dict(sample.get("metadata", {})),
                    "logprob_sum": float(sample.get("logprob_sum", 0.0)),
                }
                row_by_sample_id[row["sample_id"]] = row
                all_rows.append(row)
                if dropped:
                    num_dropped += 1
                    dropped_rows.append(row)
                else:
                    num_accepted += 1
                    accepted_rows.append(row)
                    replay_buffer.append(row)
                    reward_by_step.append({"step": row["step"], "reward": row["reward"]})
                    pass_rate_by_step.append({"step": row["step"], "pass_rate": 1.0 if row["pass"] else 0.0})
                    tokens_per_sec_by_step.append({"step": row["step"], "tokens_per_sec": row["tokens_per_sec"]})

        # Dispatch full replay batches.
        while len(replay_buffer) >= args.update_batch_size:
            batch_rows = replay_buffer[: args.update_batch_size]
            replay_buffer = replay_buffer[args.update_batch_size :]
            trainer_id = _dispatch_training_batch(
                trainer_queues=trainer_queues,
                trainer_rr_idx=trainer_rr_idx,
                batch_id=batch_id,
                batch_rows=batch_rows,
            )
            trainer_rr_idx += 1
            in_flight_batches += 1
            for row in batch_rows:
                row["trainer_worker_id"] = trainer_id
            batch_id += 1

        # Once rollout is complete, flush tail batch once.
        rollout_done = num_total_seen >= total_tasks
        if rollout_done and replay_buffer and not replay_tail_dispatched:
            batch_rows = list(replay_buffer)
            replay_buffer = []
            trainer_id = _dispatch_training_batch(
                trainer_queues=trainer_queues,
                trainer_rr_idx=trainer_rr_idx,
                batch_id=batch_id,
                batch_rows=batch_rows,
            )
            trainer_rr_idx += 1
            in_flight_batches += 1
            for row in batch_rows:
                row["trainer_worker_id"] = trainer_id
            batch_id += 1
            replay_tail_dispatched = True

        # Process parameter service commits.
        while True:
            try:
                commit = parameter_commit_queue.get_nowait()
            except queue.Empty:
                break
            if commit.get("type") != "update_commit":
                continue
            in_flight_batches = max(0, in_flight_batches - 1)
            if commit.get("updated", False):
                update_count += 1
                update_history.append(
                    {
                        "update_id": int(update_count),
                        "trainer_worker_id": int(commit["trainer_worker_id"]),
                        "policy_version": int(commit["policy_version"]),
                        "batch_id": int(commit["batch_id"]),
                        "loss": float(commit.get("loss", 0.0)),
                        "avg_reward": float(commit.get("avg_reward", 0.0)),
                        "sample_ids": list(commit.get("sample_ids", [])),
                        "timestamp": float(commit.get("timestamp", time.time())),
                    }
                )
                for sample_id in commit.get("sample_ids", []):
                    row = row_by_sample_id.get(int(sample_id))
                    if row is not None:
                        row["trainer_worker_id"] = int(commit["trainer_worker_id"])
                        row["updated_by_trainer"] = True
                        row["policy_version_after_commit"] = int(commit["policy_version"])

        if rollout_done and in_flight_batches == 0 and not replay_buffer:
            break

    wall_clock_sec = time.time() - start_time
    return {
        "rows": all_rows,
        "accepted_rows": accepted_rows,
        "dropped_rows": dropped_rows,
        "update_history": update_history,
        "reward_by_step": reward_by_step,
        "pass_rate_by_step": pass_rate_by_step,
        "tokens_per_sec_by_step": tokens_per_sec_by_step,
        "queue_depth_trace": queue_depth_trace,
        "replay_buffer_size_trace": replay_buffer_size_trace,
        "num_total_seen": int(num_total_seen),
        "num_accepted": int(num_accepted),
        "num_dropped": int(num_dropped),
        "interrupted_samples": int(interrupted_samples),
        "update_count": int(update_count),
        "wall_clock_sec": float(wall_clock_sec),
    }
