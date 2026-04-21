import multiprocessing as mp
import os
import time

from src.areal_controller import run_controller
from src.areal_parameter_service import run_parameter_service
from src.areal_rollout_worker import run_rollout_worker
from src.areal_trainer_worker import run_trainer_worker
from src.coding_task import build_model_prompt, load_tasks, repeat_tasks


def _safe_mean(values):
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _parse_device_list(raw):
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        return ["cpu"]
    return items


def _assign_devices(base_devices, num_workers):
    output = []
    for i in range(num_workers):
        output.append(base_devices[i % len(base_devices)])
    return output


def run_areal_style(args):
    tasks = repeat_tasks(load_tasks(args.dataset), args.epochs)
    task_payloads = []
    for task in tasks:
        task_payloads.append({"task": task.to_dict(), "prompt": build_model_prompt(task), "task_id": task.task_id})

    # IMPORTANT: do NOT instantiate a GPU-resident backend in the main process.
    # Forking a process that has a live CUDA context silently hangs any CUDA
    # call in the children. Workers will build their own backend on their
    # assigned device after spawn. Shared state starts empty; each worker
    # bootstraps weights from the HF cache (deterministic for a fixed
    # model_id), and subsequent updates flow through the parameter service.
    rollout_device_list = _assign_devices(_parse_device_list(args.rollout_devices), args.num_rollout_workers)
    trainer_device_list = _assign_devices(_parse_device_list(args.trainer_devices), args.num_trainer_workers)

    # Use a spawn context so children get a fresh interpreter and a clean
    # CUDA state. This lets the main process stay CUDA-free and workers each
    # initialize torch.cuda on their assigned GPU.
    ctx = mp.get_context("spawn")

    manager = ctx.Manager()
    shared_state = manager.dict()
    shared_state["trainable_state"] = {}
    policy_version_value = ctx.Value("i", 0)
    global_update_count_value = ctx.Value("i", 0)
    trainer_update_counts = manager.dict()

    stop_event = ctx.Event()
    parameter_update_queue = ctx.Queue(maxsize=args.queue_maxsize)
    parameter_commit_queue = ctx.Queue(maxsize=args.queue_maxsize)
    rollout_sample_queue = ctx.Queue(maxsize=args.queue_maxsize)
    rollout_status_queue = ctx.Queue()
    trainer_status_queue = ctx.Queue()
    event_queue = ctx.Queue(maxsize=max(args.queue_maxsize * 4, 256))
    # Workers put a ready message here after their backend is on-device.
    # The driver blocks on this queue before starting the controller timer,
    # so cold-start / HF-cache-miss model-load time is NOT counted in
    # wall_clock_sec. It is reported separately as setup_wall_clock_sec.
    worker_ready_queue = ctx.Queue()

    rollout_task_queues = []
    for _ in range(args.num_rollout_workers):
        rollout_task_queues.append(ctx.Queue(maxsize=args.queue_maxsize))

    trainer_input_queues = []
    for _ in range(args.num_trainer_workers):
        trainer_input_queues.append(ctx.Queue(maxsize=args.queue_maxsize))

    parameter_service_proc = ctx.Process(
        target=run_parameter_service,
        args=(
            parameter_update_queue,
            parameter_commit_queue,
            shared_state,
            policy_version_value,
            global_update_count_value,
            trainer_update_counts,
            stop_event,
            event_queue,
        ),
        daemon=True,
    )
    parameter_service_proc.start()

    rollout_procs = []
    for worker_id in range(args.num_rollout_workers):
        proc = ctx.Process(
            target=run_rollout_worker,
            args=(
                worker_id,
                rollout_device_list[worker_id],
                args,
                rollout_task_queues[worker_id],
                rollout_sample_queue,
                rollout_status_queue,
                shared_state,
                policy_version_value,
                stop_event,
                args.interrupt_check_interval_sec,
                args.generation_chunk_size,
                args.rollout_chunk_delay_sec,
                event_queue,
                worker_ready_queue,
            ),
            daemon=True,
        )
        proc.start()
        rollout_procs.append(proc)

    trainer_procs = []
    for worker_id in range(args.num_trainer_workers):
        proc = ctx.Process(
            target=run_trainer_worker,
            args=(
                worker_id,
                trainer_device_list[worker_id],
                args,
                trainer_input_queues[worker_id],
                parameter_update_queue,
                trainer_status_queue,
                shared_state,
                policy_version_value,
                stop_event,
                args.learner_delay_sec,
                event_queue,
                worker_ready_queue,
            ),
            daemon=True,
        )
        proc.start()
        trainer_procs.append(proc)

    # ------------------------------------------------------------------
    # Worker-ready barrier
    # ------------------------------------------------------------------
    # Wait for every rollout and trainer worker to finish building its
    # backend on-device before we let the controller start its clock.
    # This excludes cold-start model-load time (HF cache miss, page cache
    # cold, etc.) from wall_clock_sec. A slow disk or shared filesystem
    # can otherwise add tens of seconds of "work-free" time to the
    # measurement and deflate updates/sec for no algorithmic reason.
    import sys as _sys

    total_workers_to_wait = int(args.num_rollout_workers) + int(args.num_trainer_workers)
    setup_timeout_sec = float(os.environ.get("AREAL_SETUP_TIMEOUT_SEC", "900"))
    setup_poll_sec = float(os.environ.get("AREAL_SETUP_POLL_SEC", "5"))
    setup_t0 = time.time()
    _sys.stderr.write(
        "[areal-setup] waiting for %d workers to build backends (timeout=%.0fs)...\n"
        % (total_workers_to_wait, setup_timeout_sec)
    )
    _sys.stderr.flush()
    ready_workers = 0
    while ready_workers < total_workers_to_wait:
        elapsed_setup = time.time() - setup_t0
        if elapsed_setup > setup_timeout_sec:
            raise RuntimeError(
                "AReaL setup timed out: only %d/%d workers ready after %.1fs"
                % (ready_workers, total_workers_to_wait, elapsed_setup)
            )
        try:
            msg = worker_ready_queue.get(timeout=setup_poll_sec)
        except Exception:
            msg = None
        if msg is None:
            for p in list(rollout_procs) + list(trainer_procs):
                if (not p.is_alive()) and (p.exitcode not in (None, 0)):
                    raise RuntimeError(
                        "Worker process died during backend build: pid=%s exitcode=%s"
                        % (p.pid, p.exitcode)
                    )
            _sys.stderr.write(
                "[areal-setup] t=%5.1fs still waiting (%d/%d ready)\n"
                % (elapsed_setup, ready_workers, total_workers_to_wait)
            )
            _sys.stderr.flush()
            continue
        ready_workers += 1
        _sys.stderr.write(
            "[areal-setup] t=%5.1fs %s.%s ready on %s (%d/%d)\n"
            % (
                time.time() - setup_t0,
                msg.get("type", "?"),
                msg.get("worker_id", "?"),
                msg.get("device", "?"),
                ready_workers,
                total_workers_to_wait,
            )
        )
        _sys.stderr.flush()
    setup_wall_clock_sec = time.time() - setup_t0
    _sys.stderr.write(
        "[areal-setup] all workers ready after %.1fs. starting controller timer now.\n"
        % setup_wall_clock_sec
    )
    _sys.stderr.flush()

    heartbeat_stop = [False]

    def _heartbeat():
        import threading as _th
        import time as _time
        from src.progress import ProgressReporter as _PR
        reporter = _PR(tag="areal", total=len(task_payloads))
        reporter.note(
            "starting: tasks=%d rollout=%s trainer=%s k=%s update_batch=%s"
            % (
                len(task_payloads),
                rollout_device_list,
                trainer_device_list,
                getattr(args, "staleness_k", "?"),
                getattr(args, "update_batch_size", "?"),
            )
        )
        interval = max(1.0, float(os.environ.get("AREAL_HEARTBEAT_SEC", "5")))
        while not heartbeat_stop[0]:
            try:
                rsq = rollout_sample_queue.qsize()
            except Exception:
                rsq = -1
            try:
                tiqs = [q.qsize() for q in trainer_input_queues]
            except Exception:
                tiqs = []
            reporter.log(
                step=int(global_update_count_value.value),
                pv=int(policy_version_value.value),
                sq=rsq,
                tq=",".join(str(x) for x in tiqs),
            )
            _time.sleep(interval)

    import threading as _threading
    heartbeat_thread = _threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        controller_out = run_controller(
            args=args,
            tasks=task_payloads,
            rollout_task_queues=rollout_task_queues,
            rollout_sample_queue=rollout_sample_queue,
            trainer_queues=trainer_input_queues,
            parameter_commit_queue=parameter_commit_queue,
            policy_version_value=policy_version_value,
            event_queue=event_queue,
        )
    finally:
        heartbeat_stop[0] = True

    for q in trainer_input_queues:
        q.put(None)
    for proc in rollout_procs:
        proc.join(timeout=5.0)
    for proc in trainer_procs:
        proc.join(timeout=5.0)

    stop_event.set()
    parameter_update_queue.put(None)
    parameter_service_proc.join(timeout=5.0)

    rollout_worker_stats = {}
    for _ in range(args.num_rollout_workers):
        try:
            stat = rollout_status_queue.get(timeout=1.0)
        except Exception:
            continue
        rollout_worker_stats[str(stat["worker_id"])] = stat

    trainer_worker_stats = {}
    for _ in range(args.num_trainer_workers):
        try:
            stat = trainer_status_queue.get(timeout=1.0)
        except Exception:
            continue
        trainer_worker_stats[str(stat["worker_id"])] = stat

    accepted_rows = controller_out["accepted_rows"]
    dropped_rows = controller_out["dropped_rows"]
    rows = controller_out["rows"]
    events = list(controller_out.get("events", []))
    while True:
        try:
            events.append(event_queue.get_nowait())
        except Exception:
            break

    rewards = [float(r.get("reward", 0.0)) for r in accepted_rows]
    pass_vals = [1.0 if r.get("pass", False) else 0.0 for r in accepted_rows]
    staleness_vals = [int(r.get("staleness", 0)) for r in accepted_rows]
    tokens_per_sec_vals = [float(r.get("tokens_per_sec", 0.0)) for r in accepted_rows]
    total_seen = float(controller_out["num_total_seen"])
    accepted = float(controller_out["num_accepted"])
    dropped = float(controller_out["num_dropped"])
    wall_clock_sec = float(controller_out["wall_clock_sec"])
    update_count = int(controller_out["update_count"])
    total_reward = sum(rewards)
    total_passes = sum(pass_vals)

    summary = {
        "mode": "async_areal_style",
        "num_rollout_workers": int(args.num_rollout_workers),
        "num_trainer_workers": int(args.num_trainer_workers),
        "rollout_devices": rollout_device_list,
        "trainer_devices": trainer_device_list,
        "update_count": update_count,
        "final_policy_version": int(policy_version_value.value),
        "num_total_seen": int(total_seen),
        "num_accepted": int(accepted),
        "num_dropped": int(dropped),
        "dropped_stale_count": int(controller_out.get("dropped_stale_count", 0)),
        "dropped_interrupted_count": int(controller_out.get("dropped_interrupted_count", 0)),
        "accepted_fraction": accepted / max(total_seen, 1.0),
        "dropped_fraction": dropped / max(total_seen, 1.0),
        "mean_staleness": _safe_mean([float(x) for x in staleness_vals]),
        "max_staleness": max(staleness_vals) if staleness_vals else 0,
        "avg_reward": _safe_mean(rewards),
        "pass_rate": _safe_mean(pass_vals),
        "wall_clock_sec": wall_clock_sec,
        "setup_wall_clock_sec": float(setup_wall_clock_sec),
        "total_wall_clock_sec": float(wall_clock_sec + setup_wall_clock_sec),
        "avg_tokens_per_sec": _safe_mean(tokens_per_sec_vals),
        "queue_depth_trace": controller_out["queue_depth_trace"],
        "replay_buffer_size_trace": controller_out["replay_buffer_size_trace"],
        "per_rollout_worker_generated_count": {
            worker_id: int(info.get("generated_count", 0))
            for worker_id, info in rollout_worker_stats.items()
        },
        "per_rollout_worker_interrupt_count": {
            worker_id: int(info.get("interrupt_count", 0))
            for worker_id, info in rollout_worker_stats.items()
        },
        "per_rollout_worker_last_loaded_policy_version": {
            worker_id: int(info.get("last_loaded_policy_version", 0))
            for worker_id, info in rollout_worker_stats.items()
        },
        "per_trainer_worker_update_count": {
            worker_id: int(info.get("update_count", 0))
            for worker_id, info in trainer_worker_stats.items()
        },
        "interrupted_samples_count": int(controller_out["interrupted_samples"]),
        "interrupt_count_total": int(
            sum([int(info.get("interrupt_count", 0)) for info in rollout_worker_stats.values()])
        ),
        "per_rollout_worker_loaded_versions": {
            worker_id: list(info.get("loaded_versions", [])) for worker_id, info in rollout_worker_stats.items()
        },
        "reward_by_step": controller_out["reward_by_step"],
        "pass_rate_by_step": controller_out["pass_rate_by_step"],
        "tokens_per_sec_by_step": controller_out["tokens_per_sec_by_step"],
        "update_history": controller_out["update_history"],
        "version_change_events": [
            e for e in events if e.get("type") == "version_change"
        ],
        "trainer_commit_timestamps": [
            float(e.get("timestamp", 0.0)) for e in events if e.get("type") in ("controller_trainer_commit", "version_change")
        ],
        "rollout_emit_timestamps": [
            float(e.get("timestamp", 0.0)) for e in events if e.get("type") == "rollout_emit"
        ],
        "staleness_histogram": {},
        "event_trace": events,
        "effective_updates_per_second": update_count / max(wall_clock_sec, 1e-9),
        "reward_per_second": total_reward / max(wall_clock_sec, 1e-9),
        "reward_per_update": total_reward / max(float(update_count), 1.0),
        "pass_rate_per_update": total_passes / max(float(update_count), 1.0),
        "config": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "seed": args.seed,
            "backend": args.backend,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "device": args.device,
            "lr": args.lr,
            "update_batch_size": args.update_batch_size,
            "staleness_k": args.staleness_k,
            "queue_maxsize": args.queue_maxsize,
            "producer_delay_sec": args.producer_delay_sec,
            "learner_delay_sec": args.learner_delay_sec,
            "interrupt_check_interval_sec": args.interrupt_check_interval_sec,
            "generation_chunk_size": args.generation_chunk_size,
            "rollout_chunk_delay_sec": args.rollout_chunk_delay_sec,
            "replay_dispatch_delay_sec": args.replay_dispatch_delay_sec,
            "controller_consume_delay_sec": args.controller_consume_delay_sec,
            "max_interrupt_retries": args.max_interrupt_retries,
            "num_rollout_workers": args.num_rollout_workers,
            "num_trainer_workers": args.num_trainer_workers,
            "rollout_devices": args.rollout_devices,
            "trainer_devices": args.trainer_devices,
            "decoupled_objective": int(getattr(args, "decoupled_objective", 1)),
            "hf_dtype": getattr(args, "hf_dtype", None),
            "hf_attn_impl": getattr(args, "hf_attn_impl", None),
            "hf_chat_template": int(getattr(args, "hf_chat_template", 1)),
            "grpo_epsilon": float(getattr(args, "grpo_epsilon", 0.2)),
            "grpo_kl_coef": float(getattr(args, "grpo_kl_coef", 0.02)),
            "grpo_group_size": int(getattr(args, "grpo_group_size", 0)),
            "grad_clip": float(getattr(args, "grad_clip", 1.0)),
            "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        },
    }
    terminal_seen = float(int(controller_out.get("num_accepted", 0)) + int(controller_out.get("num_dropped", 0)))
    summary["num_terminal_seen"] = int(terminal_seen)
    summary["accepted_fraction_terminal"] = accepted / max(terminal_seen, 1.0)
    summary["dropped_fraction_terminal"] = dropped / max(terminal_seen, 1.0)
    histogram = {}
    for value in staleness_vals:
        key = str(int(value))
        histogram[key] = int(histogram.get(key, 0)) + 1
    summary["staleness_histogram"] = histogram
    return {"rows": rows, "summary": summary}
