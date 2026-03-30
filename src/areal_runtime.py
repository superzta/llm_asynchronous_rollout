import multiprocessing as mp
import time

from src.areal_controller import run_controller
from src.areal_parameter_service import run_parameter_service
from src.areal_rollout_worker import run_rollout_worker
from src.areal_trainer_worker import run_trainer_worker
from src.coding_task import build_model_prompt, load_tasks, repeat_tasks
from src.model_backends import build_backend


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

    base_backend = build_backend(args)
    initial_state = base_backend.get_trainable_state()

    rollout_device_list = _assign_devices(_parse_device_list(args.rollout_devices), args.num_rollout_workers)
    trainer_device_list = _assign_devices(_parse_device_list(args.trainer_devices), args.num_trainer_workers)

    manager = mp.Manager()
    shared_state = manager.dict()
    shared_state["trainable_state"] = initial_state
    policy_version_value = mp.Value("i", 0)
    global_update_count_value = mp.Value("i", 0)
    trainer_update_counts = manager.dict()

    stop_event = mp.Event()
    parameter_update_queue = mp.Queue(maxsize=args.queue_maxsize)
    parameter_commit_queue = mp.Queue(maxsize=args.queue_maxsize)
    rollout_sample_queue = mp.Queue(maxsize=args.queue_maxsize)
    rollout_status_queue = mp.Queue()
    trainer_status_queue = mp.Queue()

    rollout_task_queues = []
    for _ in range(args.num_rollout_workers):
        rollout_task_queues.append(mp.Queue(maxsize=args.queue_maxsize))

    trainer_input_queues = []
    for _ in range(args.num_trainer_workers):
        trainer_input_queues.append(mp.Queue(maxsize=args.queue_maxsize))

    parameter_service_proc = mp.Process(
        target=run_parameter_service,
        args=(
            parameter_update_queue,
            parameter_commit_queue,
            shared_state,
            policy_version_value,
            global_update_count_value,
            trainer_update_counts,
            stop_event,
        ),
        daemon=True,
    )
    parameter_service_proc.start()

    rollout_procs = []
    for worker_id in range(args.num_rollout_workers):
        proc = mp.Process(
            target=run_rollout_worker,
            args=(
                worker_id,
                rollout_device_list[worker_id],
                base_backend,
                rollout_task_queues[worker_id],
                rollout_sample_queue,
                rollout_status_queue,
                shared_state,
                policy_version_value,
                stop_event,
                args.interrupt_check_interval_sec,
                args.generation_chunk_size,
            ),
            daemon=True,
        )
        proc.start()
        rollout_procs.append(proc)

    trainer_procs = []
    for worker_id in range(args.num_trainer_workers):
        proc = mp.Process(
            target=run_trainer_worker,
            args=(
                worker_id,
                trainer_device_list[worker_id],
                base_backend,
                trainer_input_queues[worker_id],
                parameter_update_queue,
                trainer_status_queue,
                shared_state,
                policy_version_value,
                stop_event,
                args.learner_delay_sec,
            ),
            daemon=True,
        )
        proc.start()
        trainer_procs.append(proc)

    controller_out = run_controller(
        args=args,
        tasks=task_payloads,
        rollout_task_queues=rollout_task_queues,
        rollout_sample_queue=rollout_sample_queue,
        trainer_queues=trainer_input_queues,
        parameter_commit_queue=parameter_commit_queue,
        policy_version_value=policy_version_value,
    )

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
        "accepted_fraction": accepted / max(total_seen, 1.0),
        "dropped_fraction": dropped / max(total_seen, 1.0),
        "mean_staleness": _safe_mean([float(x) for x in staleness_vals]),
        "max_staleness": max(staleness_vals) if staleness_vals else 0,
        "avg_reward": _safe_mean(rewards),
        "pass_rate": _safe_mean(pass_vals),
        "wall_clock_sec": wall_clock_sec,
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
        "reward_by_step": controller_out["reward_by_step"],
        "pass_rate_by_step": controller_out["pass_rate_by_step"],
        "tokens_per_sec_by_step": controller_out["tokens_per_sec_by_step"],
        "update_history": controller_out["update_history"],
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
            "num_rollout_workers": args.num_rollout_workers,
            "num_trainer_workers": args.num_trainer_workers,
            "rollout_devices": args.rollout_devices,
            "trainer_devices": args.trainer_devices,
        },
    }
    return {"rows": rows, "summary": summary}
