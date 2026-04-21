import queue
import time

from src.coding_task import CodingTask


def _resolve_device(device):
    if str(device).startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def _emit_event(event_queue, payload):
    try:
        event_queue.put_nowait(payload)
    except queue.Full:
        pass


def run_rollout_worker(
    worker_id,
    device,
    backend,
    task_queue,
    sample_queue,
    status_queue,
    shared_state,
    policy_version_value,
    stop_event,
    interrupt_check_interval_sec,
    generation_chunk_size,
    rollout_chunk_delay_sec,
    event_queue,
):
    worker_device = _resolve_device(device)
    local_backend = backend.clone_for_device(worker_device)
    local_backend.load_trainable_state(shared_state.get("trainable_state", {}))
    local_version = int(policy_version_value.value)

    generated_count = 0
    interrupt_count = 0
    interrupted_samples = 0
    last_loaded_policy_version = local_version
    loaded_versions = [int(local_version)]

    while not stop_event.is_set():
        try:
            payload = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if payload is None:
            break

        sample_id = int(payload["sample_id"])
        task = CodingTask.from_dict(payload["task"])
        prompt = payload["prompt"]
        enqueue_time = float(payload["enqueue_time"])
        step = int(payload.get("step", 0))
        retry_count = int(payload.get("retry_count", 0))

        generation_state = None
        interrupted = False
        while True:
            global_version = int(policy_version_value.value)
            if global_version > local_version:
                # Interrupt before next safe chunk boundary and refresh weights.
                interrupt_count += 1
                interrupted_samples += 1
                interrupted = True
                _emit_event(
                    event_queue,
                    {
                        "type": "rollout_interrupt_observed",
                        "worker_id": int(worker_id),
                        "sample_id": int(sample_id),
                        "old_version": int(local_version),
                        "new_version": int(global_version),
                        "timestamp": time.time(),
                    },
                )
                local_backend.load_trainable_state(shared_state.get("trainable_state", {}))
                local_version = global_version
                last_loaded_policy_version = local_version
                loaded_versions.append(int(local_version))
                _emit_event(
                    event_queue,
                    {
                        "type": "rollout_load_version",
                        "worker_id": int(worker_id),
                        "sample_id": int(sample_id),
                        "loaded_version": int(local_version),
                        "timestamp": time.time(),
                    },
                )
                break

            chunk_result = local_backend.maybe_generate_chunk(
                prompt=prompt,
                task=task,
                generation_state=generation_state,
                chunk_size=generation_chunk_size,
            )
            if chunk_result.get("done", False):
                result = chunk_result["result"]
                generated_count += 1
                sample_queue.put(
                    {
                        "sample_id": sample_id,
                        "task_id": task.task_id,
                        "task": task.to_dict(),
                        "prompt": prompt,
                        "response": result.text,
                        "response_length": len(result.text),
                        "rollout_worker_id": int(worker_id),
                        "rollout_device": worker_device,
                        "policy_version": local_version,
                        "enqueue_time": enqueue_time,
                        "dequeue_time": time.time(),
                        "generation_latency_sec": float(result.latency_sec),
                        "tokens_per_sec": float(result.tokens_per_sec),
                        "logprob_sum": float(result.logprob_sum),
                        "metadata": dict(result.metadata),
                        "interrupted": False,
                        "step": step,
                        "retry_count": retry_count,
                    }
                )
                _emit_event(
                    event_queue,
                    {
                        "type": "rollout_emit",
                        "worker_id": int(worker_id),
                        "sample_id": int(sample_id),
                        "policy_version": int(local_version),
                        "timestamp": time.time(),
                    },
                )
                break

            generation_state = chunk_result.get("generation_state")
            if rollout_chunk_delay_sec > 0:
                time.sleep(rollout_chunk_delay_sec)
            if interrupt_check_interval_sec > 0:
                time.sleep(interrupt_check_interval_sec)

        if interrupted:
            sample_queue.put(
                {
                    "sample_id": sample_id,
                    "task_id": task.task_id,
                    "task": task.to_dict(),
                    "prompt": prompt,
                    "response": "",
                    "response_length": 0,
                    "rollout_worker_id": int(worker_id),
                    "rollout_device": worker_device,
                    "policy_version": local_version,
                    "enqueue_time": enqueue_time,
                    "dequeue_time": time.time(),
                    "generation_latency_sec": 0.0,
                    "tokens_per_sec": 0.0,
                    "logprob_sum": 0.0,
                    "metadata": {},
                    "interrupted": True,
                    "step": step,
                    "retry_count": retry_count,
                }
            )
            _emit_event(
                event_queue,
                {
                    "type": "rollout_sample_interrupted",
                    "worker_id": int(worker_id),
                    "sample_id": int(sample_id),
                    "policy_version": int(local_version),
                    "timestamp": time.time(),
                },
            )

    status_queue.put(
        {
            "worker_id": int(worker_id),
            "generated_count": int(generated_count),
            "interrupt_count": int(interrupt_count),
            "interrupted_samples": int(interrupted_samples),
            "last_loaded_policy_version": int(last_loaded_policy_version),
            "loaded_versions": loaded_versions,
            "device": worker_device,
        }
    )
