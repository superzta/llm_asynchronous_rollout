import queue
import time


def _resolve_device(device):
    if str(device).startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def run_trainer_worker(
    worker_id,
    device,
    backend,
    trainer_queue,
    parameter_update_queue,
    status_queue,
    shared_state,
    policy_version_value,
    stop_event,
    learner_delay_sec,
):
    worker_device = _resolve_device(device)
    local_backend = backend.clone_for_device(worker_device)
    local_backend.load_trainable_state(shared_state.get("trainable_state", {}))
    local_version = int(policy_version_value.value)

    local_update_count = 0
    last_seen_policy_version = local_version

    while not stop_event.is_set():
        try:
            payload = trainer_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if payload is None:
            break

        latest_version = int(policy_version_value.value)
        if latest_version > local_version:
            local_backend.load_trainable_state(shared_state.get("trainable_state", {}))
            local_version = latest_version
            last_seen_policy_version = latest_version

        batch_id = payload["batch_id"]
        sample_ids = list(payload["sample_ids"])
        train_samples = list(payload["train_samples"])

        update_info = local_backend.policy_gradient_update(train_samples)
        if learner_delay_sec > 0:
            time.sleep(learner_delay_sec)

        if update_info.get("updated", False):
            local_update_count += 1
            new_state = local_backend.get_trainable_state()
        else:
            new_state = {}

        parameter_update_queue.put(
            {
                "type": "trainer_update",
                "trainer_worker_id": int(worker_id),
                "batch_id": int(batch_id),
                "sample_ids": sample_ids,
                "update_info": update_info,
                "new_state": new_state,
                "local_policy_version_before_commit": int(local_version),
            }
        )

    status_queue.put(
        {
            "worker_id": int(worker_id),
            "update_count": int(local_update_count),
            "last_seen_policy_version": int(last_seen_policy_version),
            "device": worker_device,
        }
    )
