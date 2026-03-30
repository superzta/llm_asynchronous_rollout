import queue
import time


def run_parameter_service(
    update_queue,
    commit_queue,
    shared_state,
    policy_version_value,
    global_update_count_value,
    trainer_update_counts,
    stop_event,
):
    """
    Applies trainer updates serially and owns global policy_version advancement.
    """
    while not stop_event.is_set():
        try:
            message = update_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if message is None:
            break
        if message.get("type") != "trainer_update":
            continue

        trainer_worker_id = int(message["trainer_worker_id"])
        batch_id = message["batch_id"]
        sample_ids = list(message.get("sample_ids", []))
        update_info = dict(message.get("update_info", {}))
        new_state = message.get("new_state", {})

        if not update_info.get("updated", False):
            commit_queue.put(
                {
                    "type": "update_commit",
                    "trainer_worker_id": trainer_worker_id,
                    "batch_id": batch_id,
                    "sample_ids": sample_ids,
                    "updated": False,
                    "policy_version": int(policy_version_value.value),
                    "loss": float(update_info.get("loss", 0.0)),
                    "avg_reward": float(update_info.get("avg_reward", 0.0)),
                    "timestamp": time.time(),
                }
            )
            continue

        shared_state["trainable_state"] = new_state
        policy_version_value.value += 1
        global_update_count_value.value += 1
        trainer_update_counts[str(trainer_worker_id)] = int(
            trainer_update_counts.get(str(trainer_worker_id), 0)
        ) + 1

        commit_queue.put(
            {
                "type": "update_commit",
                "trainer_worker_id": trainer_worker_id,
                "batch_id": batch_id,
                "sample_ids": sample_ids,
                "updated": True,
                "policy_version": int(policy_version_value.value),
                "loss": float(update_info.get("loss", 0.0)),
                "avg_reward": float(update_info.get("avg_reward", 0.0)),
                "timestamp": time.time(),
            }
        )
