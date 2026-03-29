import time


def trainer_worker_fn(
    queue,
    device: str = "cuda:1",
    mode: str = "async",
    update_sleep_time: float = 2.0,
):
    current_policy_version = 0

    start_time = time.time()
    num_samples = 0
    num_updates = 0
    total_reward = 0.0
    total_lag = 0.0

    while True:
        data = queue.get()

        if data is None:
            print("[TRAINER] No more rollout data. Exiting.")
            break

        if mode == "sync":
            batch = data
            print(f"[TRAINER][SYNC] Received batch of {len(batch)} rollouts.")
        else:
            batch = [data]

        for item in batch:
            lag = current_policy_version - item["policy_version"]

            print(f"[TRAINER] Pulled task {item['task_id']} from queue")
            print(
                f"[TRAINER] Reward = {item['reward']:.2f}, "
                f"Tests Passed = {item['passed']}/{item['total_tests']}"
            )
            print(
                f"[TRAINER] Old policy version = {item['policy_version']}, "
                f"Current trainer version = {current_policy_version}, "
                f"Policy lag = {lag}"
            )

            # Placeholder for PPO update
            if update_sleep_time > 0:
                time.sleep(update_sleep_time)

            current_policy_version += 1
            num_updates += 1
            num_samples += 1
            total_reward += item["reward"]
            total_lag += lag

            elapsed = time.time() - start_time
            avg_reward = total_reward / num_samples if num_samples > 0 else 0.0
            avg_lag = total_lag / num_samples if num_samples > 0 else 0.0
            throughput = num_samples / elapsed if elapsed > 0 else 0.0

            try:
                qsize = queue.qsize()
            except (NotImplementedError, AttributeError):
                qsize = -1

            print(f"[TRAINER] Updated policy to version {current_policy_version}")
            print(
                f"[METRICS] samples={num_samples} | "
                f"updates={num_updates} | "
                f"avg_reward={avg_reward:.3f} | "
                f"avg_lag={avg_lag:.3f} | "
                f"throughput={throughput:.2f} samples/sec | "
                f"queue_size={qsize}"
            )
            print("=" * 80)