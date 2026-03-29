import time

def trainer_worker_fn(queue, device="cuda:1"):
    current_policy_version = 0

    while True:
        item = queue.get()

        if item is None:
            print("[TRAINER] No more rollout data. Exiting.")
            break

        print(f"[TRAINER] Pulled task {item['task_id']} from queue")
        print(f"[TRAINER] Reward = {item['reward']:.2f}, "
              f"Tests Passed = {item['passed']}/{item['total_tests']}")
        print(f"[TRAINER] Old policy version = {item['policy_version']}, "
              f"Current trainer version = {current_policy_version}")

        # Placeholder for PPO update
        time.sleep(2)

        current_policy_version += 1
        print(f"[TRAINER] Updated policy to version {current_policy_version}")
        print("=" * 80)