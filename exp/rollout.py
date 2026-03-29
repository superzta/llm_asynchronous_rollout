import time
from model import load_model, generate_code, get_dummy_logprobs
from reward import compute_code_reward


# collects all the rollout information 
def rollout_worker_fn(
    queue,
    tasks,
    device: str = "cuda:0",
    mode: str = "async",
    max_new_tokens: int = 128,
    sleep_time: float = 1.0,
):
    tokenizer, model = load_model(device)
    policy_version = 0

    sync_batch = []

    for task in tasks:
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        tests = task["tests"]

        generated_code = generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        reward, passed, total = compute_code_reward(
            generated_code=generated_code,
            entry_point=entry_point,
            tests=tests,
        )

        old_logprobs = get_dummy_logprobs(generated_code)

        rollout = {
            "task_id": task["task_id"],
            "prompt": prompt,
            "generated_code": generated_code,
            "reward": reward,
            "passed": passed,
            "total_tests": total,
            "old_logprobs": old_logprobs,
            "policy_version": policy_version,
            "timestamp": time.time(),
        }

        if mode == "async":
            queue.put(rollout)
            print(
                f"[ROLLOUT][ASYNC] Task {task['task_id']} | "
                f"Reward {reward:.2f} ({passed}/{total}) | "
                f"PolicyVersion {policy_version}"
            )
        else:
            sync_batch.append(rollout)
            print(
                f"[ROLLOUT][SYNC-BUFFER] Task {task['task_id']} | "
                f"Reward {reward:.2f} ({passed}/{total}) | "
                f"PolicyVersion {policy_version}"
            )

        print("-" * 80)

        # In a real system, rollout policy would periodically sync
        # from trainer and this version would change.
        policy_version += 1

        if sleep_time > 0:
            time.sleep(sleep_time)

    if mode == "sync":
        queue.put(sync_batch)
        print(f"[ROLLOUT][SYNC] Sent batch of {len(sync_batch)} rollouts to trainer.")

    queue.put(None)