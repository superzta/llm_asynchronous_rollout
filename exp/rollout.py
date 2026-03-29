import time
from model_utils import load_model, generate_code, get_dummy_logprobs
from reward_fn import compute_code_reward

def rollout_worker_fn(queue, tasks, device="cuda:0"):
    tokenizer, model = load_model(device)
    policy_version = 0

    for task in tasks:
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        tests = task["tests"]

        generated_code = generate_code(model, tokenizer, prompt, device)
        reward, passed, total = compute_code_reward(generated_code, entry_point, tests)
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
        }

        queue.put(rollout)
        print(f"[ROLLOUT] Task {task['task_id']} | Reward {reward:.2f} ({passed}/{total})")
        print(generated_code)
        print("-" * 80)

        time.sleep(1)

    queue.put(None)