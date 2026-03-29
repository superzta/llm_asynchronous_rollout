import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from src.coding_reward import evaluate_response
from src.coding_task import build_model_prompt, load_tasks, repeat_tasks
from src.metrics import summarize_sync, write_json, write_jsonl
from src.model_backends import build_backend


class PolicyGradientTrainer:
    """
    Minimal trainer wrapper around backend policy-gradient updates.
    This keeps a clean seam for future PPO replacement.
    """

    def __init__(self, backend, update_batch_size):
        self.backend = backend
        self.update_batch_size = update_batch_size
        self.pending = []  # type: List[Dict[str, Any]]
        self.update_count = 0

    def add_sample_and_maybe_update(self, sample):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        self.pending.append(sample)
        if len(self.pending) < self.update_batch_size:
            return {"updated": False, "loss": 0.0}
        update_info = self.backend.policy_gradient_update(self.pending)
        if update_info.get("updated", False):
            self.update_count += 1
        self.pending = []
        return update_info

    def finalize(self):
        # type: () -> Dict[str, Any]
        if not self.pending:
            return {"updated": False, "loss": 0.0}
        update_info = self.backend.policy_gradient_update(self.pending)
        if update_info.get("updated", False):
            self.update_count += 1
        self.pending = []
        return update_info


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal synchronous coding training baseline.")
    parser.add_argument(
        "--dataset",
        default="data/tiny_coding_train.jsonl",
        help="Path to JSONL coding task file.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="How many passes over dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["dummy", "tiny_policy", "hf_trainable"], default="tiny_policy")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2", help="HF model id for trainable HF backend.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--update-batch-size", type=int, default=4)
    parser.add_argument("--dummy-sleep-sec", type=float, default=0.01)
    parser.add_argument("--reward-timeout-sec", type=float, default=2.0)
    parser.add_argument("--results-jsonl", default="results/sync_train_results.jsonl")
    parser.add_argument("--summary-json", default="results/sync_train_summary.json")
    return parser.parse_args()


def _set_seed(seed):
    # type: (int) -> None
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def run(args):
    # type: (argparse.Namespace) -> Dict[str, Any]
    _set_seed(args.seed)
    tasks = repeat_tasks(load_tasks(args.dataset), args.epochs)
    backend = build_backend(args)
    trainer = PolicyGradientTrainer(backend=backend, update_batch_size=args.update_batch_size)
    results = []  # type: List[Dict[str, Any]]
    policy_version = 0
    reward_by_step = []  # type: List[Dict[str, Any]]
    pass_rate_by_step = []  # type: List[Dict[str, Any]]
    tokens_per_sec_by_step = []  # type: List[Dict[str, Any]]

    wall_start = time.time()
    for step_idx, task in enumerate(tasks, start=1):
        sample_policy_version = policy_version
        prompt = build_model_prompt(task)
        generation = backend.generate(prompt=prompt, task=task)
        reward = evaluate_response(
            response_text=generation.text,
            task=task,
            timeout_sec=args.reward_timeout_sec,
        )
        train_sample = {
            "task_id": task.task_id,
            "prompt": prompt,
            "response": generation.text,
            "reward": float(reward["reward"]),
            "metadata": generation.metadata,
            "token_logprobs": generation.token_logprobs,
        }
        update_info = trainer.add_sample_and_maybe_update(train_sample)
        if update_info.get("updated", False):
            policy_version += 1
        row = {
            "step": step_idx,
            "task_id": task.task_id,
            "prompt": prompt,
            "response": generation.text,
            "response_length": len(generation.text),
            "reward": float(reward["reward"]),
            "pass": bool(reward["pass"]),
            "num_passed": int(reward["num_passed"]),
            "total_tests": int(reward["total_tests"]),
            "error_type": reward.get("error_type"),
            "latency_sec": generation.latency_sec,
            "tokens_per_sec": generation.tokens_per_sec,
            "policy_version": sample_policy_version,
            "policy_version_after_step": policy_version,
            "logprob_sum": generation.logprob_sum,
            "updated_this_step": bool(update_info.get("updated", False)),
            "update_loss": float(update_info.get("loss", 0.0)),
            "update_count": trainer.update_count,
        }
        reward_by_step.append({"step": step_idx, "reward": row["reward"]})
        pass_rate_by_step.append({"step": step_idx, "pass_rate": 1.0 if row["pass"] else 0.0})
        tokens_per_sec_by_step.append({"step": step_idx, "tokens_per_sec": row["tokens_per_sec"]})
        results.append(row)

    final_update = trainer.finalize()
    if final_update.get("updated", False):
        policy_version += 1

    wall_clock_sec = time.time() - wall_start
    summary = summarize_sync(results=results, wall_clock_sec=wall_clock_sec)
    summary["mode"] = "sync_train"
    summary["update_count"] = trainer.update_count
    summary["final_policy_version"] = policy_version
    summary["seed"] = args.seed
    summary["backend"] = args.backend
    summary["lr"] = args.lr
    summary["update_batch_size"] = args.update_batch_size
    summary["reward_by_step"] = reward_by_step
    summary["pass_rate_by_step"] = pass_rate_by_step
    summary["tokens_per_sec_by_step"] = tokens_per_sec_by_step
    summary["avg_tokens_per_sec"] = (
        sum([x["tokens_per_sec"] for x in tokens_per_sec_by_step]) / max(len(tokens_per_sec_by_step), 1)
    )
    write_jsonl(args.results_jsonl, results)
    write_json(args.summary_json, summary)
    return summary


def main():
    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    summary = run(args)
    print("Sync training baseline finished.")
    print(summary)


if __name__ == "__main__":
    main()
