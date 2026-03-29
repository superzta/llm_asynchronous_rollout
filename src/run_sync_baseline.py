import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from src.coding_reward import evaluate_response
from src.coding_task import build_model_prompt, load_tasks, repeat_tasks
from src.metrics import summarize_sync, write_json, write_jsonl
from src.model_backends import build_backend


class NoOpTrainer:
    """
    Placeholder to preserve a clean insertion point for PPO updates later.
    """

    def maybe_train_step(self, sample_row):
        # type: (Dict[str, Any]) -> None
        _ = sample_row
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal synchronous coding rollout+reward baseline.")
    parser.add_argument(
        "--dataset",
        default="data/tiny_coding_train.jsonl",
        help="Path to JSONL coding task file.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="How many passes over dataset.")
    parser.add_argument("--backend", choices=["dummy", "hf"], default="dummy")
    parser.add_argument("--hf-model-id", default="", help="HF model id for --backend hf.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dummy-sleep-sec", type=float, default=0.01)
    parser.add_argument("--reward-timeout-sec", type=float, default=2.0)
    parser.add_argument("--results-jsonl", default="results/sync_results.jsonl")
    parser.add_argument("--summary-json", default="results/sync_summary.json")
    return parser.parse_args()


def run(args):
    # type: (argparse.Namespace) -> Dict[str, Any]
    tasks = repeat_tasks(load_tasks(args.dataset), args.epochs)
    backend = build_backend(args)
    trainer = NoOpTrainer()
    results = []  # type: List[Dict[str, Any]]

    wall_start = time.time()
    for task in tasks:
        prompt = build_model_prompt(task)
        generation = backend.generate(prompt=prompt, task=task)
        reward = evaluate_response(
            response_text=generation.text,
            task=task,
            timeout_sec=args.reward_timeout_sec,
        )
        row = {
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
        }
        trainer.maybe_train_step(row)
        results.append(row)

    wall_clock_sec = time.time() - wall_start
    summary = summarize_sync(results=results, wall_clock_sec=wall_clock_sec)
    write_jsonl(args.results_jsonl, results)
    write_json(args.summary_json, summary)
    return summary


def main():
    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    summary = run(args)
    print("Sync baseline finished.")
    print(summary)


if __name__ == "__main__":
    main()
