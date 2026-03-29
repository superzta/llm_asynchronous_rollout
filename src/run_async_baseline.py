import argparse
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from src.coding_reward import evaluate_response
from src.coding_task import CodingTask, build_model_prompt, load_tasks, repeat_tasks
from src.metrics import summarize_async, write_json, write_jsonl
from src.model_backends import build_backend
from src.staleness import bounded_staleness_accept


class SharedPolicyState:
    def __init__(self, version=0):
        self.version = version


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal async rollout+reward+learning baseline with staleness.")
    parser.add_argument("--dataset", default="data/tiny_coding_train.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["dummy", "tiny_policy", "hf_trainable"], default="tiny_policy")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--update-batch-size", type=int, default=4)
    parser.add_argument("--dummy-sleep-sec", type=float, default=0.01)
    parser.add_argument("--reward-timeout-sec", type=float, default=2.0)
    parser.add_argument("--staleness-k", type=int, default=1)
    parser.add_argument("--queue-maxsize", type=int, default=64)
    parser.add_argument("--queue-trace-interval-sec", type=float, default=0.1)
    parser.add_argument("--results-jsonl", default="results/async_train_results.jsonl")
    parser.add_argument("--summary-json", default="results/async_train_summary.json")
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
    sample_queue = queue.Queue(maxsize=args.queue_maxsize)
    train_queue = queue.Queue(maxsize=args.queue_maxsize)
    policy_state = SharedPolicyState(version=0)
    lock = threading.Lock()
    stop_event = threading.Event()
    producer_done = threading.Event()
    consumer_done = threading.Event()

    accepted = []  # type: List[Dict[str, Any]]
    dropped = []  # type: List[Dict[str, Any]]
    all_results = []  # type: List[Dict[str, Any]]
    train_buffer = []  # type: List[Dict[str, Any]]
    queue_depth_trace = []  # type: List[Dict[str, Any]]
    update_history = []  # type: List[Dict[str, Any]]
    reward_by_step = []  # type: List[Dict[str, Any]]
    pass_rate_by_step = []  # type: List[Dict[str, Any]]
    tokens_per_sec_by_step = []  # type: List[Dict[str, Any]]
    learner_stats = {"update_count": 0}

    def get_policy_version():
        # type: () -> int
        with lock:
            return policy_state.version

    def bump_policy_version():
        # type: () -> int
        with lock:
            policy_state.version += 1
            return policy_state.version

    def producer(task_items):
        # type: (List[CodingTask]) -> None
        for step_idx, task in enumerate(task_items, start=1):
            if stop_event.is_set():
                break
            policy_version = get_policy_version()
            prompt = build_model_prompt(task)
            generation = backend.generate(prompt=prompt, task=task)
            sample = {
                "task_id": task.task_id,
                "task": task.to_dict(),
                "prompt": prompt,
                "response": generation.text,
                "response_length": len(generation.text),
                "policy_version": policy_version,
                "generation_latency_sec": generation.latency_sec,
                "tokens_per_sec": generation.tokens_per_sec,
                "logprob_sum": generation.logprob_sum,
                "metadata": generation.metadata,
                "step": step_idx,
                "enqueue_time": time.time(),
            }
            sample_queue.put(sample)
        producer_done.set()

    def consumer():
        while True:
            if producer_done.is_set() and sample_queue.empty():
                break
            try:
                sample = sample_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            dequeue_time = time.time()
            current_policy = get_policy_version()
            decision = bounded_staleness_accept(
                current_policy_version=current_policy,
                sample_policy_version=int(sample["policy_version"]),
                bound_k=args.staleness_k,
            )
            if not decision.accepted:
                dropped_row = {
                    "task_id": sample["task_id"],
                    "policy_version": sample["policy_version"],
                    "current_policy_version": current_policy,
                    "staleness": decision.staleness,
                    "staleness_bound_k": args.staleness_k,
                    "enqueue_time": sample["enqueue_time"],
                    "dequeue_time": dequeue_time,
                    "latency_sec": sample["generation_latency_sec"],
                    "tokens_per_sec": sample["tokens_per_sec"],
                    "reward": 0.0,
                    "pass": False,
                    "num_passed": 0,
                    "total_tests": len(sample["task"]["reference_tests"]),
                    "error_type": "dropped_stale",
                    "dropped": True,
                }
                dropped.append(dropped_row)
                all_results.append(dropped_row)
                sample_queue.task_done()
                continue

            reward = evaluate_response(
                response_text=sample["response"],
                task=sample["task"],
                timeout_sec=args.reward_timeout_sec,
            )
            row = {
                "task_id": sample["task_id"],
                "prompt": sample["prompt"],
                "response": sample["response"],
                "response_length": sample["response_length"],
                "policy_version": sample["policy_version"],
                "current_policy_version": current_policy,
                "staleness": decision.staleness,
                "staleness_bound_k": args.staleness_k,
                "enqueue_time": sample["enqueue_time"],
                "dequeue_time": dequeue_time,
                "latency_sec": dequeue_time - sample["enqueue_time"],
                "generation_latency_sec": sample["generation_latency_sec"],
                "tokens_per_sec": sample["tokens_per_sec"],
                "logprob_sum": sample["logprob_sum"],
                "reward": float(reward["reward"]),
                "pass": bool(reward["pass"]),
                "num_passed": int(reward["num_passed"]),
                "total_tests": int(reward["total_tests"]),
                "error_type": reward.get("error_type"),
                "dropped": False,
                "updated_by_learner": False,
                "update_count": learner_stats["update_count"],
                "step": sample["step"],
            }
            accepted.append(row)
            all_results.append(row)
            train_buffer.append(row)
            reward_by_step.append({"step": sample["step"], "reward": row["reward"]})
            pass_rate_by_step.append({"step": sample["step"], "pass_rate": 1.0 if row["pass"] else 0.0})
            tokens_per_sec_by_step.append({"step": sample["step"], "tokens_per_sec": row["tokens_per_sec"]})
            train_queue.put(
                {
                    "task_id": sample["task_id"],
                    "prompt": sample["prompt"],
                    "response": sample["response"],
                    "reward": float(reward["reward"]),
                    "metadata": sample["metadata"],
                    "row_ref": row,
                }
            )
            sample_queue.task_done()
        consumer_done.set()

    def learner():
        pending = []
        while True:
            if consumer_done.is_set() and train_queue.empty():
                break
            try:
                item = train_queue.get(timeout=0.1)
                pending.append(item)
                train_queue.task_done()
            except queue.Empty:
                pass

            if len(pending) < args.update_batch_size:
                continue
            train_samples = [
                {
                    "task_id": x["task_id"],
                    "prompt": x["prompt"],
                    "response": x["response"],
                    "reward": x["reward"],
                    "metadata": x["metadata"],
                }
                for x in pending
            ]
            update_info = backend.policy_gradient_update(train_samples)
            if update_info.get("updated", False):
                learner_stats["update_count"] += 1
                new_version = bump_policy_version()
                update_history.append(
                    {
                        "update_id": learner_stats["update_count"],
                        "policy_version": new_version,
                        "loss": float(update_info.get("loss", 0.0)),
                        "avg_reward": float(update_info.get("avg_reward", 0.0)),
                        "batch_size": int(update_info.get("batch_size", len(pending))),
                    }
                )
                for x in pending:
                    x["row_ref"]["updated_by_learner"] = True
                    x["row_ref"]["update_count"] = learner_stats["update_count"]
            pending = []

        if pending:
            train_samples = [
                {
                    "task_id": x["task_id"],
                    "prompt": x["prompt"],
                    "response": x["response"],
                    "reward": x["reward"],
                    "metadata": x["metadata"],
                }
                for x in pending
            ]
            update_info = backend.policy_gradient_update(train_samples)
            if update_info.get("updated", False):
                learner_stats["update_count"] += 1
                new_version = bump_policy_version()
                update_history.append(
                    {
                        "update_id": learner_stats["update_count"],
                        "policy_version": new_version,
                        "loss": float(update_info.get("loss", 0.0)),
                        "avg_reward": float(update_info.get("avg_reward", 0.0)),
                        "batch_size": int(update_info.get("batch_size", len(pending))),
                    }
                )
                for x in pending:
                    x["row_ref"]["updated_by_learner"] = True
                    x["row_ref"]["update_count"] = learner_stats["update_count"]

    def queue_monitor(start_time):
        # type: (float) -> None
        while not stop_event.is_set():
            queue_depth_trace.append(
                {
                    "t_sec": time.time() - start_time,
                    "sample_queue_depth": sample_queue.qsize(),
                    "train_queue_depth": train_queue.qsize(),
                }
            )
            time.sleep(args.queue_trace_interval_sec)
            if producer_done.is_set() and consumer_done.is_set() and sample_queue.empty() and train_queue.empty():
                queue_depth_trace.append(
                    {
                        "t_sec": time.time() - start_time,
                        "sample_queue_depth": sample_queue.qsize(),
                        "train_queue_depth": train_queue.qsize(),
                    }
                )
                break

    wall_start = time.time()
    producer_thread = threading.Thread(target=producer, args=(tasks,), daemon=True)
    consumer_thread = threading.Thread(target=consumer, daemon=True)
    learner_thread = threading.Thread(target=learner, daemon=True)
    monitor_thread = threading.Thread(target=queue_monitor, args=(wall_start,), daemon=True)

    producer_thread.start()
    consumer_thread.start()
    learner_thread.start()
    monitor_thread.start()

    producer_thread.join()
    consumer_thread.join()
    learner_thread.join()
    stop_event.set()
    monitor_thread.join(timeout=1.0)

    wall_clock_sec = time.time() - wall_start
    summary = summarize_async(
        accepted_results=accepted,
        dropped_results=dropped,
        wall_clock_sec=wall_clock_sec,
        queue_depth_trace=queue_depth_trace,
    )
    summary["mode"] = "async_train"
    summary["backend"] = args.backend
    summary["seed"] = args.seed
    summary["lr"] = args.lr
    summary["update_batch_size"] = args.update_batch_size
    summary["update_count"] = learner_stats["update_count"]
    summary["final_policy_version"] = get_policy_version()
    summary["train_buffer_size"] = len(train_buffer)
    summary["dropped_stale_count"] = len(dropped)
    summary["reward_by_step"] = reward_by_step
    summary["pass_rate_by_step"] = pass_rate_by_step
    summary["tokens_per_sec_by_step"] = tokens_per_sec_by_step
    summary["avg_tokens_per_sec"] = (
        sum([x["tokens_per_sec"] for x in tokens_per_sec_by_step]) / max(len(tokens_per_sec_by_step), 1)
    )
    summary["update_history"] = update_history
    write_jsonl(args.results_jsonl, all_results)
    write_json(args.summary_json, summary)
    return summary


def main():
    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    summary = run(args)
    print("Async training baseline finished.")
    print(summary)


if __name__ == "__main__":
    main()
