import argparse
import random
from pathlib import Path

from src.areal_runtime import run_areal_style
from src.metrics import write_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="AReaL-style async runner (lightweight multiprocessing).")
    parser.add_argument("--dataset", default="data/tiny_coding_train.jsonl")
    parser.add_argument("--num-rollout-workers", type=int, default=1)
    parser.add_argument("--num-trainer-workers", type=int, default=1)
    parser.add_argument("--rollout-devices", default="cpu")
    parser.add_argument("--trainer-devices", default="cpu")
    parser.add_argument("--backend", choices=["dummy", "tiny_policy", "hf_trainable"], default="tiny_policy")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--update-batch-size", type=int, default=4)
    parser.add_argument("--staleness-k", type=int, default=1)
    parser.add_argument("--queue-maxsize", type=int, default=128)
    parser.add_argument("--queue-trace-interval-sec", type=float, default=0.1)
    parser.add_argument("--producer-delay-sec", type=float, default=0.0)
    parser.add_argument("--learner-delay-sec", type=float, default=0.0)
    parser.add_argument("--interrupt-check-interval-sec", type=float, default=0.02)
    parser.add_argument("--generation-chunk-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy-sleep-sec", type=float, default=0.01)
    parser.add_argument("--reward-timeout-sec", type=float, default=2.0)
    parser.add_argument("--results-jsonl", default="results/async_areal_results.jsonl")
    parser.add_argument("--summary-json", default="results/async_areal_summary.json")
    return parser.parse_args()


def _set_seed(seed):
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main():
    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    _set_seed(args.seed)
    out = run_areal_style(args)
    write_jsonl(args.results_jsonl, out["rows"])
    write_json(args.summary_json, out["summary"])
    print("AReaL-style async run finished.")
    print(out["summary"])


if __name__ == "__main__":
    main()
