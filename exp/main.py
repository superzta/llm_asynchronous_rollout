import argparse
import multiprocessing as mp

from buffer import create_queue
from rollout import rollout_worker_fn
from train import trainer_worker_fn
from data_exp import CODING_TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="Toy async/sync coding RL demo")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["async", "sync"],
        default="async",
        help="Run asynchronous queue-based training or synchronous batched training",
    )
    parser.add_argument(
        "--queue_maxsize",
        type=int,
        default=20,
        help="Maximum queue size",
    )
    parser.fadd_argument(
        "--rollout_device",
        type=str,
        default="cuda:0",
        help="Device for rollout worker",
    )
    parser.add_argument(
        "--trainer_device",
        type=str,
        default="cuda:1",
        help="Device for trainer worker",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--rollout_sleep_time",
        type=float,
        default=1.0,
        help="Artificial sleep in rollout worker",
    )
    parser.add_argument(
        "--update_sleep_time",
        type=float,
        default=2.0,
        help="Artificial sleep in trainer worker",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    queue = create_queue(maxsize=args.queue_maxsize)

    rollout_process = mp.Process(
        target=rollout_worker_fn,
        args=(
            queue,
            CODING_TASKS,
            args.rollout_device,
            args.mode,
            args.max_new_tokens,
            args.rollout_sleep_time,
        ),
    )

    trainer_process = mp.Process(
        target=trainer_worker_fn,
        args=(
            queue,
            args.trainer_device,
            args.mode,
            args.update_sleep_time,
        ),
    )

    print(f"[MAIN] Starting demo in {args.mode.upper()} mode")
    print(f"[MAIN] Rollout device: {args.rollout_device}")
    print(f"[MAIN] Trainer device: {args.trainer_device}")

    rollout_process.start()
    trainer_process.start()

    rollout_process.join()
    trainer_process.join()

    print("[MAIN] Coding RL demo finished.")