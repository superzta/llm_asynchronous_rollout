import multiprocessing as mp
from queue_buffer import create_queue
from rollout_worker import rollout_worker_fn
from trainer_worker import trainer_worker_fn
from coding_tasks import CODING_TASKS

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    queue = create_queue(maxsize=20)

    rollout_process = mp.Process(
        target=rollout_worker_fn,
        args=(queue, CODING_TASKS, "cuda:0")
    )

    trainer_process = mp.Process(
        target=trainer_worker_fn,
        args=(queue, "cuda:1")
    )

    rollout_process.start()
    trainer_process.start()

    rollout_process.join()
    trainer_process.join()

    print("Async coding RL demo finished.")