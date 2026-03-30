You are working in llm_async_project on PSC.

Current status:
- We already have a lightweight sync training baseline.
- We already have a lightweight async training baseline with staleness filtering.
- We already have a minimal real-learning path.
- But the current async path is still not AReaL-style: it is not a true multi-process multi-role architecture with interruptible rollout workers, controller, replay buffer, and parameter service.

Goal:
Implement a new AReaL-style async runner that is still runnable on PSC and supports:
- N rollout workers
- M trainer workers
- explicit rollout devices
- explicit trainer devices
- interruptible rollout workers
- replay buffer / rollout controller
- parameter service with global policy version
- bounded staleness filtering

Important:
- Do NOT delete or replace the existing minimal runners.
- Add a new path in parallel, for example:
- src/run_async_areal_style.py
- Keep the code PSC-friendly and runnable on 2 GPUs by default.
- Do NOT add FlashAttention, Apex, Transformer Engine, Ray, DeepSpeed, or heavyweight distributed stack.
- Use only Python multiprocessing or torch.multiprocessing.
- Keep the tiny benchmark and reward harness unchanged.
- Preserve current scripts and results; add new scripts/files cleanly.

Architecture to implement:
- Parameter Service:
- stores latest global trainable state
- stores global policy_version
- applies trainer updates serially
- increments policy_version only after a real update is committed
- Rollout Controller:
- dispatches prompts/tasks to rollout workers
- receives generated trajectories
- sends rewards to/from reward service if needed
- computes current staleness using global policy_version
- drops stale samples when staleness > K
- aggregates accepted samples into replay buffer
- Replay Buffer:
- stores accepted samples for training
- serves full batches to trainer workers
- Rollout Workers:
- each owns its own local backend replica on assigned device
- each tracks currently loaded policy_version
- each can be interrupted when new weights are available
- each refreshes weights from parameter service after interrupt
- Trainer Workers:
- each owns its own local backend replica on assigned device
- each consumes batches from replay buffer
- each computes a real update
- each sends update result / updated state to parameter service

Interruptible rollout behavior:
This is required.
Implement rollout workers so they can receive an interrupt signal when parameter service version increases.
The rollout worker should:
- periodically check for interrupt requests while generating
- stop current generation at a safe boundary
- mark the sample as interrupted
- reload latest weights/state from parameter service
- continue with the next queued prompt using the new version

Safe boundary definition:
- For tiny_policy: interruption may happen before final sample emit or between task/sample steps.
- For hf_trainable: implement chunked generation or token-block generation with periodic polling every few tokens, so interruption is possible mid-generation in a lightweight way.
- If true token-level interruption is too invasive, implement chunk-level interruption and document it clearly.

This interruptibility is important because we are reproducing the AReaL-style idea, not just generic async queues.

New files to create:
- src/run_async_areal_style.py
- src/areal_runtime.py
- src/areal_parameter_service.py
- src/areal_rollout_worker.py
- src/areal_trainer_worker.py
- src/areal_controller.py
- scripts/run_async_areal_style.sh
- scripts/run_async_areal_style_psc_2gpu.sh
- README_async_areal_style.md

You may combine some of the Python files if that is cleaner, but keep the code modular by role.

Required CLI args for src/run_async_areal_style.py:
- --num-rollout-workers
- --num-trainer-workers
- --rollout-devices
- --trainer-devices
- --backend
- --hf-model-id
- --epochs
- --seed
- --lr
- --update-batch-size
- --staleness-k
- --queue-maxsize
- --producer-delay-sec
- --learner-delay-sec
- --interrupt-check-interval-sec
- --generation-chunk-size
- --results-jsonl
- --summary-json

Worker/device behavior:
- rollout-devices is a comma-separated list, e.g. cuda:0,cuda:0
- trainer-devices is a comma-separated list, e.g. cuda:1,cuda:1
- if device list is shorter than worker count, assign round-robin
- if CUDA is unavailable, allow CPU fallback
- PSC default intended topology:
- 1 rollout worker on cuda:0
- 1 trainer worker on cuda:1

Backend requirements:
Extend model_backends.py so trainable backends support:
- get_trainable_state()
- load_trainable_state(state)
- clone_for_device(device)
- maybe_generate_chunk(...) or equivalent helper for interruptible generation if needed

These methods must work for:
- tiny_policy
- hf_trainable

For tiny_policy:
- serialize task logits cleanly
- allow state sync across processes

For hf_trainable:
- use state_dict transfer
- keep model tiny and PSC-friendly
- no heavy optimizations

Interrupt signal design:
- parameter service sets a newer policy_version after update
- rollout controller or parameter service can notify rollout workers
- rollout workers must observe the new version and interrupt safely
- track:
- interrupt_count per rollout worker
- last_loaded_policy_version per rollout worker
- interrupted_samples count

Replay / training logic:
- controller receives rollout samples
- controller computes staleness = current_policy_version - sample.policy_version
- controller drops sample if staleness > K
- accepted samples go to replay buffer
- replay buffer sends full batches to trainer workers
- trainer workers update local model
- trainer sends update result to parameter service
- parameter service applies update, increments version, and exposes latest state

It is acceptable for trainer updates to be serialized through the parameter service.
We do NOT need distributed gradient averaging in this step.

Metrics and outputs:
Write:
- results/async_areal_results.jsonl
- results/async_areal_summary.json

Summary must include:
- num_rollout_workers
- num_trainer_workers
- rollout_devices
- trainer_devices
- update_count
- final_policy_version
- num_total_seen
- num_accepted
- num_dropped
- accepted_fraction
- dropped_fraction
- mean_staleness
- max_staleness
- avg_reward
- pass_rate
- wall_clock_sec
- avg_tokens_per_sec
- queue depth trace
- replay buffer size trace
- per-rollout-worker generated count
- per-rollout-worker interrupt_count
- per-rollout-worker last_loaded_policy_version
- per-trainer-worker update_count
- interrupted_samples count
- config block

Results rows should also include enough metadata to debug:
- task_id
- rollout_worker_id
- trainer_worker_id if trained
- policy_version used for generation
- current global policy_version when consumed
- staleness
- interrupted flag
- dropped flag
- reward
- pass

New scripts:
1) scripts/run_async_areal_style.sh
- default CPU smoke test
- backend tiny_policy
- small worker counts

2) scripts/run_async_areal_style_psc_2gpu.sh
- intended PSC default
- num_rollout_workers = 1
- num_trainer_workers = 1
- rollout_devices = cuda:0
- trainer_devices = cuda:1
- backend tiny_policy by default
- keep it runnable

Acceptance criteria:
A. CPU smoke test works:
bash scripts/run_async_areal_style.sh

B. PSC 2-GPU topology works:
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_async_areal_style_psc_2gpu.sh

C. Parameterized worker-count run works:
python3 -m src.run_async_areal_style --backend tiny_policy --num-rollout-workers 2 --num-trainer-workers 1 --rollout-devices cpu,cpu --trainer-devices cpu --staleness-k 1 --epochs 2

D. Interruptibility is real:
- policy updates cause rollout workers to observe newer version
- rollout workers can stop at safe boundary
- rollout workers reload weights
- interrupt_count becomes nonzero in at least one stress test when producer/learner timing is configured to induce overlap

Add one stress-test mode or script to make interruptibility observable, for example by:
- slowing learner slightly
- using chunked generation
- using enough tasks/epochs to overlap rollout and training

Documentation:
Create README_async_areal_style.md explaining:
- how this differs from previous threaded async baseline
- controller / replay buffer / parameter service / rollout workers / trainer workers
- interruptible rollout behavior
- safe interruption boundary
- PSC 2-GPU usage
- CPU fallback
- limitations relative to full AReaL and full PPO

Scope control:
- Do NOT implement full PPO, GAE, value heads, distributed optimizer, or Slime internals in this step.
- We only want an AReaL-style systems topology approximation that is runnable and clearly closer to the paper architecture.

At the end provide:
- all created/modified files
- exact commands to run CPU smoke test
- exact commands to run PSC 2-GPU test
- exact commands to run an interruptibility stress test
- short explanation of how interrupt signals, weight refresh, and worker/device split now work