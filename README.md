# `llm_asynchronous_rollout`: A Lightweight AReaL-Style Async RL System for LLM Reasoning

[Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · PPO/GRPO · Bounded staleness · 2× V100 32 GB

A from-scratch implementation of the asynchronous reinforcement-learning system described in *"AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"* (Fu et al., 2025), scaled down so the full pipeline fits on two V100-32 GB GPUs and a short PSC interactive session.

The project re-implements the core systems ideas of AReaL — **decoupled rollout and training processes, a parameter-server policy-version clock, a bounded-staleness replay buffer, interruptible rollouts, and the decoupled PPO/GRPO objective** — and uses them to reproduce the paper's key qualitative findings (Fig 5a, 5b, 5c, 6b) on a tractable benchmark.

It is also a self-contained research harness: one command runs the full sweep, regenerates every figure and table in the report/poster, and supports crash-resumption.

---

## Table of contents

1. [What this repo is](#what-this-repo-is)
2. [Results at a glance](#results-at-a-glance)
3. [System architecture](#system-architecture)
4. [The three runners](#the-three-runners)
5. [Repository layout](#repository-layout)
6. [Installation](#installation)
7. [Data](#data)
8. [Quickstart](#quickstart)
9. [Reproducing the full final report](#reproducing-the-full-final-report)
10. [Resuming an interrupted run](#resuming-an-interrupted-run)
11. [Outputs and figures](#outputs-and-figures)
12. [Key configuration knobs](#key-configuration-knobs)
13. [Design notes and lessons learned](#design-notes-and-lessons-learned)
14. [Limitations](#limitations)
15. [Acknowledgements](#acknowledgements)

---

## What this repo is

This project answers three linked questions:

1. **Systems:** can we build an AReaL-style decoupled async RL pipeline on commodity (2× V100) hardware and still observe the async throughput advantage?
2. **Algorithmic:** does the AReaL "decoupled objective" (Fig 5b) actually stabilize learning under staleness when the model is a real 0.5 B-parameter LLM, not a toy?
3. **Reproducibility:** can a single driver script regenerate every plot and table a short paper/poster needs, with resume-from-crash support?

Concretely the repo provides:

- A faithful lightweight re-implementation of the **AReaL controller / parameter-service / rollout-worker / trainer-worker topology** in ~1 k LOC of pure-Python `multiprocessing` (`src/areal_*.py`).
- A real **GRPO/PPO update** (group-normalized advantage, clipped surrogate objective, Schulman k3 KL, decoupled-objective ablation switch, gradient clipping, token-length normalization) wrapping Hugging Face Transformers (`src/model_backends.py::TrainableHFCausalLMBackend`).
- **Three runners** that share reward harness, tokenizer, and logging so sync / threaded-async / AReaL-style are apples-to-apples:
  - `src.run_sync_baseline` — synchronous PPO/GRPO.
  - `src.run_async_baseline` — single-process threaded async baseline (producer / consumer + bounded staleness).
  - `src.run_async_areal_style` — multi-process AReaL topology with a policy-version clock, an interruptible rollout worker, and a serial parameter server.
- A **grid-sweep driver** (`src.run_experiment_grid`) that fans each runner out across seeds × staleness `K` × decoupled-on/off × datasets, with `--skip-existing` crash resume.
- **One mega pipeline script** (`scripts/run_final_report.sh` + `scripts/submit_final_report.sbatch`) that runs the sweep on GSM8K and a tiny coding benchmark and produces every figure + table a short paper/poster needs.
- **Two plotting layers**: an AReaL-paper reproduction plotter (`src.plot_paper_repro`: Fig 5a / 5b / 5c / 6b + Table 2) and a richer final-report plotter (`src.plot_final_report`: Pareto, time-to-threshold, realized-staleness violins, sample efficiency, trainer throughput, poster overview panel).

---

## Results at a glance

With 2× V100-32 GB, Qwen 2.5-0.5B-Instruct, 128 GSM8K training prompts, 3 seeds, staleness `K ∈ {0, 2, 4, 8}`, and decoupled ∈ {0, 1}:

| Comparison | Take-away |
|---|---|
| **`fig_pass_rate_curves.png`** | async_areal_style matches sync_train in final pass rate and beats it in wall-clock. |
| **`fig_pareto.png`** | Pareto frontier over accuracy vs. throughput is dominated by async_areal_style with decoupled=1. |
| **`fig_time_to_threshold.png`** | async_areal_style reaches pass=0.10 in the fewest seconds. |
| **`fig5b_decoupled_learning_curves.png`** | Reproduces AReaL Fig 5b: decoupled=1 tolerates larger `K`; decoupled=0 degrades as `K` grows. |
| **`fig5c_throughput_vs_staleness.png`** | Reproduces Fig 5c: updates/sec rises then plateaus with `K`. |
| **`fig_staleness_violin.png`** | Realized staleness distribution widens with `K`, confirming the bound is active. |

The exact, published numbers are in `results/experiments/<EXP_NAME>/<dataset>/tables/summary_headline.md` and `mode_aggregate.md` produced by the driver.

---

## System architecture

```
                          ┌─────────────────────────────┐
                          │ controller (main process)   │
                          │  - dispatches prompts       │
                          │  - computes staleness       │
                          │  - filters on K bound       │
                          │  - batches replay buffer    │
                          └──────┬────────────┬─────────┘
                                 │            │
                      rollout_task_queue  trainer_input_queue
                                 │            │
   ┌─────────────────────┐       ▼            ▼     ┌─────────────────────┐
   │  parameter_service  │  ┌──────────┐  ┌──────────┐  │  parameter_service │
   │  (serial committer) │◄─┤ rollout  │  │ trainer  ├─►│   policy_version   │
   │   policy_version++  │  │ worker 0 │  │ worker 0 │  │      counter       │
   └─────────┬───────────┘  │ (cuda:0) │  │ (cuda:1) │  └─────────┬──────────┘
             │              └──────────┘  └──────────┘            │
             └────────────── shared_state (mp.Manager) ───────────┘
```

Key systems ingredients, all implemented in pure-python `multiprocessing` with the `spawn` start-method (so each worker initializes its own CUDA context on its assigned GPU — critical for stability):

- **Parameter service** (`src/areal_parameter_service.py`): the *only* writer of `shared_state["trainable_state"]`. Applies trainer commits serially and monotonically increments `policy_version`.
- **Controller** (`src/areal_controller.py`): dispatches tasks, runs the reward harness, computes `staleness = current_policy_version − sample.policy_version`, drops samples with `staleness > K`, and batches the replay buffer into `trainer_input_queues`.
- **Rollout worker** (`src/areal_rollout_worker.py`): maintains a local model replica on `cuda:0`, polls `policy_version` *during* chunked generation, interrupts at safe chunk boundaries when it sees a newer version, reloads weights from `shared_state`, and resumes.
- **Trainer worker** (`src/areal_trainer_worker.py`): consumes replay batches on `cuda:1`, runs a **real** GRPO/PPO step (see below), pushes the new state through the parameter service.
- **Worker-ready barrier** (new): before the controller starts its wall-clock, the driver blocks until every worker signals that its backend is on-device. Cold-start model-load time is reported separately as `setup_wall_clock_sec` so `wall_clock_sec` measures exactly the async rollout+train phase and is comparable across warm/cold runs.
- **Event trace**: every meaningful event (worker ready, version load, interrupt trigger/observe, sample emit, commit, dispatch) is pushed to a shared `event_queue` and persisted in `summary.json` so downstream plotters can reconstruct the timeline.

### The GRPO/PPO update

`TrainableHFCausalLMBackend.policy_gradient_update` implements:

- **Group-normalized advantage** (GRPO): `A = (R − mean_group(R)) / (std_group(R) + ε)`.
- **Clipped surrogate objective**: `min(r·A, clip(r, 1−ε, 1+ε)·A)` with `ε = --grpo-epsilon` (default 0.2).
- **KL penalty**: Schulman k3 estimator `exp(log r) − 1 − log r ≥ 0`, weighted by `--grpo-kl-coef` (default 0.02). This is numerically stable and correctly *penalizes* drift from the sampling policy — an earlier signed-difference implementation would invert into an incentive and silently regressed decoupled=1 runs. Fixed; see [Design notes](#design-notes-and-lessons-learned).
- **Token-length normalization** and **gradient clipping** (`--grad-clip`, default 1.0).
- **Decoupled-objective flag** (`--decoupled-objective 0|1`): when 1, the clipped surrogate uses the *rollout* policy's log-probs as the `π_old` reference (AReaL Fig 5b). When 0, it uses the live trainer policy, matching standard on-policy PPO semantics. This switch is the core ablation of the project.
- **FP16 autocast** (`HF_USE_AUTOCAST=1`) around `generate()` and `_recompute_new_logprobs()` for V100 throughput, with NaN/Inf guards on logits (`_NanInfGuard`).

---

## The three runners

All three runners share `src/coding_task.py` (prompt builder, JSONL loader), `src/coding_reward.py` + `src/gsm8k_reward.py` (reward harness), `src/metrics.py` (summary writer), and `src/model_backends.py` (model interface), so they only differ in *how they dispatch rollouts and updates*.

| Runner | Process model | Update flow | Staleness source |
|---|---|---|---|
| `src.run_sync_baseline` | 1 process, 1 GPU | alternating generate → reward → update | none (always fresh) |
| `src.run_async_baseline` | 1 process, threaded | producer thread + consumer thread + learner thread, one queue | simulated (producer tags samples with policy version) |
| `src.run_async_areal_style` | N+M+1 processes, 2 GPUs | parameter service + rollout worker(s) + trainer worker(s) | **real**: measured between the sample's rollout-time version and the trainer's current version |

Each writes a `summary.json` with identical core fields (reward, pass rate, tokens/sec, `policy_version`, `wall_clock_sec`, `setup_wall_clock_sec`, staleness histogram, queue depth trace, event trace, config) so cross-runner comparison is direct.

---

## Repository layout

```
llm_asynchronous_rollout/
├── README.md                              ← you are here
├── data/
│   ├── gsm8k_train.jsonl, gsm8k_eval.jsonl        GSM8K (auto-downloaded on first run)
│   └── tiny_coding_train.jsonl, ..._eval.jsonl    12 deterministic coding tasks
├── src/
│   ├── coding_task.py, coding_reward.py           task schema, sandboxed test runner
│   ├── gsm8k_data.py, gsm8k_reward.py             dataset loader, numeric-answer reward
│   ├── model_backends.py                          DummyBackend, TinyPolicyBackend, TrainableHFCausalLMBackend
│   ├── staleness.py, metrics.py                   bounded-staleness helper, summary writer
│   ├── progress.py                                heartbeat / live progress reporter
│   │
│   ├── areal_parameter_service.py                 AReaL: parameter server
│   ├── areal_controller.py                        AReaL: dispatch / reward / staleness / replay
│   ├── areal_rollout_worker.py                    AReaL: interruptible rollout process
│   ├── areal_trainer_worker.py                    AReaL: PPO/GRPO trainer process
│   ├── areal_runtime.py                           AReaL: top-level orchestrator (worker-ready barrier lives here)
│   │
│   ├── run_sync_baseline.py                       runner 1 (synchronous)
│   ├── run_async_baseline.py                      runner 2 (threaded async)
│   ├── run_async_areal_style.py                   runner 3 (AReaL-style multi-process)
│   ├── run_experiment_grid.py                     grid sweep driver with --skip-existing
│   │
│   ├── analyze_experiments.py                     legacy per-run CSV/JSON aggregator
│   ├── plot_paper_repro.py                        AReaL Fig 5a / 5b / 5c / 6b + Table 2
│   └── plot_final_report.py                       richer report/poster plots + tables
├── scripts/
│   ├── run_sync.sh, run_async.sh                  single-runner smoke
│   ├── run_sync_train.sh, run_async_train.sh      training-enabled smoke
│   ├── run_async_areal_style.sh                   AReaL CPU smoke
│   ├── run_async_areal_style_psc_2gpu.sh          AReaL 2× V100 topology
│   ├── run_async_areal_style_debug.sh             single strongly-instrumented config
│   ├── run_async_areal_style_stress.sh            interruptibility stress test
│   ├── run_experiments.sh                         default grid sweep
│   ├── run_experiments_3way_pressure.sh           sync vs async vs AReaL under pressure
│   ├── run_experiments_stress.sh                  AReaL-only stress sweep
│   ├── run_paper_repro.sh                         Fig 5a/5b/5c/6b reproduction
│   ├── run_real_comparison.sh                     real-model comparison
│   ├── run_final_report.sh                        ★ mega driver (all figures, all tables)
│   ├── submit_final_report.sbatch                 ★ Slurm wrapper for the mega driver
│   └── submit_paper_repro.sbatch, submit_real_comparison.sbatch
├── tests/
│   └── test_coding_reward.py                      code-extraction / timeout / syntax tests
├── exp/                                           experimental scratch (buffer.py, train.py, …)
├── slime_examples/psc_async_project/              lightweight Slime adapter scaffold
├── results/experiments/<EXP_NAME>/                all sweep artifacts (see Outputs section)
└── logs/                                          Slurm stdout/stderr + tee'd interactive logs
```

---

## Installation

The project is developed and tested on the **PSC Bridges-2 GPU-shared partition** with two Tesla V100-SXM2-32 GB (`sm_70`). V100 has no native BF16, so FP32 weights with optional FP16 autocast is the tested configuration. FlashAttention-2 is *not* used.

```bash
# From a clean Python 3.12 conda/mamba env:
pip install -r exp/requirements.txt
```

Pinned minima (see `exp/requirements.txt`):

- `torch>=2.2.0`, `transformers>=4.40.0`, `accelerate>=0.30.0`, `datasets>=2.19.0`
- `numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10`, `matplotlib>=3.7`, `tqdm>=4.66`

PSC-specific: a pre-built env is assumed at `$WORKSPACE/conda_envs/slime312` and activated by `$WORKSPACE/psc_slime_env.sh` (see `submit_final_report.sbatch`). Point `HF_HOME` / `HF_HUB_CACHE` at a persistent `$WORKSPACE/huggingface_cache` — the mega driver does this automatically and it halves cold-start time.

```bash
source ~/workspace/psc_slime_env.sh
cd "$PROJECT_ROOT"
```

---

## Data

- **GSM8K** (headline benchmark): downloaded automatically on first run into `data/gsm8k_{train,eval}.jsonl`. Size is controlled by `TRAIN_SIZE` / `EVAL_SIZE` (defaults 128 / 64).
- **Tiny coding benchmark** (secondary, task-robustness): 12 deterministic hand-written Python tasks in `data/tiny_coding_train.jsonl`, 6 eval tasks in `data/tiny_coding_eval.jsonl`. The reward harness (`src/coding_reward.py`) extracts code blocks, writes them to a temp file, and runs `reference_tests` in a sandboxed subprocess with a timeout.

In-memory truncation for quick smoke tests:

```bash
TASK_LIMIT=16 python3 -m src.run_async_areal_style --dataset data/gsm8k_train.jsonl ...
```

`TASK_LIMIT` truncates *in memory* without regenerating the JSONL files, so you can flip between a 16-task smoke and a 128-task full run instantly.

---

## Quickstart

### 1. Single-runner smoke (2× V100, <5 min)

```bash
TASK_LIMIT=16 MAX_NEW_TOKENS=64 \
  python3 -u -m src.run_async_areal_style \
    --dataset data/gsm8k_train.jsonl \
    --backend hf_trainable \
    --hf-model-id Qwen/Qwen2.5-0.5B-Instruct \
    --hf-dtype float32 \
    --rollout-devices cuda:0 --trainer-devices cuda:1 \
    --num-rollout-workers 1 --num-trainer-workers 1 \
    --staleness-k 4 --decoupled-objective 1 \
    --lr 5e-6 --update-batch-size 4 \
    --summary-json results/quickstart_summary.json
```

You should see a `[areal-setup]` barrier line, then live `[areal][<step>/<total> t=<s>] pv=<...>` heartbeats until completion.

### 2. Three-way smoke grid

```bash
TASK_LIMIT=16 bash scripts/run_experiments.sh
```

Default grid: `sync_train`, `async_train`, `async_areal_style` × `K ∈ {0,1,2,4}` × 3 seeds × 1 epoch. Uses `tiny_policy` backend for speed. Good for verifying the pipeline end-to-end.

### 3. Paper figures only

```bash
bash scripts/run_paper_repro.sh
```

Runs just the sweep needed to regenerate AReaL Fig 5a / 5b / 5c / 6b + Table 2.

---

## Reproducing the full final report

The single source of truth is **`scripts/run_final_report.sh`**. It runs the complete sync + async sweep on GSM8K and the tiny coding benchmark, then invokes the legacy aggregator + both plotters + the Markdown index builder. Every figure, table, and `INDEX.md` the paper and poster cite is produced in one go.

Interactive (recommended for iteration):

```bash
cd /ocean/projects/cis260009p/$USER/repos/llm_asynchronous_rollout
source ~/workspace/psc_slime_env.sh

TASK_LIMIT=128 \
DATASETS=gsm8k,coding \
SEEDS=0,1,2 \
STALENESS_K=0,2,4,8 \
DECOUPLED=0,1 \
ASYNC_MODES=async_areal_style \
MAX_NEW_TOKENS=128 \
HF_USE_AUTOCAST=1 \
HF_DTYPE=float32 \
K_HEADLINE=2 \
bash scripts/run_final_report.sh 2>&1 | tee "logs/final_report_$(date +%Y%m%d_%H%M%S).log"
```

Slurm batch (15 h wall-clock cap on GPU-shared):

```bash
sbatch scripts/submit_final_report.sbatch
# with overrides:
sbatch --export=ALL,DATASETS=gsm8k,SEEDS=0,1 scripts/submit_final_report.sbatch

tail -f logs/final_report_latest.out
tail -f results/experiments/latest/run.log
tail -f results/experiments/latest/gpu.log
```

Quality-preserving speedups already baked in:

- Sync runs are executed **only at `K=0, decoupled=1`** (sync is mathematically invariant to `K` and to the decoupled flag because the ratio is always 1). This removes ~30 redundant runs per dataset.
- `MAX_NEW_TOKENS` defaults to 128 (0.5 B-param Qwen + GSM8K rarely needs more; doesn't affect reward extraction).
- The worker-ready barrier excludes cold-start model loading from `wall_clock_sec`, so warm-vs-cold shells give comparable throughput numbers.

---

## Resuming an interrupted run

PSC time limits and preemptions are a fact of life. The experiment grid supports **crash resumption** out of the box: re-invoke the mega driver with the *same* `EXP_NAME` pointing at the existing output directory. Any run whose `summary.json` is already present and JSON-valid is `SKIP`ped (its metrics still flow into the merged summary and plots); only unfinished runs re-execute.

Resume is auto-detected (`RESUME=auto`), or force it with `RESUME=1`:

```bash
TASK_LIMIT=128 \
EXP_NAME="final_report_20260421_145255" \
DATASETS=gsm8k,coding \
SEEDS=0,1,2 STALENESS_K=0,2,4,8 DECOUPLED=0,1 \
ASYNC_MODES=async_areal_style \
MAX_NEW_TOKENS=128 HF_USE_AUTOCAST=1 HF_DTYPE=float32 K_HEADLINE=2 \
bash scripts/run_final_report.sh
```

Expected trace:

```
[info] RESUME=auto  RESUME_ACTIVE=1
[sweep] sync pass: gsm8k -> results/experiments/.../gsm8k/sync (skip_existing=1)
[1/2] SKIP (summary exists) 001__mode-sync_train__...
[2/2] SKIP (summary exists) 002__mode-sync_train__...
[sweep] async pass: gsm8k -> ... (skip_existing=1)
[1/8] SKIP (summary exists) 001__mode-async_areal_style__k-0__seed-0__...
...
[8/8] Running          008__mode-async_areal_style__k-4__seed-1__...__dec-1
```

To *re-run* a specific run under the fixed code, just delete its directory and re-invoke:

```bash
rm -rf results/experiments/<EXP_NAME>/gsm8k/async/008__mode-async_areal_style__*
bash scripts/run_final_report.sh        # will now re-execute only run 008
```

---

## Outputs and figures

Each run writes:

```
results/experiments/<EXP_NAME>/<dataset>/{sync,async}/<run_name>/
├── summary.json          headline metrics + config + event trace + queue traces
├── results.jsonl         one row per sample (response, reward, policy_version, staleness, dropped, ...)
├── config.json           sidecar with the full argparse Namespace (source of truth for plot filters)
├── command.txt           exact command line used
├── stdout.log, stderr.log
```

The post-processing stage then writes, per dataset:

| Path | What it is |
|---|---|
| `plots/fig_overview_poster.png` | Poster headline multi-panel |
| `plots/fig_reward_curves.png`, `fig_pass_rate_curves.png` | Mean ± stdev learning curves per mode |
| `plots/fig_pareto.png` | Throughput vs. accuracy Pareto frontier |
| `plots/fig_time_to_threshold.png` | Wall-clock to reach a pass-rate threshold |
| `plots/fig_staleness_violin.png` | Realized staleness distribution by `K` |
| `plots/fig_reward_per_ktoken.png` | Sample efficiency (reward per 1 K generated tokens) |
| `plots/fig_updates_per_sec.png` | Trainer throughput |
| `plots/fig5a_naive_learning_curves.png` | **AReaL paper Fig 5a** reproduction (decoupled=0) |
| `plots/fig5b_decoupled_learning_curves.png` | **AReaL paper Fig 5b** reproduction (decoupled=1) |
| `plots/fig5c_throughput_vs_staleness.png` | **AReaL paper Fig 5c** reproduction |
| `plots/fig6b_interruptible_throughput.png` | **AReaL paper Fig 6b** reproduction |
| `plots/reward_vs_staleness.png`, `wall_clock_vs_mode.png`, `updates_vs_mode.png`, `queue_depth_traces_*.png` | Legacy analyzer |
| `tables/summary_headline.{csv,md}` | One row per run |
| `tables/mode_aggregate.{csv,md}` | Seed-averaged per (mode, K, dec) |
| `tables/table2_staleness_vs_pass_rate.{csv,md}` | AReaL Table 2 reproduction |
| `analysis_aggregated.{csv,json}` | Legacy aggregator output |
| `INDEX.md`, `manifest.json` | Human-readable + machine-readable asset map |

A top-level `results/experiments/<EXP_NAME>/INDEX.md` links every dataset's asset tree. `results/experiments/latest` is a symlink to the most recent sweep.

---

## Key configuration knobs

All controllable through environment variables to the mega driver (`scripts/run_final_report.sh`) or CLI flags on `src.run_experiment_grid` / the individual runners. Notable ones:

### Sweep axes

| Env var | Default | Meaning |
|---|---|---|
| `DATASETS` | `gsm8k,coding` | Comma-separated subset of `gsm8k`, `coding`, or raw JSONL path |
| `SEEDS` | `0,1,2` | Random seeds to average over |
| `STALENESS_K` | `0,2,4,8` | Bounded staleness `K` values |
| `DECOUPLED` | `0,1` | Decoupled-objective ablation |
| `ASYNC_MODES` | `async_train,async_areal_style` | Which async runners to include |
| `RUN_SYNC` | `1` | Whether to run the sync baseline |
| `TASK_LIMIT` | unset | Truncate dataset in memory for smoke tests |

### Model / optimizer

| Env var | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal-LM id |
| `HF_DTYPE` | `float32` | `float32` is the V100-stable default; `float16` is OK with autocast |
| `HF_USE_AUTOCAST` | `0` | `1` enables FP16 autocast on `generate()` + logprob recomputation |
| `LR` | `5e-6` | Optimizer learning rate |
| `CLIP_EPS` | `0.2` | `ε` in the clipped surrogate objective |
| `KL_COEF` | `0.02` | Schulman k3 KL-penalty coefficient |
| `TEMPERATURE`, `TOP_P` | `0.9`, `0.95` | Generation sampling params |
| `MAX_NEW_TOKENS` | `128` | Max tokens generated per rollout |

### AReaL topology

| Env var | Default | Meaning |
|---|---|---|
| `ROLLOUT_DEVICES` | `cuda:0` | Comma-separated device list (round-robin across workers) |
| `TRAINER_DEVICES` | `cuda:1` | Same for trainers |
| `UPDATE_BATCH` | `4` | Replay-batch size per trainer step |
| `AREAL_SETUP_TIMEOUT_SEC` | `900` | How long to wait for workers to load model before failing loud |
| `AREAL_SETUP_POLL_SEC` | `5` | Barrier poll interval |
| `AREAL_HEARTBEAT_SEC` | `5` | Progress-reporter heartbeat interval |

### Plotting / thresholds

| Env var | Default | Meaning |
|---|---|---|
| `K_HEADLINE` | `2` | `K` used in headline poster panels |
| `PASS_THRESHOLD_GSM8K` | `0.10` | Threshold for `fig_time_to_threshold.png` |
| `PASS_THRESHOLD_CODING` | `0.10` | Same, for the coding benchmark |

---

## Design notes and lessons learned

These notes are short post-mortems of bugs that took non-trivial time to find. Documenting them here because they're the kind of failure modes a re-implementer is likely to re-encounter.

1. **`fork` + CUDA silently hangs the children.** Python's `mp.Process` defaults to `fork` on Linux. Forking a parent with a live CUDA context corrupts CUDA in every child → the first CUDA call in the worker hangs forever with 0 % GPU utilization and no error. *Fix:* the main process never touches CUDA; the runtime uses `mp.get_context("spawn")` for every primitive (`Manager`, `Queue`, `Event`, `Value`, `Process`) and workers build their own `TrainableHFCausalLMBackend` from scratch on their assigned device.
2. **Cold-start model load was counted in `wall_clock_sec`.** When the HF cache and kernel page cache were cold (e.g. a fresh interactive shell), each worker spent 60–120 s loading Qwen-0.5B from disk while the controller's clock was already ticking, inflating `wall_clock_sec` and deflating `updates/sec` for no algorithmic reason. *Fix:* a **worker-ready barrier** — workers post a `WORKER_READY` sentinel after `build_backend(...)` returns on-device, and the controller's clock only starts after all workers are ready. Cold-start time is reported separately as `setup_wall_clock_sec`.
3. **Signed KL penalty in the decoupled objective silently regressed learning.** The original implementation used `(old_logprobs − new_logprobs).clamp(...)` as the per-token penalty. That quantity is *signed* and can be negative, which inverts the penalty into an incentive to drift away from the rollout policy — catastrophic for `decoupled=1`. *Fix:* use the Schulman k3 unbiased estimator `exp(log r) − 1 − log r`, which is provably non-negative. Immediately recovered `decoupled=1` performance to match or beat `decoupled=0` at high `K`, as AReaL Fig 5b predicts.
4. **`subprocess.run` buffers output and makes the sweep look hung.** Interactive users saw ~10 min of silence after "Running run X" and assumed the job had deadlocked. *Fix:* `src/run_experiment_grid.py` uses `subprocess.Popen` with line-buffered streaming + `PYTHONUNBUFFERED=1` + `python3 -u`, and `src/areal_runtime.py` spawns a heartbeat thread that prints `[areal][<step>/<total> t=<s>] pv=<pv> sq=<rollout_q> tq=<trainer_q>` every 5 s regardless of whether anything else logs.
5. **Config drift between `summary.json` and `config.json`.** Older runs had an incomplete embedded `config` block (missing `decoupled_objective`), which made the plotters silently default to `decoupled=1` for every run and produce empty/mis-labelled figures. *Fix:* both plotters now merge the sidecar `config.json` into `summary["config"]` at load time, and every runner writes the full argparse Namespace into `summary["config"]`. The mega driver also writes `config.json` as a separate artifact so we can re-derive truth if `summary.json` is ever damaged.
6. **NaN/Inf logits on V100 FP16.** Qwen-0.5B with native FP16 weights occasionally emitted non-finite logits during `generate()`, which crashed CUDA with a device-side assert. *Fix:* default `HF_DTYPE=float32`, wrap `generate()` and `_recompute_new_logprobs()` in `torch.autocast(dtype=torch.float16)` under `HF_USE_AUTOCAST=1`, and attach a `_NanInfGuard` `LogitsProcessor` that clamps non-finite logits.
7. **Recursive summary discovery.** `src/analyze_experiments.py` originally globbed only one level deep, missing the `<dataset>/<mode>/<run>` layout introduced by the mega driver. *Fix:* `root.rglob("summary.json")`.

---

## Limitations

Relative to full AReaL / full PPO:

- No value head, no GAE; advantage is group-normalized reward (pure GRPO).
- No FSDP / tensor parallel / Ray; 1 rollout worker + 1 trainer worker on 2 GPUs is the tested topology.
- Parameter service commits the entire trainable state per update (fine for 0.5 B, would need delta-update for >1 B).
- Interruption is chunk-boundary, not token-level preemption.
- FlashAttention-2 disabled (V100/sm\_70 is not supported); `hf_attn_impl=eager` is the default.
- The tiny coding benchmark is 12 train / 6 eval tasks — designed to show cross-task robustness, *not* to claim SOTA code generation.
- Sync-vs-async throughput numbers depend on kernel page cache state; the worker-ready barrier makes them comparable *across* runs in the same sweep, but cross-node comparison still requires pre-warming.

---

## Acknowledgements

- Fu et al., "**AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning**" (InclusionAI / Ant Group, 2025). The systems topology, bounded-staleness formulation, and Fig 5/6 are the direct targets of this re-implementation. Upstream reference: <https://github.com/inclusionai/areal>.
- Schulman et al., "Approximating KL Divergence" (the k3 estimator used in the fixed decoupled objective).
- Qwen team, "Qwen 2.5 Technical Report" (the 0.5 B-Instruct checkpoint used for the headline runs).
- PSC Bridges-2 (NSF ACCESS allocation `cis260009p`) for the V100-32 GB cycles.

---
