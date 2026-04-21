#!/bin/bash
# Mega driver: runs the full comparison sweep + all three analyzers and
# packages every figure and table the final report / poster needs.
#
# Produces (all under results/experiments/${EXP_NAME}/):
#
#   plots/fig_overview_poster.png         -- poster headline, 4-in-1 panel
#   plots/fig_reward_curves.png           -- per-mode reward curves (mean ± stdev)
#   plots/fig_pass_rate_curves.png        -- per-mode pass-rate curves
#   plots/fig_pareto.png                  -- throughput vs accuracy tradeoff
#   plots/fig_time_to_threshold.png       -- wall-clock to reach pass-rate threshold
#   plots/fig_staleness_violin.png        -- realized staleness distribution
#   plots/fig_reward_per_ktoken.png       -- sample efficiency
#   plots/fig_updates_per_sec.png         -- trainer throughput
#
#   plots/fig5a_naive_learning_curves.png      -- AReaL Fig 5a repro
#   plots/fig5b_decoupled_learning_curves.png  -- AReaL Fig 5b repro
#   plots/fig5c_throughput_vs_staleness.png    -- AReaL Fig 5c repro
#   plots/fig6b_interruptible_throughput.png   -- AReaL Fig 6b repro
#   plots/table2_staleness_vs_pass_rate.{csv,md}  -- AReaL Table 2 repro
#
#   plots/reward_vs_staleness.png         -- legacy analyze_experiments
#   plots/wall_clock_vs_mode.png          -- legacy analyze_experiments
#   plots/updates_vs_mode.png             -- legacy analyze_experiments
#   plots/queue_depth_traces_*.png        -- legacy analyze_experiments
#
#   tables/summary_headline.{csv,md}      -- one row per run
#   tables/mode_aggregate.{csv,md}        -- seed-averaged per (mode,k,decoupled)
#   analysis_aggregated.{csv,json}        -- legacy aggregates
#
#   INDEX.md                              -- explains which asset goes where
#   merged_summary.{json,csv}             -- every run summary concatenated
#   manifest.json                         -- grid config that produced the runs
#
# Preconditions (interactive):
#   source ~/workspace/psc_slime_env.sh
#   cd $PROJECT_ROOT
#
# For batch submission use scripts/submit_final_report.sbatch.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

: "${WORKSPACE:=/ocean/projects/cis260009p/${USER:-tzeng1}}"
: "${HF_HOME:=${WORKSPACE}/huggingface}"
: "${HF_HUB_CACHE:=${WORKSPACE}/huggingface_cache}"
export HF_HOME HF_HUB_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ----- Sweep defaults (override via env) ------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-data/gsm8k_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/gsm8k_eval.jsonl}"
TRAIN_SIZE="${TRAIN_SIZE:-128}"
EVAL_SIZE="${EVAL_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
UPDATE_BATCH="${UPDATE_BATCH:-4}"
LR="${LR:-5e-6}"
SEEDS="${SEEDS:-0,1,2}"
STALENESS_K="${STALENESS_K:-0,1,2,4,8}"
DECOUPLED="${DECOUPLED:-0,1}"
MODES="${MODES:-sync_train,async_train,async_areal_style}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
KL_COEF="${KL_COEF:-0.02}"
CLIP_EPS="${CLIP_EPS:-0.2}"
HF_DTYPE="${HF_DTYPE:-float32}"

ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-cuda:0}"
TRAINER_DEVICES="${TRAINER_DEVICES:-cuda:1}"

K_HEADLINE="${K_HEADLINE:-2}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-final_report_${STAMP}}"
EXP_DIR="results/experiments/${EXP_NAME}"

mkdir -p results/experiments data logs

if [[ ! -s "${TRAIN_DATA}" || ! -s "${EVAL_DATA}" ]]; then
  echo "[prep] Downloading GSM8K"
  python3 -m src.gsm8k_data \
    --train-out "${TRAIN_DATA}" --eval-out "${EVAL_DATA}" \
    --train-size "${TRAIN_SIZE}" --eval-size "${EVAL_SIZE}"
fi

# Keep a live "latest" symlink
ln -sfn "${EXP_NAME}" results/experiments/latest || true

echo "[info] MODEL_ID=${MODEL_ID}"
echo "[info] MODES=${MODES}"
echo "[info] SEEDS=${SEEDS}  STALENESS_K=${STALENESS_K}  DECOUPLED=${DECOUPLED}"
echo "[info] EXP_DIR=${EXP_DIR}"
nvidia-smi --list-gpus || true

# ----- Step 1: grid sweep ---------------------------------------------------
python3 -m src.run_experiment_grid \
  --experiment-name "${EXP_NAME}" \
  --output-root results/experiments \
  --dataset "${TRAIN_DATA}" \
  --modes "${MODES}" \
  --staleness-k-values "${STALENESS_K}" \
  --seeds "${SEEDS}" \
  --update-batch-sizes "${UPDATE_BATCH}" \
  --queue-maxsizes 256 \
  --epochs-values "${EPOCHS}" \
  --backend-values hf_trainable \
  --hf-model-id "${MODEL_ID}" \
  --hf-dtype "${HF_DTYPE}" \
  --hf-attn-impl eager \
  --hf-chat-template 1 \
  --grpo-epsilon "${CLIP_EPS}" \
  --grpo-kl-coef "${KL_COEF}" \
  --grpo-group-size 0 \
  --grad-clip 1.0 \
  --weight-decay 0.0 \
  --decoupled-objective-values "${DECOUPLED}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --lr-values "${LR}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-rollout-workers-values 1 \
  --num-trainer-workers-values 1 \
  --rollout-devices "${ROLLOUT_DEVICES}" \
  --trainer-devices "${TRAINER_DEVICES}" \
  --producer-delay-sec 0.0 \
  --learner-delay-sec 0.0 \
  --interrupt-check-interval-sec 0.05 \
  --generation-chunk-size 1 \
  --rollout-chunk-delay-sec 0.0 \
  --replay-dispatch-delay-sec 0.0 \
  --controller-consume-delay-sec 0.0 \
  --max-interrupt-retries 1 \
  --reward-timeout-sec 2.0 \
  "$@"

# ----- Step 2: legacy analyzer (wall-clock, reward_vs_staleness, queues) ----
echo "[post] Legacy analyze_experiments -> ${EXP_DIR}"
python3 -m src.analyze_experiments --experiment-dir "${EXP_DIR}" || true

# ----- Step 3: paper-style repro plots --------------------------------------
echo "[post] plot_paper_repro -> ${EXP_DIR}/plots"
python3 -m src.plot_paper_repro --experiment-dir "${EXP_DIR}" || true

# ----- Step 4: final-report plots + tables + INDEX --------------------------
echo "[post] plot_final_report -> ${EXP_DIR}/plots + ${EXP_DIR}/tables"
python3 -m src.plot_final_report \
  --experiment-dir "${EXP_DIR}" \
  --k-headline "${K_HEADLINE}" \
  --threshold "${PASS_THRESHOLD}" || true

echo "[done] ${EXP_DIR}"
echo ""
echo "===== Asset summary ====="
echo "Plots:   ${EXP_DIR}/plots/"
ls -1 "${EXP_DIR}/plots/" 2>/dev/null | sed 's|^|  |'
echo "Tables:  ${EXP_DIR}/tables/"
ls -1 "${EXP_DIR}/tables/" 2>/dev/null | sed 's|^|  |'
echo "Index:   ${EXP_DIR}/INDEX.md"
