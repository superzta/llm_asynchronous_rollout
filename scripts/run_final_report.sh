#!/bin/bash
# Mega driver: runs the full comparison sweep + all three analyzers and
# packages every figure / table / index the final report + poster need.
#
# Quality-preserving speedups built in:
#   * Sync runs are only executed at k=0, decoupled=1 (sync is mathematically
#     invariant to k and to the decoupled/naive objective since the ratio is
#     always 1). This removes ~30 redundant runs per dataset.
#   * Staleness ticks trimmed from {0,1,2,4,8} -> {0,2,4,8}.
#   * MAX_NEW_TOKENS 192 -> 128 (GSM8K answers + 0.5B model rarely need more
#     than 128 tokens; has no effect on reward extraction).
#
# Second-task validation:
#   * Runs the same sweep on BOTH data/gsm8k_*.jsonl (headline) and
#     data/tiny_coding_*.jsonl (secondary) so the report can show cross-task
#     robustness. Set DATASETS=gsm8k to skip coding, or DATASETS=coding to
#     skip gsm8k.
#
# Produces, per dataset, under results/experiments/${EXP_NAME}/<dataset>/:
#
#   plots/fig_overview_poster.png         -- poster headline panel
#   plots/fig_reward_curves.png           -- mean +/- stdev per-mode reward
#   plots/fig_pass_rate_curves.png        -- mean +/- stdev per-mode pass rate
#   plots/fig_pareto.png                  -- throughput vs accuracy
#   plots/fig_time_to_threshold.png       -- wall-clock to reach threshold
#   plots/fig_staleness_violin.png        -- realized staleness
#   plots/fig_reward_per_ktoken.png       -- sample efficiency
#   plots/fig_updates_per_sec.png         -- trainer throughput
#
#   plots/fig5a_naive_learning_curves.png       -- AReaL paper Fig 5a
#   plots/fig5b_decoupled_learning_curves.png   -- AReaL paper Fig 5b
#   plots/fig5c_throughput_vs_staleness.png     -- AReaL paper Fig 5c
#   plots/fig6b_interruptible_throughput.png    -- AReaL paper Fig 6b
#   plots/table2_staleness_vs_pass_rate.{csv,md}
#
#   plots/reward_vs_staleness.png, wall_clock_vs_mode.png, updates_vs_mode.png,
#   queue_depth_traces_*.png   -- legacy analyzer
#
#   tables/summary_headline.{csv,md}     -- one row per run
#   tables/mode_aggregate.{csv,md}       -- seed-averaged per (mode, k, dec)
#   analysis_aggregated.{csv,json}       -- legacy aggregator
#   INDEX.md, manifest.json
#
# And a top-level ${EXP_NAME}/INDEX.md linking every dataset's asset tree.
#
# Preconditions (interactive):
#   source ~/workspace/psc_slime_env.sh && cd $PROJECT_ROOT
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

# -------- sweep defaults (override via env) ---------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASETS="${DATASETS:-gsm8k,coding}"
TRAIN_SIZE="${TRAIN_SIZE:-128}"
EVAL_SIZE="${EVAL_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
UPDATE_BATCH="${UPDATE_BATCH:-4}"
LR="${LR:-5e-6}"
SEEDS="${SEEDS:-0,1,2}"
STALENESS_K="${STALENESS_K:-0,2,4,8}"
DECOUPLED="${DECOUPLED:-0,1}"
ASYNC_MODES="${ASYNC_MODES:-async_train,async_areal_style}"
RUN_SYNC="${RUN_SYNC:-1}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
KL_COEF="${KL_COEF:-0.02}"
CLIP_EPS="${CLIP_EPS:-0.2}"
HF_DTYPE="${HF_DTYPE:-float32}"

ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-cuda:0}"
TRAINER_DEVICES="${TRAINER_DEVICES:-cuda:1}"

K_HEADLINE="${K_HEADLINE:-2}"
PASS_THRESHOLD_GSM8K="${PASS_THRESHOLD_GSM8K:-0.10}"
PASS_THRESHOLD_CODING="${PASS_THRESHOLD_CODING:-0.10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-final_report_${STAMP}}"
EXP_DIR_REL="results/experiments/${EXP_NAME}"
EXP_DIR="${REPO_ROOT}/${EXP_DIR_REL}"

mkdir -p "${EXP_DIR}" data logs
ln -sfn "${EXP_NAME}" results/experiments/latest || true

echo "[info] MODEL_ID=${MODEL_ID}"
echo "[info] DATASETS=${DATASETS}"
echo "[info] ASYNC_MODES=${ASYNC_MODES}  RUN_SYNC=${RUN_SYNC}"
echo "[info] SEEDS=${SEEDS}  STALENESS_K=${STALENESS_K}  DECOUPLED=${DECOUPLED}"
echo "[info] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[info] EXP_DIR=${EXP_DIR}"
nvidia-smi --list-gpus || true

# -------- dataset resolution ------------------------------------------------
dataset_train_path() {
  case "$1" in
    gsm8k)  echo "data/gsm8k_train.jsonl" ;;
    coding) echo "data/tiny_coding_train.jsonl" ;;
    *)      echo "$1" ;;   # treat as raw path
  esac
}

dataset_threshold() {
  case "$1" in
    gsm8k)  echo "${PASS_THRESHOLD_GSM8K}" ;;
    coding) echo "${PASS_THRESHOLD_CODING}" ;;
    *)      echo "0.10" ;;
  esac
}

prepare_gsm8k_if_missing() {
  if [[ ! -s data/gsm8k_train.jsonl || ! -s data/gsm8k_eval.jsonl ]]; then
    echo "[prep] Downloading GSM8K"
    python3 -m src.gsm8k_data \
      --train-out data/gsm8k_train.jsonl \
      --eval-out data/gsm8k_eval.jsonl \
      --train-size "${TRAIN_SIZE}" --eval-size "${EVAL_SIZE}"
  fi
}

# -------- common flag set ---------------------------------------------------
# Expanded into the "$@" position of run_experiment_grid.
common_flags() {
  local dataset_path="$1"
  cat <<EOF
--output-root ${EXP_DIR_REL}
--dataset ${dataset_path}
--update-batch-sizes ${UPDATE_BATCH}
--queue-maxsizes 256
--epochs-values ${EPOCHS}
--backend-values hf_trainable
--hf-model-id ${MODEL_ID}
--hf-dtype ${HF_DTYPE}
--hf-attn-impl eager
--hf-chat-template 1
--grpo-epsilon ${CLIP_EPS}
--grpo-kl-coef ${KL_COEF}
--grpo-group-size 0
--grad-clip 1.0
--weight-decay 0.0
--temperature ${TEMPERATURE}
--top-p ${TOP_P}
--lr-values ${LR}
--max-new-tokens ${MAX_NEW_TOKENS}
--num-rollout-workers-values 1
--num-trainer-workers-values 1
--rollout-devices ${ROLLOUT_DEVICES}
--trainer-devices ${TRAINER_DEVICES}
--producer-delay-sec 0.0
--learner-delay-sec 0.0
--interrupt-check-interval-sec 0.05
--generation-chunk-size 1
--rollout-chunk-delay-sec 0.0
--replay-dispatch-delay-sec 0.0
--controller-consume-delay-sec 0.0
--max-interrupt-retries 1
--reward-timeout-sec 2.0
EOF
}

run_sync_pass() {
  local dataset="$1" dataset_path="$2" ds_dir="$3"
  local sub="${ds_dir#${EXP_DIR_REL}/}"   # e.g. "gsm8k/sync"
  echo "[sweep] sync pass: ${dataset} -> ${ds_dir}/sync"
  python3 -m src.run_experiment_grid \
    --experiment-name "${sub}/sync" \
    --modes sync_train \
    --staleness-k-values 0 \
    --seeds "${SEEDS}" \
    --decoupled-objective-values 1 \
    $(common_flags "${dataset_path}")
}

run_async_pass() {
  local dataset="$1" dataset_path="$2" ds_dir="$3"
  local sub="${ds_dir#${EXP_DIR_REL}/}"   # e.g. "gsm8k/async"
  echo "[sweep] async pass: ${dataset} -> ${ds_dir}/async"
  python3 -m src.run_experiment_grid \
    --experiment-name "${sub}/async" \
    --modes "${ASYNC_MODES}" \
    --staleness-k-values "${STALENESS_K}" \
    --seeds "${SEEDS}" \
    --decoupled-objective-values "${DECOUPLED}" \
    $(common_flags "${dataset_path}")
}

# -------- per-dataset loop --------------------------------------------------
IFS=',' read -r -a DATASET_LIST <<<"${DATASETS}"
for dataset in "${DATASET_LIST[@]}"; do
  dataset_path="$(dataset_train_path "${dataset}")"
  threshold="$(dataset_threshold "${dataset}")"
  ds_dir_rel="${EXP_DIR_REL}/${dataset}"
  ds_dir="${REPO_ROOT}/${ds_dir_rel}"
  mkdir -p "${ds_dir}"

  if [[ "${dataset}" == "gsm8k" ]]; then
    prepare_gsm8k_if_missing
  fi

  if [[ ! -s "${dataset_path}" ]]; then
    echo "[warn] ${dataset_path} missing, skipping dataset=${dataset}"
    continue
  fi

  if [[ "${RUN_SYNC}" == "1" ]]; then
    run_sync_pass "${dataset}" "${dataset_path}" "${ds_dir_rel}"
  fi
  run_async_pass "${dataset}" "${dataset_path}" "${ds_dir_rel}"

  echo "[post][${dataset}] legacy analyze_experiments"
  python3 -m src.analyze_experiments --experiment-dir "${ds_dir_rel}" || true

  echo "[post][${dataset}] plot_paper_repro"
  python3 -m src.plot_paper_repro --experiment-dir "${ds_dir_rel}" || true

  echo "[post][${dataset}] plot_final_report"
  python3 -m src.plot_final_report \
    --experiment-dir "${ds_dir_rel}" \
    --k-headline "${K_HEADLINE}" \
    --threshold "${threshold}" || true

  echo "[done][${dataset}] ${ds_dir}"
done

# -------- top-level INDEX.md ------------------------------------------------
cat > "${EXP_DIR}/INDEX.md" <<EOF
# Final report assets: ${EXP_NAME}

Generated $(date).

## Datasets included
$(printf -- "- **%s**: [INDEX](%s/INDEX.md) | [plots](%s/plots/) | [tables](%s/tables/)\n" \
    "${DATASET_LIST[@]}" "${DATASET_LIST[@]}" "${DATASET_LIST[@]}" "${DATASET_LIST[@]}")

## Headline assets
For the main body of the report and the poster, use GSM8K as the headline dataset:
- Poster panel: \`gsm8k/plots/fig_overview_poster.png\`
- Learning curves: \`gsm8k/plots/fig_reward_curves.png\`, \`gsm8k/plots/fig_pass_rate_curves.png\`
- Speedup bar: \`gsm8k/plots/fig_time_to_threshold.png\`
- Pareto: \`gsm8k/plots/fig_pareto.png\`
- AReaL Fig 5b (staleness tolerance with decoupled objective): \`gsm8k/plots/fig5b_decoupled_learning_curves.png\`
- AReaL Fig 5c (throughput vs staleness): \`gsm8k/plots/fig5c_throughput_vs_staleness.png\`

## Secondary (task-robustness) assets
To support the claim that the ordering generalizes beyond GSM8K, cite the matching coding plots:
- \`coding/plots/fig_reward_curves.png\`
- \`coding/plots/fig_pareto.png\`
- \`coding/tables/mode_aggregate.md\`

## Raw inputs
Under each dataset sub-dir:
- \`sync/\` -- runs produced by the compact sync pass
- \`async/\` -- runs produced by the async sweep
- Each run dir has \`summary.json\`, \`results.jsonl\`, \`config.json\`, \`stdout.log\`, \`stderr.log\`.
EOF

echo ""
echo "===== Asset summary ====="
for dataset in "${DATASET_LIST[@]}"; do
  ds_dir="${EXP_DIR}/${dataset}"
  [[ -d "${ds_dir}" ]] || continue
  echo "[${dataset}] ${ds_dir}"
  echo "  plots:"
  ls -1 "${ds_dir}/plots/" 2>/dev/null | sed 's|^|    |'
  echo "  tables:"
  ls -1 "${ds_dir}/tables/" 2>/dev/null | sed 's|^|    |'
done
echo ""
echo "Top-level index: ${EXP_DIR}/INDEX.md"
