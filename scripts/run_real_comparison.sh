#!/bin/bash
# Real comparison on 2x V100: GSM8K + Qwen2.5-0.5B-Instruct, FP16, SDPA, real GRPO/PPO.
#
# Preconditions (run in the GPU interact shell FIRST):
#   source ~/workspace/psc_slime_env.sh
#   cd $PROJECT_ROOT
#
# Then:
#   bash scripts/run_real_comparison.sh
#
# Caches and outputs live on /ocean via $WORKSPACE.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

: "${WORKSPACE:=/ocean/projects/cis260009p/${USER:-tzeng1}}"
: "${HF_HOME:=${WORKSPACE}/huggingface}"
: "${HF_HUB_CACHE:=${WORKSPACE}/huggingface_cache}"
export HF_HOME HF_HUB_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# V100 safety knobs (no FlashAttention v2, no bf16).
export TORCH_COMPILE_DISABLE=1
export PYTORCH_SDPA_FLASH_ATTENTION=0
export TRANSFORMERS_OFFLINE=0

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-data/gsm8k_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/gsm8k_eval.jsonl}"
TRAIN_SIZE="${TRAIN_SIZE:-128}"
EVAL_SIZE="${EVAL_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
UPDATE_BATCH="${UPDATE_BATCH:-4}"
LR="${LR:-5e-6}"
SEEDS="${SEEDS:-0,1}"
STALENESS_K="${STALENESS_K:-0,2,4}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
KL_COEF="${KL_COEF:-0.02}"
CLIP_EPS="${CLIP_EPS:-0.2}"

# Device placement: rollout on cuda:0, trainer on cuda:1.
ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-cuda:0}"
TRAINER_DEVICES="${TRAINER_DEVICES:-cuda:1}"

mkdir -p results/experiments data

# --- 1. Download GSM8K locally if missing -----------------------------------
if [[ ! -s "${TRAIN_DATA}" || ! -s "${EVAL_DATA}" ]]; then
  echo "[prep] Downloading GSM8K into ${TRAIN_DATA} / ${EVAL_DATA}"
  python3 -m src.gsm8k_data \
    --train-out "${TRAIN_DATA}" \
    --eval-out "${EVAL_DATA}" \
    --train-size "${TRAIN_SIZE}" \
    --eval-size "${EVAL_SIZE}"
fi

echo "[info] MODEL_ID=${MODEL_ID}"
echo "[info] TRAIN=${TRAIN_DATA} ($(wc -l < "${TRAIN_DATA}") rows)"
echo "[info] nvidia-smi --list-gpus:"
nvidia-smi --list-gpus || true

# --- 2. Launch grid ---------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-gsm8k_qwen_${STAMP}}"

python3 -m src.run_experiment_grid \
  --experiment-name "${EXP_NAME}" \
  --output-root results/experiments \
  --dataset "${TRAIN_DATA}" \
  --modes sync_train,async_train,async_areal_style \
  --staleness-k-values "${STALENESS_K}" \
  --seeds "${SEEDS}" \
  --update-batch-sizes "${UPDATE_BATCH}" \
  --queue-maxsizes 256 \
  --epochs-values "${EPOCHS}" \
  --backend-values hf_trainable \
  --hf-model-id "${MODEL_ID}" \
  --hf-dtype "${HF_DTYPE:-float32}" \
  --hf-attn-impl eager \
  --hf-chat-template 1 \
  --grpo-epsilon "${CLIP_EPS}" \
  --grpo-kl-coef "${KL_COEF}" \
  --grpo-group-size 0 \
  --grad-clip 1.0 \
  --weight-decay 0.0 \
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

# --- 3. Aggregate + plots ---------------------------------------------------
EXP_DIR="results/experiments/${EXP_NAME}"
echo "[post] Analyzing ${EXP_DIR}"
python3 -m src.analyze_experiments --experiment-dir "${EXP_DIR}" || true
echo "[done] results in ${EXP_DIR}"
