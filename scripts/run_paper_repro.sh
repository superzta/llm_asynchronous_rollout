#!/bin/bash
# Paper-reproduction driver: sweeps the axes that produce Fig 5a, 5b, 5c, 6b
# and Table 2 of the AReaL paper, at our 2x V100 / Qwen2.5-0.5B scale.
#
# Preconditions (interactive shell):
#   source ~/workspace/psc_slime_env.sh
#   cd $PROJECT_ROOT
#
# Submitting as a batch job: use scripts/submit_paper_repro.sbatch.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

: "${WORKSPACE:=/ocean/projects/cis260009p/${USER:-tzeng1}}"
: "${HF_HOME:=${WORKSPACE}/huggingface}"
: "${HF_HUB_CACHE:=${WORKSPACE}/huggingface_cache}"
export HF_HOME HF_HUB_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

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
STALENESS_K="${STALENESS_K:-0,1,2,4,8}"
DECOUPLED="${DECOUPLED:-0,1}"
HF_DTYPE="${HF_DTYPE:-float32}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
KL_COEF="${KL_COEF:-0.02}"
CLIP_EPS="${CLIP_EPS:-0.2}"

ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-cuda:0}"
TRAINER_DEVICES="${TRAINER_DEVICES:-cuda:1}"

mkdir -p results/experiments data

if [[ ! -s "${TRAIN_DATA}" || ! -s "${EVAL_DATA}" ]]; then
  echo "[prep] Downloading GSM8K locally"
  python3 -m src.gsm8k_data \
    --train-out "${TRAIN_DATA}" --eval-out "${EVAL_DATA}" \
    --train-size "${TRAIN_SIZE}" --eval-size "${EVAL_SIZE}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-paper_repro_${STAMP}}"

echo "[info] MODEL_ID=${MODEL_ID}"
echo "[info] EXP_NAME=${EXP_NAME}"
echo "[info] SEEDS=${SEEDS}  STALENESS_K=${STALENESS_K}  DECOUPLED=${DECOUPLED}"
nvidia-smi --list-gpus || true

# The grid expands modes x staleness x decoupled. We run:
#   - sync_train              (oracle / baseline; staleness ignored)
#   - async_train             (non-interruptible; for Fig 6b)
#   - async_areal_style       (interruptible; main AReaL curve)
# All with decoupled ∈ {0,1} to reproduce Fig 5a/5b and Table 2.
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

EXP_DIR="results/experiments/${EXP_NAME}"
echo "[post] Standard analysis -> ${EXP_DIR}"
python3 -m src.analyze_experiments --experiment-dir "${EXP_DIR}" || true

echo "[post] Paper-style plots -> ${EXP_DIR}/plots"
python3 -m src.plot_paper_repro --experiment-dir "${EXP_DIR}" || true

echo "[done] ${EXP_DIR}"
