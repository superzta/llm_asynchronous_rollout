#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the real GSM8K + Qwen 3-way comparison as a Slurm batch job on PSC.
#
# Submit with:
#     sbatch scripts/submit_real_comparison.sbatch
#
# While the job is running you can live-tail:
#     tail -f logs/real_comparison_latest.out
#     tail -f results/experiments/latest/run.log
#     tail -f results/experiments/latest/gpu.log
# ---------------------------------------------------------------------------
#SBATCH --job-name=async_areal_cmp
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=2
#SBATCH --time=15:00:00
#SBATCH --output=logs/real_comparison_%j.out
#SBATCH --error=logs/real_comparison_%j.err

set -euo pipefail

# ---------- Project paths ---------------------------------------------------
WORKSPACE_DEFAULT=/ocean/projects/cis260009p/${USER:-tzeng1}
: "${WORKSPACE:=${WORKSPACE_DEFAULT}}"
PROJECT_ROOT="${WORKSPACE}/repos/llm_async_project"
cd "${PROJECT_ROOT}"

mkdir -p logs results/experiments

# Make a stable "latest" pointer pointing at logs/ for quick ssh-tail.
ln -sfn "logs/real_comparison_${SLURM_JOB_ID:-local}.out" logs/real_comparison_latest.out
ln -sfn "logs/real_comparison_${SLURM_JOB_ID:-local}.err" logs/real_comparison_latest.err

# ---------- Load modules + conda env ---------------------------------------
source "${WORKSPACE}/psc_slime_env.sh"

# Unbuffered Python so stdout streams live.
export PYTHONUNBUFFERED=1

# ---------- Experiment naming / live log paths -----------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
export EXP_NAME="gsm8k_qwen_${SLURM_JOB_ID:-local}_${STAMP}"
EXP_DIR="results/experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

# "latest" symlink so you can tail without knowing the timestamp.
ln -sfn "${EXP_NAME}" results/experiments/latest
RUN_LOG="${EXP_DIR}/run.log"
GPU_LOG="${EXP_DIR}/gpu.log"
touch "${RUN_LOG}" "${GPU_LOG}"

# ---------- Banner ---------------------------------------------------------
{
  echo "===================================================================="
  echo " Slurm job:          ${SLURM_JOB_ID:-N/A}"
  echo " Node:               $(hostname)"
  echo " Date:               $(date)"
  echo " Project root:       ${PROJECT_ROOT}"
  echo " Experiment name:    ${EXP_NAME}"
  echo " Experiment dir:     ${EXP_DIR}"
  echo " Conda env:          ${CONDA_PREFIX:-unknown}"
  echo " Live run log:       ${RUN_LOG}"
  echo " Live gpu log:       ${GPU_LOG}"
  echo "===================================================================="
  nvidia-smi -L || true
  echo "===================================================================="
} | tee -a "${RUN_LOG}"

# ---------- Background GPU monitor -----------------------------------------
(
  while true; do
    ts="$(date +%H:%M:%S)"
    echo "[${ts}]"
    nvidia-smi \
      --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader
    echo ""
    sleep 30
  done
) >> "${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
trap 'kill ${GPU_MONITOR_PID} 2>/dev/null || true' EXIT

# ---------- Knobs (override via --export=ALL,FOO=bar when submitting) ------
: "${MODEL_ID:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${TRAIN_SIZE:=128}"
: "${EVAL_SIZE:=64}"
: "${EPOCHS:=1}"
: "${MAX_NEW_TOKENS:=192}"
: "${UPDATE_BATCH:=4}"
: "${LR:=5e-6}"
: "${SEEDS:=0,1}"
: "${STALENESS_K:=0,2,4}"
: "${TEMPERATURE:=0.9}"
: "${TOP_P:=0.95}"
: "${KL_COEF:=0.02}"
: "${CLIP_EPS:=0.2}"
: "${HF_DTYPE:=float32}"
: "${ROLLOUT_DEVICES:=cuda:0}"
: "${TRAINER_DEVICES:=cuda:1}"
export MODEL_ID TRAIN_SIZE EVAL_SIZE EPOCHS MAX_NEW_TOKENS UPDATE_BATCH LR \
       SEEDS STALENESS_K TEMPERATURE TOP_P KL_COEF CLIP_EPS HF_DTYPE \
       ROLLOUT_DEVICES TRAINER_DEVICES

{
  echo "[config] MODEL_ID=${MODEL_ID}"
  echo "[config] TRAIN_SIZE=${TRAIN_SIZE}  EVAL_SIZE=${EVAL_SIZE}  EPOCHS=${EPOCHS}"
  echo "[config] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}  UPDATE_BATCH=${UPDATE_BATCH}"
  echo "[config] LR=${LR}  TEMPERATURE=${TEMPERATURE}  TOP_P=${TOP_P}"
  echo "[config] SEEDS=${SEEDS}  STALENESS_K=${STALENESS_K}"
  echo "[config] KL_COEF=${KL_COEF}  CLIP_EPS=${CLIP_EPS}  HF_DTYPE=${HF_DTYPE}"
  echo "[config] ROLLOUT_DEVICES=${ROLLOUT_DEVICES}  TRAINER_DEVICES=${TRAINER_DEVICES}"
} | tee -a "${RUN_LOG}"

# ---------- Run ------------------------------------------------------------
# stdbuf + tee => live output both to slurm .out/.err AND to EXP_DIR/run.log
stdbuf -oL -eL \
  bash scripts/run_real_comparison.sh 2>&1 \
  | tee -a "${RUN_LOG}"

# ---------- Done -----------------------------------------------------------
{
  echo "===================================================================="
  echo " Finished at:        $(date)"
  echo " Results dir:        ${EXP_DIR}"
  echo " Plots:              ${EXP_DIR}/plots"
  echo "===================================================================="
} | tee -a "${RUN_LOG}"
