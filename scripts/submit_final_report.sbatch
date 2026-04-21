#!/bin/bash
# Submit the full final-report sweep as a Slurm batch job on PSC.
#
# From repo root:
#     sbatch scripts/submit_final_report.sbatch
#
# Live tail:
#     tail -f logs/final_report_latest.out
#     tail -f results/experiments/latest/run.log
#     tail -f results/experiments/latest/gpu.log
#
# The job calls scripts/run_final_report.sh which produces ALL assets needed
# for the final report and the poster. See INDEX.md inside the experiment dir.
#
#SBATCH --job-name=areal_final_report
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:2
#SBATCH --time=15:00:00
#SBATCH --output=logs/final_report_%j.out
#SBATCH --error=logs/final_report_%j.err

set -euo pipefail

: "${WORKSPACE:=/ocean/projects/cis260009p/${USER:-tzeng1}}"
PROJECT_ROOT="${WORKSPACE}/repos/llm_async_project"
cd "${PROJECT_ROOT}"

mkdir -p logs results/experiments

ln -sfn "final_report_${SLURM_JOB_ID:-local}.out" logs/final_report_latest.out
ln -sfn "final_report_${SLURM_JOB_ID:-local}.err" logs/final_report_latest.err

source "${WORKSPACE}/psc_slime_env.sh"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
export EXP_NAME="final_report_${SLURM_JOB_ID:-local}_${STAMP}"
EXP_DIR="results/experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}"
ln -sfn "${EXP_NAME}" results/experiments/latest
RUN_LOG="${EXP_DIR}/run.log"
GPU_LOG="${EXP_DIR}/gpu.log"
touch "${RUN_LOG}" "${GPU_LOG}"

{
  echo "===================================================================="
  echo " Slurm job:       ${SLURM_JOB_ID:-N/A}"
  echo " Node:            $(hostname)"
  echo " Date:            $(date)"
  echo " Experiment dir:  ${EXP_DIR}"
  echo " Conda env:       ${CONDA_PREFIX:-unknown}"
  echo "===================================================================="
  nvidia-smi -L || true
  echo "===================================================================="
} | tee -a "${RUN_LOG}"

# Background GPU monitor
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

# Default sweep knobs (comprehensive but quality-preserving). Override at
# submission time, e.g.:
#     sbatch --export=ALL,DATASETS=gsm8k,SEEDS=0,1 scripts/submit_final_report.sbatch
# or skip coding:
#     sbatch --export=ALL,DATASETS=gsm8k scripts/submit_final_report.sbatch
: "${MODEL_ID:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${DATASETS:=gsm8k,coding}"
: "${ASYNC_MODES:=async_train,async_areal_style}"
: "${RUN_SYNC:=1}"
: "${SEEDS:=0,1,2}"
: "${STALENESS_K:=0,2,4,8}"
: "${DECOUPLED:=0,1}"
: "${EPOCHS:=1}"
: "${MAX_NEW_TOKENS:=128}"
: "${TRAIN_SIZE:=128}"
: "${EVAL_SIZE:=64}"
: "${LR:=5e-6}"
: "${HF_DTYPE:=float32}"
: "${UPDATE_BATCH:=4}"
: "${K_HEADLINE:=2}"
: "${PASS_THRESHOLD_GSM8K:=0.10}"
: "${PASS_THRESHOLD_CODING:=0.10}"
export MODEL_ID DATASETS ASYNC_MODES RUN_SYNC SEEDS STALENESS_K DECOUPLED \
       EPOCHS MAX_NEW_TOKENS TRAIN_SIZE EVAL_SIZE LR HF_DTYPE UPDATE_BATCH \
       K_HEADLINE PASS_THRESHOLD_GSM8K PASS_THRESHOLD_CODING

{
  echo "[config] MODEL_ID=${MODEL_ID}"
  echo "[config] DATASETS=${DATASETS}  ASYNC_MODES=${ASYNC_MODES}  RUN_SYNC=${RUN_SYNC}"
  echo "[config] SEEDS=${SEEDS}  STALENESS_K=${STALENESS_K}  DECOUPLED=${DECOUPLED}"
  echo "[config] EPOCHS=${EPOCHS}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
  echo "[config] TRAIN_SIZE=${TRAIN_SIZE}  EVAL_SIZE=${EVAL_SIZE}"
  echo "[config] LR=${LR}  HF_DTYPE=${HF_DTYPE}  UPDATE_BATCH=${UPDATE_BATCH}"
  echo "[config] K_HEADLINE=${K_HEADLINE}"
  echo "[config] PASS_THRESHOLD_GSM8K=${PASS_THRESHOLD_GSM8K}  PASS_THRESHOLD_CODING=${PASS_THRESHOLD_CODING}"
} | tee -a "${RUN_LOG}"

stdbuf -oL -eL \
  bash scripts/run_final_report.sh 2>&1 \
  | tee -a "${RUN_LOG}"

{
  echo "===================================================================="
  echo " Finished at:     $(date)"
  echo " Results dir:     ${EXP_DIR}"
  echo " INDEX:           ${EXP_DIR}/INDEX.md"
  echo " Plots:           ${EXP_DIR}/plots/"
  echo " Tables:          ${EXP_DIR}/tables/"
  echo "===================================================================="
} | tee -a "${RUN_LOG}"
