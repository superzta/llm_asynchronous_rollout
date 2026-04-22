#!/bin/bash
# 8-hour GPU job: full final-report driver (GSM8K + coding, plots, tables).
#
# Defaults below are tuned so the sweep usually finishes within ~8h on
# 2× V100-32GB (Qwen2.5-0.5B). Override any variable at submit time, e.g.:
#   sbatch --export=ALL,DATASETS=gsm8k,SEEDS=0,1,2 scripts/submit_final_report_8h.sbatch
#
# From repo root (or any cwd; script cds to PROJECT_ROOT):
#   sbatch /path/to/llm_async_project/scripts/submit_final_report_8h.sbatch
#
# Live tail:
#   tail -f logs/final_report_8h_%j.out
#   tail -f results/experiments/latest/run.log
#   tail -f results/experiments/latest/gpu.log
#
#SBATCH --job-name=areal_final_8h
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_report_8h_%j.out
#SBATCH --error=logs/final_report_8h_%j.err

set -euo pipefail

: "${WORKSPACE:=/ocean/projects/cis260009p/${USER:-tzeng1}}"
PROJECT_ROOT="${WORKSPACE}/repos/llm_async_project"
cd "${PROJECT_ROOT}"

mkdir -p logs results/experiments

ln -sfn "final_report_8h_${SLURM_JOB_ID:-local}.out" logs/final_report_8h_latest.out
ln -sfn "final_report_8h_${SLURM_JOB_ID:-local}.err" logs/final_report_8h_latest.err

source "${WORKSPACE}/psc_slime_env.sh"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
export EXP_NAME="final_report_8h_${SLURM_JOB_ID:-local}_${STAMP}"
EXP_DIR="results/experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}"
ln -sfn "${EXP_NAME}" results/experiments/latest
RUN_LOG="${EXP_DIR}/run.log"
GPU_LOG="${EXP_DIR}/gpu.log"
touch "${RUN_LOG}" "${GPU_LOG}"

{
  echo "===================================================================="
  echo " Slurm job (8h):  ${SLURM_JOB_ID:-N/A}"
  echo " Node:            $(hostname)"
  echo " Date:            $(date)"
  echo " Experiment dir:  ${EXP_DIR}"
  echo " Conda env:       ${CONDA_PREFIX:-unknown}"
  echo "===================================================================="
  nvidia-smi -L || true
  echo "===================================================================="
} | tee -a "${RUN_LOG}"

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

# --- 8h-friendly defaults (override with sbatch --export=ALL,KEY=val,...) ---
# ~34 runs/dataset × ~6–7 min ≈ 3.5–4 h/dataset + post-process; two datasets ≈ 7–8 h.
: "${MODEL_ID:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${DATASETS:=gsm8k,coding}"
: "${ASYNC_MODES:=async_train,async_areal_style}"
: "${RUN_SYNC:=1}"
: "${SEEDS:=0,1}"
: "${STALENESS_K:=0,2,4,8}"
: "${DECOUPLED:=0,1}"
: "${EPOCHS:=1}"
: "${MAX_NEW_TOKENS:=96}"
: "${TRAIN_SIZE:=128}"
: "${EVAL_SIZE:=64}"
: "${TASK_LIMIT:=96}"
: "${LR:=5e-6}"
: "${HF_DTYPE:=float32}"
: "${HF_USE_AUTOCAST:=1}"
: "${UPDATE_BATCH:=4}"
: "${K_HEADLINE:=2}"
: "${PASS_THRESHOLD_GSM8K:=0.10}"
: "${PASS_THRESHOLD_CODING:=0.10}"
export MODEL_ID DATASETS ASYNC_MODES RUN_SYNC SEEDS STALENESS_K DECOUPLED \
       EPOCHS MAX_NEW_TOKENS TRAIN_SIZE EVAL_SIZE TASK_LIMIT LR HF_DTYPE \
       HF_USE_AUTOCAST UPDATE_BATCH K_HEADLINE PASS_THRESHOLD_GSM8K PASS_THRESHOLD_CODING

{
  echo "[config] wall_time=08:00:00  partition=GPU-shared  gpus=v100-32:2"
  echo "[config] MODEL_ID=${MODEL_ID}"
  echo "[config] DATASETS=${DATASETS}  ASYNC_MODES=${ASYNC_MODES}  RUN_SYNC=${RUN_SYNC}"
  echo "[config] SEEDS=${SEEDS}  STALENESS_K=${STALENESS_K}  DECOUPLED=${DECOUPLED}"
  echo "[config] EPOCHS=${EPOCHS}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}  TASK_LIMIT=${TASK_LIMIT}"
  echo "[config] TRAIN_SIZE=${TRAIN_SIZE}  EVAL_SIZE=${EVAL_SIZE}"
  echo "[config] LR=${LR}  HF_DTYPE=${HF_DTYPE}  HF_USE_AUTOCAST=${HF_USE_AUTOCAST}  UPDATE_BATCH=${UPDATE_BATCH}"
  echo "[config] K_HEADLINE=${K_HEADLINE}"
} | tee -a "${RUN_LOG}"

stdbuf -oL -eL \
  bash scripts/run_final_report.sh 2>&1 \
  | tee -a "${RUN_LOG}"

{
  echo "===================================================================="
  echo " Finished at:     $(date)"
  echo " Results dir:     ${EXP_DIR}"
  echo " INDEX:           ${EXP_DIR}/INDEX.md"
  echo "===================================================================="
} | tee -a "${RUN_LOG}"
