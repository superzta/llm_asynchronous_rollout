#!/bin/bash
# Submit the paper-reproduction sweep as a Slurm batch job on PSC.
#
# Submit from repo root:
#     cd $PROJECT_ROOT
#     sbatch scripts/submit_paper_repro.sbatch
#
# While running, live-tail on any node (incl. login):
#     tail -f logs/paper_repro_latest.out
#     tail -f results/experiments/latest/run.log
#     tail -f results/experiments/latest/gpu.log
#
# Produces plots:
#     results/experiments/latest/plots/fig5a_naive_learning_curves.png
#     results/experiments/latest/plots/fig5b_decoupled_learning_curves.png
#     results/experiments/latest/plots/fig5c_throughput_vs_staleness.png
#     results/experiments/latest/plots/fig6b_interruptible_throughput.png
#     results/experiments/latest/plots/table2_staleness_vs_pass_rate.{csv,md}
#
#SBATCH --job-name=areal_paper_repro
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/paper_repro_%j.out
#SBATCH --error=logs/paper_repro_%j.err

set -euo pipefail

: "${WORKSPACE:=/ocean/projects/cis260009p/${USER:-tzeng1}}"
PROJECT_ROOT="${WORKSPACE}/repos/llm_async_project"
cd "${PROJECT_ROOT}"

mkdir -p logs results/experiments

ln -sfn "paper_repro_${SLURM_JOB_ID:-local}.out" logs/paper_repro_latest.out
ln -sfn "paper_repro_${SLURM_JOB_ID:-local}.err" logs/paper_repro_latest.err

source "${WORKSPACE}/psc_slime_env.sh"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
export EXP_NAME="paper_repro_${SLURM_JOB_ID:-local}_${STAMP}"
EXP_DIR="results/experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}"
ln -sfn "${EXP_NAME}" results/experiments/latest
RUN_LOG="${EXP_DIR}/run.log"
GPU_LOG="${EXP_DIR}/gpu.log"
touch "${RUN_LOG}" "${GPU_LOG}"

{
  echo "===================================================================="
  echo " Slurm job:          ${SLURM_JOB_ID:-N/A}"
  echo " Node:               $(hostname)"
  echo " Date:               $(date)"
  echo " Experiment dir:     ${EXP_DIR}"
  echo " Conda env:          ${CONDA_PREFIX:-unknown}"
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

# Default sweep (override via --export=ALL,FOO=bar,... when submitting).
: "${MODEL_ID:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${SEEDS:=0,1}"
: "${STALENESS_K:=0,1,2,4,8}"
: "${DECOUPLED:=0,1}"
: "${EPOCHS:=1}"
: "${MAX_NEW_TOKENS:=192}"
: "${TRAIN_SIZE:=128}"
: "${EVAL_SIZE:=64}"
: "${LR:=5e-6}"
: "${HF_DTYPE:=float32}"
: "${UPDATE_BATCH:=4}"
export MODEL_ID SEEDS STALENESS_K DECOUPLED EPOCHS MAX_NEW_TOKENS TRAIN_SIZE \
       EVAL_SIZE LR HF_DTYPE UPDATE_BATCH

{
  echo "[config] MODEL_ID=${MODEL_ID}"
  echo "[config] SEEDS=${SEEDS}  STALENESS_K=${STALENESS_K}  DECOUPLED=${DECOUPLED}"
  echo "[config] EPOCHS=${EPOCHS}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
  echo "[config] TRAIN_SIZE=${TRAIN_SIZE}  EVAL_SIZE=${EVAL_SIZE}"
  echo "[config] LR=${LR}  HF_DTYPE=${HF_DTYPE}  UPDATE_BATCH=${UPDATE_BATCH}"
} | tee -a "${RUN_LOG}"

stdbuf -oL -eL \
  bash scripts/run_paper_repro.sh 2>&1 \
  | tee -a "${RUN_LOG}"

{
  echo "===================================================================="
  echo " Finished at:        $(date)"
  echo " Results dir:        ${EXP_DIR}"
  echo " Plots:              ${EXP_DIR}/plots"
  echo "===================================================================="
} | tee -a "${RUN_LOG}"
