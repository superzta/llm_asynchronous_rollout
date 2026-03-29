#!/bin/bash
set -e

module purge
module load anaconda3/2024.10-1
module load cuda/12.6.1
conda activate /ocean/projects/cis260009p/tzeng1/conda_envs/slime312

export WORKSPACE=/ocean/projects/cis260009p/tzeng1
export HF_HOME=$WORKSPACE/huggingface
export HF_HUB_CACHE=$WORKSPACE/huggingface_cache
export CONDA_PKGS_DIRS=$WORKSPACE/conda_pkgs
export PYTHONPATH=$WORKSPACE/repos/slime:$WORKSPACE/repos/Megatron-LM:$PYTHONPATH

python - <<'PY'
import torch, slime, sglang, ray
print("environment ok")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY