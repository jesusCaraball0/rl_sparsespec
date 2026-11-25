#!/bin/bash
#SBATCH --job-name=sparsespec-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/sparsespec-smoke-%j.out
#SBATCH --error=logs/sparsespec-smoke-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda-sparsespec"
export CUDA_HOME="$CONDA_PREFIX"
export INSTALL_PREFIX="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"

# Override inherited CUDA targets after Conda activation, before importing FlashInfer.
TORCH_CUDA_ARCH_LIST=$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')
export TORCH_CUDA_ARCH_LIST

nvidia-smi
srun python -u inference.py
