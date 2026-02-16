#!/usr/bin/env bash
#SBATCH --job-name=codeq
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
# Uncomment and set your partition / account:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# ============================================================
# SLURM job script for CoDeQ training
#
# Usage:
#   sbatch slurm/train.sbatch                              # baseline
#   sbatch slurm/train.sbatch group-lasso-channels-delay50
#   sbatch slurm/train.sbatch coupled-group-lasso-delay50
#
# List experiments:
#   python -m experiments.run --list
# ============================================================
set -euo pipefail

EXPERIMENT="${1:-baseline-no-sparsity}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# -----------------------------------------------------------
# Load modules (adjust for your cluster)
# -----------------------------------------------------------
# module purge
# module load python/3.11
# module load cuda/12.1

# -----------------------------------------------------------
# Activate environment
# -----------------------------------------------------------
source .venv/bin/activate

# -----------------------------------------------------------
# Verify GPU is available
# -----------------------------------------------------------
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# -----------------------------------------------------------
# Set W&B to offline mode if compute nodes lack internet.
# Sync later with: wandb sync slurm/wandb_runs/run-*
# -----------------------------------------------------------
# Uncomment the next two lines if compute nodes have no internet:
# export WANDB_MODE=offline
# export WANDB_DIR="$PROJECT_DIR/slurm/wandb_runs"
# mkdir -p "$WANDB_DIR"

# -----------------------------------------------------------
# Run experiment
# -----------------------------------------------------------
echo "============================================"
echo "Experiment: $EXPERIMENT"
echo "Node:       $(hostname)"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Date:       $(date)"
echo "============================================"

python3 -m experiments.run "$EXPERIMENT"

echo "Job finished at $(date)"
