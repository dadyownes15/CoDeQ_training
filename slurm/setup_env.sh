#!/usr/bin/env bash
# ============================================================
# One-time setup on the cluster (run from a login node).
#
# This script:
#   1. Creates a Python venv and installs dependencies
#   2. Downloads CIFAR-10 so compute nodes don't need internet
#   3. Logs into Weights & Biases
#
# Usage:
#   cd /path/to/CoDeQ
#   bash slurm/setup_env.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# -----------------------------------------------------------
# 1. Load modules (adjust to your cluster's module system)
# -----------------------------------------------------------
# Uncomment and adapt for your cluster:
# module purge
# module load python/3.11
# module load cuda/12.1

# -----------------------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Using Python: $(which python3)"

pip install --upgrade pip
pip install -r requirements.txt

# -----------------------------------------------------------
# 3. Download CIFAR-10 dataset (needs internet — login nodes)
# -----------------------------------------------------------
echo "Downloading CIFAR-10 dataset..."
python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print('CIFAR-10 downloaded to ./data/')
"

# -----------------------------------------------------------
# 4. Log into Weights & Biases
# -----------------------------------------------------------
echo ""
echo "=== Weights & Biases Setup ==="
echo "You need a W&B API key. Get one at: https://wandb.ai/authorize"
echo ""
wandb login

echo ""
echo "Setup complete! You can now submit jobs with:"
echo "  sbatch slurm/train.sbatch"
echo "  sbatch slurm/train.sbatch group-lasso-channels-delay50"
