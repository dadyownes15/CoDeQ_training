#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=6:00:00

set -euo pipefail

# node/GPU layout
hostname
nvidia-smi

# Load modules & conda
module load miniconda
module load cuda

conda_environment="qe_max_experiment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$conda_environment"

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"

echo "Starting training..."

python ../run_training.py --config ../configs/deadzone_vit.yaml --device cuda
