#!/usr/bin/env bash
# ============================================================
# Submit all 4 CoDeQ experiment variants as separate SLURM jobs.
#
# Experiments (all 300 epochs, ResNet-20 on CIFAR-10):
#   1. baseline                    — no quantization, plain ResNet-20
#   2. codeq-paper                 — CoDeQ with L2 dead-zone regularization only
#   3. coupled-group-lasso         — CoDeQ + coupled group lasso (no delay)
#   4. coupled-group-lasso-delay50 — CoDeQ + coupled group lasso (delay=50, ramp=50)
#
# Usage:
#   bash slurm/batch_all.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.sbatch"

EXPERIMENTS=(
    "baseline"
    "codeq-paper"
    "coupled-group-lasso"
    "coupled-group-lasso-delay50"
)

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting ${#EXPERIMENTS[@]} CoDeQ experiments..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    job_id=$(sbatch --job-name="codeq-${exp}" "$TRAIN_SCRIPT" "$exp" | awk '{print $4}')
    echo "  Submitted: $exp  (job $job_id)"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
