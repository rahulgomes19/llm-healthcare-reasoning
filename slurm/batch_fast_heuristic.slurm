#!/bin/bash
#SBATCH --job-name=dc3-batch-fast
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs

echo "Running full-batch scalable evaluation in ${PROJECT_DIR}"
date

python3 scripts/evaluate_models.py --model ml
python3 scripts/evaluate_models.py --model rag --rag-batch-mode heuristic
python3 scripts/evaluate_models.py --model icl --icl-mode few_shot --icl-batch-mode heuristic

date
echo "Full-batch scalable evaluation completed"
