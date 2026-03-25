#!/bin/bash
#SBATCH --job-name=dc3-smoke
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs

echo "Running single-record smoke tests in ${PROJECT_DIR}"
date

python3 scripts/run_experiment.py --model ml
python3 scripts/run_experiment.py --model rag
python3 scripts/run_experiment.py --model icl --icl-mode few_shot

date
echo "Single-record smoke tests completed"
