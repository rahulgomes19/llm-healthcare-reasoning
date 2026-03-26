#!/bin/bash
#SBATCH --job-name=dc3-matrix
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs
mkdir -p artifacts/eval/logs

OUTPUT_DIR="artifacts/eval/matrix_${SLURM_JOB_ID}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

echo "Running full experiment matrix in ${PROJECT_DIR}"
if [[ -n "${LIMIT:-}" ]]; then
  echo "Subset limit: ${LIMIT}"
fi
date

python3 scripts/run_experiment_matrix.py --split test --output-dir "${OUTPUT_DIR}" --log-dir "artifacts/eval/logs/matrix_${SLURM_JOB_ID}" "${LIMIT_ARG[@]}"

date
echo "Experiment matrix run completed"
