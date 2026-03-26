#!/bin/bash
#SBATCH --job-name=dc3-batch-llm
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs
mkdir -p artifacts/eval/logs

LIMIT="${LIMIT:-100}"

echo "Running limited LLM-backed evaluation in ${PROJECT_DIR}"
echo "Subset limit: ${LIMIT}"
date

python3 scripts/evaluate_models.py --model rag --rag-batch-mode llm --limit "${LIMIT}" --log-file "artifacts/eval/logs/rag_llm_limit_${LIMIT}_${SLURM_JOB_ID}.txt"
python3 scripts/evaluate_models.py --model icl --icl-mode few_shot --icl-batch-mode llm --limit "${LIMIT}" --log-file "artifacts/eval/logs/icl_few_shot_llm_limit_${LIMIT}_${SLURM_JOB_ID}.txt"

date
echo "Limited LLM-backed evaluation completed"
