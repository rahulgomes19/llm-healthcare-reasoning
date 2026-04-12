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

RAG_LLM_MODEL="${RAG_LLM_MODEL:-${LLM_MODEL:-}}"
ICL_LLM_MODEL="${ICL_LLM_MODEL:-${LLM_MODEL:-}}"

if [[ -n "${RAG_LLM_MODEL}" && -n "${ICL_LLM_MODEL}" && "${RAG_LLM_MODEL}" == "${ICL_LLM_MODEL}" ]]; then
  MODEL_LABEL="${RAG_LLM_MODEL}"
elif [[ -n "${RAG_LLM_MODEL}" || -n "${ICL_LLM_MODEL}" ]]; then
  MODEL_LABEL="rag_${RAG_LLM_MODEL:-default}__icl_${ICL_LLM_MODEL:-default}"
else
  MODEL_LABEL="config_default_models"
fi

SANITIZED_LABEL="$(printf '%s' "${MODEL_LABEL}" | tr '/: ' '___')"
OUTPUT_DIR="artifacts/eval/by_llm/${SANITIZED_LABEL}/matrix_${SLURM_JOB_ID}"
LOG_DIR="artifacts/eval/logs/by_llm/${SANITIZED_LABEL}/matrix_${SLURM_JOB_ID}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

MODEL_ARG=()
if [[ -n "${RAG_LLM_MODEL}" ]]; then
  MODEL_ARG+=(--rag-llm-model "${RAG_LLM_MODEL}")
fi
if [[ -n "${ICL_LLM_MODEL}" ]]; then
  MODEL_ARG+=(--icl-llm-model "${ICL_LLM_MODEL}")
fi

echo "Running full experiment matrix in ${PROJECT_DIR}"
if [[ -n "${LIMIT:-}" ]]; then
  echo "Subset limit: ${LIMIT}"
fi
echo "Model label: ${MODEL_LABEL}"
if [[ -n "${RAG_LLM_MODEL}" ]]; then
  echo "RAG LLM model: ${RAG_LLM_MODEL}"
fi
if [[ -n "${ICL_LLM_MODEL}" ]]; then
  echo "ICL LLM model: ${ICL_LLM_MODEL}"
fi
date

python3 scripts/run_experiment_matrix.py --split test --output-dir "${OUTPUT_DIR}" --log-dir "${LOG_DIR}" "${MODEL_ARG[@]}" "${LIMIT_ARG[@]}"

date
echo "Experiment matrix run completed"
