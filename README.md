# diabetes_coach_v3

This folder is a cleaner experimental framework derived from `diabetes_coach_v2`.

Included paradigms:
- ML: supervised structured-feature model
- RAG: retrieval over diabetes guideline documents
- ICL: few-shot exemplar prompting baseline

Single-record smoke tests:
- Use [run_experiment.py](scripts/run_experiment.py) when you want a fast check on one patient record.
- This is the main entry point for quick testing, debugging prompts, and verifying that a model runs end-to-end.
- `python scripts/run_experiment.py --model ml`
  Runs one patient through the trained ML model.
- `python scripts/run_experiment.py --model rag`
  Runs one patient through true RAG with retrieved guideline context and an LLM call.
- `python scripts/run_experiment.py --model icl --icl-mode zero_shot`
  Runs one patient through zero-shot ICL.
- `python scripts/run_experiment.py --model icl --icl-mode few_shot`
  Runs one patient through few-shot ICL with exemplar examples.
- `python scripts/run_experiment.py --model icl --icl-mode train_context`
  Runs one patient through ICL using training-derived summary context.
- `python scripts/run_experiment.py --model all`
  Runs one patient through `ml`, `rag`, and one selected `icl` mode for quick comparison.

ML workflow:
- Use [train_ml.py](scripts/train_ml.py) when you want to retrain the supervised model and overwrite the files in `artifacts/ml`.
- `python scripts/train_ml.py`
  Trains the ML model and writes updated ML artifacts.
- `python scripts/run_experiment.py --model ml`
  Uses the already trained model for inference only.
- The ML inference path does not train automatically.

ICL artifact workflow:
- Use [build_icl_exemplars.py](scripts/build_icl_exemplars.py) when you want to rebuild the exemplar pool and train-context artifacts from `train.csv`.
- `python scripts/build_icl_exemplars.py`
  Rebuilds `artifacts/icl/exemplar_set.json` and `artifacts/icl/train_context.json`.

RAG index workflow:
- Use [build_rag_index.py](scripts/build_rag_index.py) when guideline documents change or when the RAG index needs to be rebuilt.
- `python scripts/build_rag_index.py`
  Rebuilds the vector index used by true RAG.
- If true RAG retrieval or the LLM call fails, the code returns a warning and explicitly reports that fallback logic was used.

Batch evaluation:
- Use [evaluate_models.py](scripts/evaluate_models.py) when you want metrics over `train.csv` or `test.csv` instead of one patient.
- This writes JSON reports, confusion matrices, and timing summaries.
- `python scripts/evaluate_models.py --model ml`
  Evaluates the ML model on the full selected split.
- `python scripts/evaluate_models.py --model rag --rag-batch-mode heuristic`
  Evaluates RAG in scalable heuristic mode on the full selected split.
- `python scripts/evaluate_models.py --model rag --rag-batch-mode llm`
  Evaluates true LLM-backed RAG on the selected split. This is high-cost and usually better with `--limit`.
- `python scripts/evaluate_models.py --model icl --icl-mode zero_shot --icl-batch-mode heuristic`
  Evaluates zero-shot ICL with heuristic batch logic.
- `python scripts/evaluate_models.py --model icl --icl-mode zero_shot --icl-batch-mode llm --limit 100`
  Evaluates zero-shot ICL with the LLM on a smaller subset.
- `python scripts/evaluate_models.py --model icl --icl-mode few_shot --icl-batch-mode heuristic`
  Evaluates few-shot ICL with exemplar-voting style batch logic.
- `python scripts/evaluate_models.py --model icl --icl-mode few_shot --icl-batch-mode llm --limit 100`
  Evaluates few-shot ICL with the LLM on a smaller subset.
- `python scripts/evaluate_models.py --model icl --icl-mode train_context --icl-batch-mode heuristic`
  Evaluates train-context ICL with heuristic batch logic.
- `python scripts/evaluate_models.py --model icl --icl-mode train_context --icl-batch-mode llm --limit 100`
  Evaluates train-context ICL with the LLM on a smaller subset.
- `python scripts/evaluate_models.py --model all --icl-mode train_context`
  Runs one shared batch configuration for `ml`, `rag`, and one chosen ICL mode. This is not the full matrix.

Full experiment matrix:
- Use [run_experiment_matrix.py](scripts/run_experiment_matrix.py) when you want every supported model/mode combination saved separately.
- `python scripts/run_experiment_matrix.py`
  Runs the full matrix on the selected split.
- `python scripts/run_experiment_matrix.py --limit 100`
  Runs the full matrix on a smaller subset for a faster check.

Slurm jobs:
- `sbatch slurm/smoke_single_record.slurm`
  Single-record smoke tests for `ml`, `rag`, and `icl few_shot`.
- `sbatch slurm/batch_fast_heuristic.slurm`
  Full-batch scalable evaluation using `ml`, `rag heuristic`, and `icl few_shot heuristic`.
- `LIMIT=100 sbatch slurm/batch_llm_subset.slurm`
  Smaller high-fidelity LLM subset run for `rag llm` and `icl few_shot llm`.
- `LIMIT=100 sbatch slurm/batch_matrix_all.slurm`
  Full experiment matrix as a Slurm job, usually best with a subset limit first.
