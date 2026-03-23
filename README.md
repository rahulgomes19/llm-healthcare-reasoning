# diabetes_coach_v3

This folder is a cleaner experimental framework derived from `diabetes_coach_v2`.

Included paradigms:
- ML: supervised structured-feature model
- RAG: retrieval over diabetes guideline documents
- ICL: few-shot exemplar prompting baseline

Main entry point:
- `python scripts/run_experiment.py --model ml`
- `python scripts/run_experiment.py --model rag`
- `python scripts/run_experiment.py --model icl`
- `python scripts/run_experiment.py --model all`

ML workflow:
- `python scripts/train_ml.py`
- `python scripts/run_experiment.py --model ml`

The ML inference path does not train automatically. Training is a separate step.

ICL workflow:
- `python scripts/build_icl_exemplars.py`
- `python scripts/run_experiment.py --model icl --icl-mode zero_shot`
- `python scripts/run_experiment.py --model icl --icl-mode few_shot`
- `python scripts/run_experiment.py --model icl --icl-mode train_context`
