from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.icl_model import ICLModel
from src.models.ml_model import MLModel
from src.models.rag_model import RAGModel
from src.utils.timing import timed_call


def load_case(case_arg: str | None, case_file: str | None) -> dict:
    if case_file:
        return json.loads(Path(case_file).read_text())
    if case_arg:
        return json.loads(case_arg)
    return {
        'age': 55,
        'gender': 'male',
        'hypertension': 1,
        'heart_disease': 0,
        'smoking_history': 'former',
        'bmi': 31.4,
        'HbA1c_level': 7.1,
        'blood_glucose_level': 168,
    }


def build_model(name: str):
    if name == 'ml':
        return MLModel()
    if name == 'rag':
        return RAGModel()
    if name == 'icl':
        return ICLModel()
    raise ValueError(f'Unknown model: {name}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ml', 'rag', 'icl', 'all'], required=True)
    parser.add_argument(
        '--icl-mode',
        choices=['zero_shot', 'few_shot', 'train_context'],
        default='few_shot',
        help='ICL sub-mode to run when --model icl or --model all is used.',
    )
    parser.add_argument('--case', help='JSON string for one patient case')
    parser.add_argument('--case-file', help='Path to JSON file for one patient case')
    args = parser.parse_args()

    case = load_case(args.case, args.case_file)

    if args.model == 'all':
        results = {}
        total_runtime = 0.0
        for model_name in ['ml', 'rag', 'icl']:
            model = ICLModel(mode=args.icl_mode) if model_name == 'icl' else build_model(model_name)
            result, elapsed = timed_call(model.predict_one, case)
            result['runtime_seconds'] = elapsed
            total_runtime += elapsed
            results[model_name] = result

        print(json.dumps({
            'case': case,
            'models': results,
            'total_runtime_seconds': total_runtime,
        }, indent=2))
        return

    model = ICLModel(mode=args.icl_mode) if args.model == 'icl' else build_model(args.model)
    result, elapsed = timed_call(model.predict_one, case)
    result['runtime_seconds'] = elapsed
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
