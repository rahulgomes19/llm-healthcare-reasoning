from __future__ import annotations


def build_rag_prompt(case: dict, context: str) -> str:
    return f"""Use the guideline context to answer whether this patient has diabetes.

Context:
{context}

Case:
- Age: {case.get('age')}
- Gender: {case.get('gender')}
- BMI: {case.get('bmi')}
- HbA1c: {case.get('HbA1c_level')}
- Blood glucose: {case.get('blood_glucose_level')}
- Hypertension: {case.get('hypertension')}
- Heart disease: {case.get('heart_disease')}
- Smoking history: {case.get('smoking_history')}

Respond in exactly this format:
PREDICTION: YES or NO
CONFIDENCE: 0.0-1.0"""


def build_icl_prompt(case: dict, exemplars: list[dict]) -> str:
    shots = []
    for ex in exemplars:
        shots.append(f"Case: {ex['case']}\nLabel: {ex['label']}")
    joined = "\n\n".join(shots)
    return f"""Here are labeled examples:

{joined}

Now classify this case:
{case}

Respond in exactly this format:
PREDICTION: YES or NO
CONFIDENCE: 0.0-1.0"""


def build_icl_zero_shot_prompt(case: dict) -> str:
    return f"""Classify whether this patient has diabetes based on the structured patient data.

Case:
{case}

Respond in exactly this format:
PREDICTION: YES or NO
CONFIDENCE: 0.0-1.0"""


def build_icl_train_context_prompt(case: dict, train_context: dict) -> str:
    return f"""Use the following training-derived summary to classify whether this patient has diabetes.

Training context:
{train_context}

Case:
{case}

Respond in exactly this format:
PREDICTION: YES or NO
CONFIDENCE: 0.0-1.0"""
