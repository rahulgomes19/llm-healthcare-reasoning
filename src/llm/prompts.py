from __future__ import annotations


def _format_case(case: dict) -> str:
    return "\n".join([
        f"- Age: {case.get('age')}",
        f"- Gender: {case.get('gender')}",
        f"- BMI: {case.get('bmi')}",
        f"- HbA1c_level: {case.get('HbA1c_level')}",
        f"- blood_glucose_level (current/random glucose): {case.get('blood_glucose_level')}",
        f"- Hypertension: {case.get('hypertension')}",
        f"- Heart disease: {case.get('heart_disease')}",
        f"- Smoking history: {case.get('smoking_history')}",
    ])


def _shared_instruction_block() -> str:
    return """Role:
You are a clinical decision support assistant for diabetes classification.

Task:
Determine whether the patient should be classified as having diabetes (YES or NO).

Reasoning requirements:
- Reason carefully through the diagnostic markers first, especially HbA1c and blood glucose.
- Consider age, BMI, hypertension, heart disease, and smoking history as supporting risk factors, especially when the major markers are borderline.
- Treat blood_glucose_level as a current/random glucose measurement for interpretation purposes.
- Do not assume unsupported facts.
- Produce a concise final answer only.

Output schema:
Return exactly these four lines:
PREDICTION: YES or NO
CONFIDENCE: 0.0-1.0
RATIONALE: one short sentence
EVIDENCE: one short sentence"""


def build_rag_prompt(case: dict, context: str) -> str:
    return f"""{_shared_instruction_block()}

RAG-specific requirements:
- Use the retrieved guideline context as the primary evidence source.
- Ground the decision in the provided context instead of relying on outside medical knowledge.
- In the EVIDENCE line, mention the most relevant diagnostic criterion or threshold from the context.
- If the retrieved context is incomplete or ambiguous, say so briefly in the RATIONALE line while still returning a best classification.

Retrieved guideline context:
{context}

Case:
{_format_case(case)}
"""


def build_icl_prompt(case: dict, exemplars: list[dict]) -> str:
    shots = []
    for ex in exemplars:
        shots.append(f"Case:\n{_format_case(ex['case'])}\nLabel: {ex['label']}")
    joined = "\n\n".join(shots)
    return f"""{_shared_instruction_block()}

ICL-specific requirements:
- Use the labeled examples as in-context guidance, but do not copy labels mechanically.
- Compare the new case to the examples using HbA1c, blood glucose, BMI, age, hypertension, and heart disease.
- In the RATIONALE line, briefly explain the most important factors driving the classification.
- In the EVIDENCE line, mention whether the decision was most influenced by the examples or by the patient's diagnostic markers.

Labeled examples:

{joined}

Now classify this case:
{_format_case(case)}"""


def build_icl_zero_shot_prompt(case: dict) -> str:
    return f"""{_shared_instruction_block()}

Zero-shot requirements:
- Base the decision on the structured patient data only.
- Give greatest weight to HbA1c and blood glucose, then use age, BMI, hypertension, heart disease, and smoking history as supporting context.
- In the RATIONALE line, briefly explain the decision using the most relevant clinical features.

Case:
{_format_case(case)}"""


def build_icl_train_context_prompt(case: dict, train_context: dict) -> str:
    return f"""{_shared_instruction_block()}

Train-context requirements:
- Use the training-derived summary as supporting context rather than as a hard rule.
- Weigh the case against the summary patterns while still prioritizing diagnostic markers such as HbA1c and blood glucose.
- In the RATIONALE line, mention whether the case aligns with the positive or negative training pattern.

Training context:
{train_context}

Case:
{_format_case(case)}"""
