from __future__ import annotations


def run_batch(model, cases: list[dict]) -> list[dict]:
    return [model.predict_one(case) for case in cases]
