from __future__ import annotations

from src.utils.timing import timed_call


def run_batch(model, cases: list[dict]) -> list[dict]:
    return [model.predict_one(case) for case in cases]


def evaluate_batch(model, cases: list[dict]) -> tuple[list[dict], list[float]]:
    results = []
    runtimes = []
    total = len(cases)

    for index, case in enumerate(cases, start=1):
        result, elapsed = timed_call(model.predict_one, case)
        result["runtime_seconds"] = elapsed
        results.append(result)
        runtimes.append(elapsed)

        if index % 100 == 0:
            print(f"Processed {index} cases")

    print(f"Processed {total} total cases")

    return results, runtimes
