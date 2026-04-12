from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_split
from src.eval.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.eval.report import write_report
from src.models.icl_model import ICLModel
from src.models.ml_model import MLModel
from src.models.rag_model import RAGModel


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def log_summary(log_path: Path, summary: dict) -> None:
    append_log(log_path, "{")
    append_log(log_path, f'  "model": "{summary["model"]}",')
    append_log(log_path, f'  "icl_mode": {json.dumps(summary["icl_mode"])},')
    append_log(log_path, f'  "icl_batch_mode": {json.dumps(summary.get("icl_batch_mode"))},')
    append_log(log_path, f'  "rag_mode": {json.dumps(summary.get("rag_mode"))},')
    append_log(log_path, f'  "rag_batch_mode": {json.dumps(summary.get("rag_batch_mode"))},')
    append_log(log_path, f'  "split": "{summary["split"]}",')
    append_log(log_path, f'  "num_cases": {summary["num_cases"]},')
    append_log(log_path, f'  "fallback_count": {summary.get("fallback_count", 0)},')
    append_log(log_path, f'  "fallback_rate": {summary.get("fallback_rate", 0.0)},')
    append_log(log_path, f'  "parse_failure_count": {summary.get("parse_failure_count", 0)},')
    append_log(log_path, f'  "parse_failure_rate": {summary.get("parse_failure_rate", 0.0)},')
    append_log(log_path, f'  "accuracy": {summary["accuracy"]},')
    append_log(log_path, f'  "precision": {summary["precision"]},')
    append_log(log_path, f'  "recall": {summary["recall"]},')
    append_log(log_path, f'  "f1": {summary["f1"]},')
    append_log(log_path, f'  "confusion_matrix": {json.dumps(summary["confusion_matrix"])},')
    append_log(log_path, f'  "zone_performance": {json.dumps(summary.get("zone_performance"))},')
    append_log(log_path, f'  "retrieval_summary": {json.dumps(summary.get("retrieval_summary"))},')
    append_log(log_path, f'  "avg_runtime_seconds": {summary["avg_runtime_seconds"]},')
    append_log(log_path, f'  "total_runtime_seconds": {summary["total_runtime_seconds"]}')
    append_log(log_path, "}")


def print_summary(summary: dict) -> None:
    print("{")
    print(f'  "model": "{summary["model"]}",')
    print(f'  "icl_mode": {json.dumps(summary["icl_mode"])},')
    print(f'  "icl_batch_mode": {json.dumps(summary.get("icl_batch_mode"))},')
    print(f'  "rag_mode": {json.dumps(summary.get("rag_mode"))},')
    print(f'  "rag_batch_mode": {json.dumps(summary.get("rag_batch_mode"))},')
    print(f'  "split": "{summary["split"]}",')
    print(f'  "num_cases": {summary["num_cases"]},')
    print(f'  "fallback_count": {summary.get("fallback_count", 0)},')
    print(f'  "fallback_rate": {summary.get("fallback_rate", 0.0)},')
    print(f'  "parse_failure_count": {summary.get("parse_failure_count", 0)},')
    print(f'  "parse_failure_rate": {summary.get("parse_failure_rate", 0.0)},')
    print(f'  "accuracy": {summary["accuracy"]},')
    print(f'  "precision": {summary["precision"]},')
    print(f'  "recall": {summary["recall"]},')
    print(f'  "f1": {summary["f1"]},')
    print(f'  "confusion_matrix": {json.dumps(summary["confusion_matrix"])},')
    print(f'  "zone_performance": {json.dumps(summary.get("zone_performance"))},')
    print(f'  "retrieval_summary": {json.dumps(summary.get("retrieval_summary"))},')
    print(f'  "avg_runtime_seconds": {summary["avg_runtime_seconds"]},')
    print(f'  "total_runtime_seconds": {summary["total_runtime_seconds"]}')
    print("}")


def build_model(
    name: str,
    icl_mode: str,
    icl_batch_mode: str,
    rag_llm_model: str | None,
    icl_llm_model: str | None,
):
    if name == "ml":
        return MLModel()
    if name == "rag":
        return RAGModel(model_name=rag_llm_model)
    if name == "icl":
        return ICLModel(mode=icl_mode, batch_mode=icl_batch_mode, model_name=icl_llm_model)
    raise ValueError(f"Unknown model: {name}")


def row_to_case(row: dict) -> dict:
    return {
        "age": float(row["age"]),
        "gender": str(row["gender"]),
        "hypertension": int(float(row["hypertension"])),
        "heart_disease": int(float(row["heart_disease"])),
        "smoking_history": str(row["smoking_history"]),
        "bmi": float(row["bmi"]),
        "HbA1c_level": float(row["HbA1c_level"]),
        "blood_glucose_level": float(row["blood_glucose_level"]),
    }


def label_to_int(label: str) -> int:
    return 1 if str(label).upper() == "YES" else 0


def classify_case_zone(case: dict) -> str:
    hba1c = float(case["HbA1c_level"])
    glucose = float(case["blood_glucose_level"])
    if hba1c >= 6.5 or glucose >= 200:
        return "deterministic_positive"
    if hba1c < 5.7 and glucose < 100:
        return "deterministic_negative"
    return "middle_zone"


def build_zone_summary(zone_name: str, true_labels: list[int], predicted_labels: list[int]) -> dict:
    return {
        "zone": zone_name,
        "num_cases": len(true_labels),
        "accuracy": accuracy_score(true_labels, predicted_labels) if true_labels else 0.0,
        "precision": precision_score(true_labels, predicted_labels) if true_labels else 0.0,
        "recall": recall_score(true_labels, predicted_labels) if true_labels else 0.0,
        "f1": f1_score(true_labels, predicted_labels) if true_labels else 0.0,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels) if true_labels else [[0, 0], [0, 0]],
    }


def evaluate_model(
    model_name: str,
    split: str,
    icl_mode: str,
    icl_batch_mode: str,
    rag_mode: str,
    rag_batch_mode: str,
    rag_llm_model: str | None,
    icl_llm_model: str | None,
    log_path: Path,
    limit: int | None,
) -> dict:
    df = load_processed_split(split)
    if limit is not None:
        df = df.head(limit)
    cases = [row_to_case(row) for row in df.to_dict(orient="records")]
    true_labels = df["diabetes"].astype(int).tolist()

    model = build_model(model_name, icl_mode, icl_batch_mode, rag_llm_model, icl_llm_model)
    predicted_labels = []
    probabilities = []
    runtimes = []
    parse_failure_count = 0
    fallback_count = 0
    zone_true_labels = {
        "deterministic_positive": [],
        "deterministic_negative": [],
        "middle_zone": [],
    }
    zone_predicted_labels = {
        "deterministic_positive": [],
        "deterministic_negative": [],
        "middle_zone": [],
    }
    retrieval_chunk_counts = []
    retrieval_failure_count = 0
    retrieved_source_names: set[str] = set()

    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        start_time = time()
        if model_name == "rag":
            if rag_mode == "heuristic":
                result = model.predict_one_heuristic(case)
            elif rag_mode == "stubbed_chunk_llm":
                result = model.predict_one_stubbed_chunk_llm(case)
            elif rag_mode == "stubbed_full_doc_llm":
                result = model.predict_one_stubbed_full_doc_llm(case)
            else:
                result = model.predict_one(case)
        elif model_name == "icl" and icl_batch_mode == "heuristic":
            result = model.predict_one_heuristic(case)
        else:
            result = model.predict_one(case)
        elapsed = time() - start_time

        predicted_labels.append(label_to_int(result["prediction"]))
        probabilities.append(float(result.get("probability", 0.0)))
        runtimes.append(elapsed)
        if result.get("fallback_used") is True:
            fallback_count += 1
        if result.get("parsed_ok") is False:
            parse_failure_count += 1

        zone_name = classify_case_zone(case)
        zone_true_labels[zone_name].append(true_labels[index - 1])
        zone_predicted_labels[zone_name].append(predicted_labels[-1])

        if model_name == "rag":
            retrieved_sources = result.get("retrieved_sources", [])
            retrieval_chunk_counts.append(len(retrieved_sources))
            if not retrieved_sources:
                retrieval_failure_count += 1
            for item in retrieved_sources:
                source_name = item.get("source")
                if source_name:
                    retrieved_source_names.add(str(source_name))

        if index % 100 == 0:
            append_log(log_path, f"Processed {index} cases")

    append_log(log_path, f"Processed {total_cases} total cases")

    zone_summary = {
        zone_name: build_zone_summary(
            zone_name,
            zone_true_labels[zone_name],
            zone_predicted_labels[zone_name],
        )
        for zone_name in zone_true_labels
    }

    summary = {
        "model": model_name,
        "icl_mode": icl_mode if model_name == "icl" else None,
        "icl_batch_mode": icl_batch_mode if model_name == "icl" else None,
        "rag_mode": rag_mode if model_name == "rag" else None,
        "rag_batch_mode": rag_batch_mode if model_name == "rag" else None,
        "split": split,
        "num_cases": len(cases),
        "fallback_count": fallback_count,
        "fallback_rate": (fallback_count / len(cases)) if cases else 0.0,
        "parse_failure_count": parse_failure_count,
        "parse_failure_rate": (parse_failure_count / len(cases)) if cases else 0.0,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels),
        "recall": recall_score(true_labels, predicted_labels),
        "f1": f1_score(true_labels, predicted_labels),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels),
        "zone_performance": zone_summary,
        "retrieval_summary": (
            {
                "retrieval_failure_count": retrieval_failure_count,
                "retrieval_failure_rate": (retrieval_failure_count / len(cases)) if cases else 0.0,
                "avg_retrieved_chunks": (
                    statistics.mean(retrieval_chunk_counts) if retrieval_chunk_counts else 0.0
                ),
                "max_retrieved_chunks": max(retrieval_chunk_counts) if retrieval_chunk_counts else 0,
                "unique_source_count": len(retrieved_source_names),
                "unique_sources": sorted(retrieved_source_names),
            }
            if model_name == "rag"
            else None
        ),
        "avg_runtime_seconds": statistics.mean(runtimes) if runtimes else 0.0,
        "total_runtime_seconds": sum(runtimes),
        "predictions": predicted_labels,
        "probabilities": probabilities,
        "true_labels": true_labels,
    }
    return summary


def output_path_for(
    model_name: str,
    split: str,
    icl_mode: str,
    icl_batch_mode: str,
    rag_mode: str,
    rag_batch_mode: str,
    output_dir: Path,
) -> Path:
    if model_name == "icl":
        return output_dir / f"icl_{icl_mode}_{icl_batch_mode}_{split}_results.json"
    if model_name == "rag":
        return output_dir / f"rag_{rag_mode}_{split}_results.json"
    return output_dir / f"{model_name}_{split}_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more models on the full processed dataset split."
    )
    parser.add_argument("--model", choices=["ml", "rag", "icl", "all"], required=True)
    parser.add_argument(
        "--icl-mode",
        choices=["zero_shot", "few_shot", "train_context"],
        default="few_shot",
        help="ICL sub-mode to evaluate when --model icl or --model all is used.",
    )
    parser.add_argument(
        "--rag-mode",
        choices=["llm", "heuristic", "stubbed_chunk_llm", "stubbed_full_doc_llm"],
        default=None,
        help="RAG experimental mode. If omitted, the legacy --rag-batch-mode value is used.",
    )
    parser.add_argument(
        "--rag-batch-mode",
        choices=["llm", "heuristic"],
        default="heuristic",
        help="RAG batch mode: 'llm' uses one Ollama call per case; 'heuristic' uses retrieved context plus deterministic thresholds.",
    )
    parser.add_argument(
        "--icl-batch-mode",
        choices=["llm", "heuristic"],
        default="heuristic",
        help="ICL batch mode: 'llm' uses one Ollama call per case; 'heuristic' uses non-generative fallback logic per mode.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Processed split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to evaluate from the chosen split.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eval",
        help="Directory to store JSON evaluation reports.",
    )
    parser.add_argument(
        "--rag-llm-model",
        default=None,
        help="Optional override for the Ollama model used by RAG LLM mode.",
    )
    parser.add_argument(
        "--icl-llm-model",
        default=None,
        help="Optional override for the Ollama model used by ICL LLM mode.",
    )
    parser.add_argument(
        "--log-file",
        default="artifacts/eval/evaluation_log.txt",
        help="Rewritable text log file for progress and metric summaries.",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    model_names = ["ml", "rag", "icl"] if args.model == "all" else [args.model]
    summaries = {}
    rag_mode = args.rag_mode or args.rag_batch_mode

    for model_name in model_names:
        append_log(log_path, f"Starting {model_name} evaluation")
        summary = evaluate_model(
            model_name,
            args.split,
            args.icl_mode,
            args.icl_batch_mode,
            rag_mode,
            args.rag_batch_mode,
            args.rag_llm_model,
            args.icl_llm_model,
            log_path,
            args.limit,
        )
        summaries[model_name] = summary
        path = output_path_for(
            model_name,
            args.split,
            args.icl_mode,
            args.icl_batch_mode,
            rag_mode,
            args.rag_batch_mode,
            output_dir,
        )
        write_report(summary, str(path))
        append_log(log_path, f"Saved {model_name} evaluation to {path}")
        log_summary(log_path, summary)
        print(f"Saved {model_name} evaluation to {path}")
        print(f"Progress and summaries are also being written to {log_path}")
        print_summary(summary)

    if args.model == "all":
        combined_path = output_dir / f"all_{args.split}_results.json"
        write_report(summaries, str(combined_path))
        append_log(log_path, f"Saved combined evaluation to {combined_path}")
        print(f"Saved combined evaluation to {combined_path}")


if __name__ == "__main__":
    main()
