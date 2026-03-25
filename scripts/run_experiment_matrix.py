from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_models.py"


def build_runs() -> list[dict]:
    runs = [
        {
            "name": "ml",
            "model": "ml",
            "args": ["--model", "ml"],
        },
        {
            "name": "rag_heuristic",
            "model": "rag",
            "rag_batch_mode": "heuristic",
            "args": ["--model", "rag", "--rag-batch-mode", "heuristic"],
        },
        {
            "name": "rag_llm",
            "model": "rag",
            "rag_batch_mode": "llm",
            "args": ["--model", "rag", "--rag-batch-mode", "llm"],
        },
    ]

    for icl_mode in ["zero_shot", "few_shot", "train_context"]:
        for icl_batch_mode in ["heuristic", "llm"]:
            runs.append(
                {
                    "name": f"icl_{icl_mode}_{icl_batch_mode}",
                    "model": "icl",
                    "icl_mode": icl_mode,
                    "icl_batch_mode": icl_batch_mode,
                    "args": [
                        "--model",
                        "icl",
                        "--icl-mode",
                        icl_mode,
                        "--icl-batch-mode",
                        icl_batch_mode,
                    ],
                }
            )

    return runs


def combined_output_path(output_dir: Path, split: str, limit: int | None) -> Path:
    if limit is None:
        return output_dir / f"all_experiments_{split}_results.json"
    return output_dir / f"all_experiments_{split}_limit_{limit}_results.json"


def result_path_for_run(output_dir: Path, split: str, run: dict) -> Path:
    if run["model"] == "ml":
        return output_dir / f"ml_{split}_results.json"
    if run["model"] == "rag":
        return output_dir / f"rag_{run['rag_batch_mode']}_{split}_results.json"

    return output_dir / f"icl_{run['icl_mode']}_{run['icl_batch_mode']}_{split}_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full v3 experiment matrix and store each result separately."
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
        "--log-dir",
        default="artifacts/eval/logs",
        help="Directory to store per-run log files.",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = PROJECT_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs()
    combined_results: dict[str, dict] = {}

    for run in runs:
        log_path = log_dir / f"{run['name']}_{args.split}.txt"
        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            *run["args"],
            "--split",
            args.split,
            "--output-dir",
            args.output_dir,
            "--log-file",
            str(log_path.relative_to(PROJECT_ROOT)),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])

        print(f"Running {run['name']}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

        result_path = result_path_for_run(output_dir, args.split, run)
        combined_results[run["name"]] = json.loads(result_path.read_text(encoding="utf-8"))

    combined_path = combined_output_path(output_dir, args.split, args.limit)
    combined_path.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
    print(f"Saved combined matrix results to {combined_path}")


if __name__ == "__main__":
    main()
