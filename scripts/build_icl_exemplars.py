from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "icl.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def row_to_case(row: pd.Series) -> dict:
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


def build_exemplar_pool(train_df: pd.DataFrame, pool_size_per_class: int) -> list[dict]:
    exemplars = []
    for label_value, label_name in [(1, "YES"), (0, "NO")]:
        class_df = train_df[train_df["diabetes"].astype(int) == label_value].copy()
        class_df["risk_score"] = (
            class_df["HbA1c_level"].astype(float) * 10.0
            + class_df["blood_glucose_level"].astype(float) / 10.0
            + class_df["bmi"].astype(float) / 10.0
        )
        class_df = class_df.sort_values("risk_score", ascending=(label_value == 0))
        selected = class_df.head(pool_size_per_class)
        for _, row in selected.iterrows():
            exemplars.append({
                "case": row_to_case(row),
                "label": label_name,
            })
    return exemplars


def numeric_summary(df: pd.DataFrame) -> dict:
    cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    return {
        col: {
            "mean": round(float(df[col].astype(float).mean()), 4),
            "median": round(float(df[col].astype(float).median()), 4),
            "min": round(float(df[col].astype(float).min()), 4),
            "max": round(float(df[col].astype(float).max()), 4),
        }
        for col in cols
    }


def build_train_context(train_df: pd.DataFrame) -> dict:
    positive_df = train_df[train_df["diabetes"].astype(int) == 1]
    negative_df = train_df[train_df["diabetes"].astype(int) == 0]

    def representative_cases(df: pd.DataFrame, n: int) -> list[dict]:
        selected = df.head(n)
        return [row_to_case(row) for _, row in selected.iterrows()]

    return {
        "train_size": int(len(train_df)),
        "positive_rate": round(float(train_df["diabetes"].astype(float).mean()), 4),
        "positive_summary": numeric_summary(positive_df) if len(positive_df) else {},
        "negative_summary": numeric_summary(negative_df) if len(negative_df) else {},
        "positive_examples": representative_cases(positive_df, 3),
        "negative_examples": representative_cases(negative_df, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ICL exemplar and train-context artifacts from train.csv."
    )
    parser.parse_args()

    config = load_config()
    exemplar_config = config.get("exemplars", {})
    context_config = config.get("train_context", {})

    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    train_df = pd.read_csv(train_path)

    pool_size_per_class = int(exemplar_config.get("pool_size_per_class", 25))
    exemplar_path = PROJECT_ROOT / exemplar_config.get("path", "artifacts/icl/exemplar_set.json")
    train_context_path = PROJECT_ROOT / context_config.get("path", "artifacts/icl/train_context.json")

    exemplars = build_exemplar_pool(train_df, pool_size_per_class)
    train_context = build_train_context(train_df)

    exemplar_path.parent.mkdir(parents=True, exist_ok=True)
    train_context_path.parent.mkdir(parents=True, exist_ok=True)

    exemplar_path.write_text(json.dumps(exemplars, indent=2), encoding="utf-8")
    train_context_path.write_text(json.dumps(train_context, indent=2), encoding="utf-8")

    print(f"Saved exemplar pool to {exemplar_path}")
    print(f"Saved train context to {train_context_path}")
    print(f"Exemplar count: {len(exemplars)}")


if __name__ == "__main__":
    main()
