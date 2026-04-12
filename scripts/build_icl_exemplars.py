from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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


def _boundary_score(df: pd.DataFrame) -> pd.Series:
    hba1c_gap = (df["HbA1c_level"].astype(float) - 6.5).abs()
    glucose_gap = (df["blood_glucose_level"].astype(float) - 200.0).abs() / 25.0
    return np.minimum(hba1c_gap.to_numpy(), glucose_gap.to_numpy())


def _severity_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["HbA1c_level"].astype(float) * 10.0
        + df["blood_glucose_level"].astype(float) / 10.0
        + df["bmi"].astype(float) / 10.0
    )


def _typical_distance(df: pd.DataFrame) -> pd.Series:
    cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    medians = df[cols].astype(float).median()
    distances = []
    for col in cols:
        scale = float(df[col].astype(float).std())
        if scale == 0 or np.isnan(scale):
            scale = 1.0
        distances.append((df[col].astype(float) - medians[col]).abs() / scale)
    return sum(distances)


def _allocate_bucket_counts(pool_size_per_class: int) -> dict[str, int]:
    base = {
        "boundary": int(round(pool_size_per_class * 0.4)),
        "typical": int(round(pool_size_per_class * 0.4)),
        "extreme": 0,
    }
    base["extreme"] = pool_size_per_class - base["boundary"] - base["typical"]
    return base


def _select_bucket(
    class_df: pd.DataFrame,
    already_selected: set[int],
    sort_columns: list[str],
    ascending: list[bool],
    count: int,
    bucket_name: str,
    label_name: str,
) -> list[dict]:
    if count <= 0:
        return []

    available = class_df.loc[~class_df.index.isin(already_selected)].copy()
    if available.empty:
        return []

    selected = available.sort_values(sort_columns, ascending=ascending).head(count)
    output = []
    for idx, row in selected.iterrows():
        already_selected.add(int(idx))
        output.append({
            "case": row_to_case(row),
            "label": label_name,
            "selection_bucket": bucket_name,
        })
    return output


def build_exemplar_pool(train_df: pd.DataFrame, pool_size_per_class: int) -> list[dict]:
    exemplars = []
    for label_value, label_name in [(1, "YES"), (0, "NO")]:
        class_df = train_df[train_df["diabetes"].astype(int) == label_value].copy()
        class_df["boundary_score"] = _boundary_score(class_df)
        class_df["severity_score"] = _severity_score(class_df)
        class_df["typical_distance"] = _typical_distance(class_df)

        bucket_counts = _allocate_bucket_counts(pool_size_per_class)
        selected_indices: set[int] = set()

        exemplars.extend(
            _select_bucket(
                class_df,
                selected_indices,
                ["boundary_score", "typical_distance"],
                [True, True],
                bucket_counts["boundary"],
                "boundary",
                label_name,
            )
        )
        exemplars.extend(
            _select_bucket(
                class_df,
                selected_indices,
                ["typical_distance", "boundary_score"],
                [True, True],
                bucket_counts["typical"],
                "typical",
                label_name,
            )
        )
        exemplars.extend(
            _select_bucket(
                class_df,
                selected_indices,
                ["severity_score", "boundary_score"],
                [label_value == 0, False],
                bucket_counts["extreme"],
                "extreme",
                label_name,
            )
        )

        remaining = pool_size_per_class - sum(1 for ex in exemplars if ex["label"] == label_name)
        if remaining > 0:
            exemplars.extend(
                _select_bucket(
                    class_df,
                    selected_indices,
                    ["typical_distance", "boundary_score"],
                    [True, True],
                    remaining,
                    "fill",
                    label_name,
                )
            )
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
        if df.empty or n <= 0:
            return []

        scored = df.copy()
        scored["boundary_score"] = _boundary_score(scored)
        scored["severity_score"] = _severity_score(scored)
        scored["typical_distance"] = _typical_distance(scored)
        label_value = int(scored["diabetes"].astype(int).iloc[0])

        selected_indices: set[int] = set()
        selected_cases: list[dict] = []

        bucket_plan = []
        if n >= 1:
            bucket_plan.append(("boundary", ["boundary_score", "typical_distance"], [True, True], 1))
        if n >= 2:
            bucket_plan.append(("typical", ["typical_distance", "boundary_score"], [True, True], 1))
        if n >= 3:
            bucket_plan.append(
                ("extreme", ["severity_score", "boundary_score"], [label_value == 0, False], 1)
            )

        for bucket_name, sort_columns, ascending, count in bucket_plan:
            bucket_cases = _select_bucket(
                scored,
                selected_indices,
                sort_columns,
                ascending,
                count,
                bucket_name,
                "YES" if label_value == 1 else "NO",
            )
            selected_cases.extend([case_entry["case"] for case_entry in bucket_cases])

        remaining = n - len(selected_cases)
        if remaining > 0:
            fill_cases = _select_bucket(
                scored,
                selected_indices,
                ["typical_distance", "boundary_score"],
                [True, True],
                remaining,
                "fill",
                "YES" if label_value == 1 else "NO",
            )
            selected_cases.extend([case_entry["case"] for case_entry in fill_cases])

        return selected_cases[:n]

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
