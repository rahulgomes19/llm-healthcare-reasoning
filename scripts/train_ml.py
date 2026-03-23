from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import encode_features, impute_numeric
from src.data.splitter import apply_split, save_split, stratified_split_indices
from src.models.numpy_nn import NumpyNeuralNetwork


class ManualMinMaxScaler:
    def __init__(self) -> None:
        self.min_ = None
        self.max_ = None

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.min_ = np.min(values, axis=0)
        self.max_ = np.max(values, axis=0)
        return self.transform(values)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler has not been fitted.")
        ranges = self.max_ - self.min_
        ranges[ranges == 0] = 1.0
        return (values - self.min_) / ranges

    def to_dict(self) -> dict:
        return {
            "type": "ManualMinMaxScaler",
            "min": self.min_.tolist(),
            "max": self.max_.tolist(),
        }


def load_ml_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "ml.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_data(test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    raw_path = PROJECT_ROOT / "data" / "raw" / "diabetes_prediction_dataset.csv"
    df = pd.read_csv(raw_path)
    split = stratified_split_indices(df["diabetes"].values, test_size=test_size, random_state=42)
    save_split(split, str(PROJECT_ROOT / "data" / "processed" / "split_indices.json"))
    train_df, test_df = apply_split(df, split)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    processed = impute_numeric(encode_features(df))
    feature_columns = processed.drop(columns=["diabetes"]).columns.tolist()
    x = processed[feature_columns].astype("float32").to_numpy()
    y = processed["diabetes"].astype("float32").to_numpy()
    return x, y, feature_columns


def accuracy_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_pred = (y_prob >= 0.5).astype(int)
    return float(np.mean(y_pred == y_true.astype(int)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ML model and write artifacts into artifacts/ml."
    )
    parser.parse_args()

    config = load_ml_config()
    model_config = config.get("model", {})
    train_config = config.get("train", {})
    artifact_config = config.get("artifacts", {})

    train_df, test_df = load_split_data(test_size=float(train_config.get("test_size", 0.30)))

    x_train, y_train, feature_columns = prepare_features(train_df)
    x_test, y_test, _ = prepare_features(test_df)

    scaler = ManualMinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    model = NumpyNeuralNetwork(
        input_dim=x_train.shape[1],
        hidden1=int(model_config.get("hidden1", 64)),
        hidden2=int(model_config.get("hidden2", 32)),
        learning_rate=float(model_config.get("learning_rate", 0.001)),
        random_state=42,
    )

    history = model.fit(
        x_train,
        y_train,
        X_val=x_test,
        y_val=y_test,
        epochs=int(train_config.get("epochs", 200)),
        batch_size=int(train_config.get("batch_size", 256)),
        patience=int(train_config.get("patience", 20)),
        verbose=True,
    )

    train_prob = model.predict(x_train).flatten()
    test_prob = model.predict(x_test).flatten()

    metrics = {
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_accuracy": accuracy_score(y_train, train_prob),
        "test_accuracy": accuracy_score(y_test, test_prob),
        "feature_columns": feature_columns,
        "history": {
            "loss": history.get("loss", []),
            "val_loss": history.get("val_loss", []),
            "accuracy": history.get("accuracy", []),
            "val_accuracy": history.get("val_accuracy", []),
        },
    }

    model_path = PROJECT_ROOT / artifact_config.get("model_path", "artifacts/ml/model.pkl")
    scaler_path = PROJECT_ROOT / artifact_config.get("scaler_path", "artifacts/ml/scaler.pkl")
    metrics_path = PROJECT_ROOT / artifact_config.get("metrics_path", "artifacts/ml/metrics.json")
    feature_path = PROJECT_ROOT / "data" / "processed" / "feature_columns.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler.to_dict(), f)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    print("Saved ML artifacts:")
    print(f"  model: {model_path}")
    print(f"  scaler: {scaler_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  feature columns: {feature_path}")
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
