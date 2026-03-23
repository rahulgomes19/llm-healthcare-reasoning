from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.data.preprocess import apply_manual_scaler, case_to_feature_row
from src.models.numpy_nn import NumpyNeuralNetwork


class MLModel:
    def __init__(
        self,
        model_path: str = 'artifacts/ml/model.pkl',
        scaler_path: str = 'artifacts/ml/scaler.pkl',
        feature_path: str = 'data/processed/feature_columns.json',
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feature_path = Path(feature_path)
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.feature_columns = self._load_feature_columns()

    def _load_model(self):
        if not self.model_path.exists():
            return None
        try:
            return NumpyNeuralNetwork.load(self.model_path)
        except Exception:
            with self.model_path.open('rb') as f:
                return pickle.load(f)

    def _load_scaler(self):
        if not self.scaler_path.exists():
            return None
        with self.scaler_path.open('rb') as f:
            return pickle.load(f)

    def _load_feature_columns(self) -> list[str]:
        if self.feature_path.exists():
            return json.loads(self.feature_path.read_text())
        return [
            'age',
            'hypertension',
            'heart_disease',
            'bmi',
            'HbA1c_level',
            'blood_glucose_level',
            'gender_enc',
            'smoking_enc',
        ]

    def _fallback_probability(self, case: Dict[str, Any]) -> float:
        score = 0.0
        if float(case.get('HbA1c_level', 0)) >= 6.5:
            score += 0.45
        if float(case.get('blood_glucose_level', 0)) >= 126:
            score += 0.35
        if float(case.get('bmi', 0)) >= 30:
            score += 0.10
        if float(case.get('age', 0)) >= 45:
            score += 0.05
        if float(case.get('hypertension', 0)) == 1:
            score += 0.05
        return max(0.0, min(1.0, score))

    def predict_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        row = case_to_feature_row(case)
        x = row[self.feature_columns].to_numpy(dtype='float32')

        if isinstance(self.scaler, dict) and self.scaler.get('type') == 'ManualMinMaxScaler':
            x = apply_manual_scaler(x, self.scaler)
        elif self.scaler is not None and hasattr(self.scaler, 'transform'):
            x = self.scaler.transform(x)

        if self.model is not None and hasattr(self.model, 'predict'):
            prob = float(self.model.predict(x)[0][0])
            raw_output = 'loaded artifact prediction'
        else:
            prob = self._fallback_probability(case)
            raw_output = 'fallback threshold-based prediction'

        prediction = 'YES' if prob >= 0.5 else 'NO'
        return {
            'prediction': prediction,
            'probability': float(prob),
            'raw_output': raw_output,
        }
