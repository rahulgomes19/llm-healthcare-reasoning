from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

GENDER_MAP = {'female': 1, 'male': 2, 'other': 0, 'no info': 0}
SMOKE_MAP = {
    'never': 0,
    'former': 1,
    'not current': 1,
    'ever': 1,
    'current': 2,
    'no info': 0,
}


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['gender', 'smoking_history']:
        out[col] = out[col].astype(str).str.strip().str.lower()
    return out


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_text_columns(df)
    out['gender_enc'] = out['gender'].map(GENDER_MAP).fillna(0).astype('float32')
    out['smoking_enc'] = out['smoking_history'].map(SMOKE_MAP).fillna(0).astype('float32')
    return out.drop(columns=['gender', 'smoking_history'])


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.columns.drop('diabetes') if 'diabetes' in out.columns else out.columns
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors='coerce')
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    return out


def case_to_feature_row(case: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([
        {
            'age': float(case.get('age', 0)),
            'gender': str(case.get('gender', 'female')).lower(),
            'hypertension': float(case.get('hypertension', 0)),
            'heart_disease': float(case.get('heart_disease', 0)),
            'smoking_history': str(case.get('smoking_history', 'never')).lower(),
            'bmi': float(case.get('bmi', 0)),
            'HbA1c_level': float(case.get('HbA1c_level', 0)),
            'blood_glucose_level': float(case.get('blood_glucose_level', 0)),
        }
    ])
    return encode_features(impute_numeric(df))


def apply_manual_scaler(values: np.ndarray, scaler: dict) -> np.ndarray:
    min_values = np.asarray(scaler['min'], dtype='float32')
    max_values = np.asarray(scaler['max'], dtype='float32')
    ranges = max_values - min_values
    ranges[ranges == 0] = 1.0
    return (values - min_values) / ranges
