from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_raw_data(path: str = 'data/raw/diabetes_prediction_dataset.csv') -> pd.DataFrame:
    return pd.read_csv(path)


def load_feature_columns(path: str = 'data/processed/feature_columns.json') -> list[str]:
    return json.loads(Path(path).read_text())


def load_processed_split(split: str = 'test') -> pd.DataFrame:
    path = Path('data/processed') / f'{split}.csv'
    return pd.read_csv(path)
