from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_split_indices(y, test_size: float = 0.30, random_state: int = 42) -> dict:
    np.random.seed(random_state)
    y_array = np.asarray(y)
    train_indices = []
    test_indices = []

    for cls in np.unique(y_array):
        cls_indices = np.where(y_array == cls)[0]
        n_test = int(len(cls_indices) * test_size)
        shuffled = cls_indices.copy()
        np.random.shuffle(shuffled)
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    train_indices = np.asarray(train_indices)
    test_indices = np.asarray(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return {
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
    }


def apply_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[split['train_indices']].reset_index(drop=True)
    test_df = df.iloc[split['test_indices']].reset_index(drop=True)
    return train_df, test_df


def save_split(split: dict, path: str = 'data/processed/split_indices.json') -> None:
    Path(path).write_text(json.dumps(split, indent=2))
