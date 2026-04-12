from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_split_indices(
    y,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be non-negative and sum to less than 1.")

    np.random.seed(random_state)
    y_array = np.asarray(y)
    train_indices = []
    val_indices = []
    test_indices = []

    for cls in np.unique(y_array):
        cls_indices = np.where(y_array == cls)[0]
        n_val = int(len(cls_indices) * val_size)
        n_test = int(len(cls_indices) * test_size)
        shuffled = cls_indices.copy()
        np.random.shuffle(shuffled)
        val_end = n_val
        test_end = n_val + n_test
        val_indices.extend(shuffled[:val_end])
        test_indices.extend(shuffled[val_end:test_end])
        train_indices.extend(shuffled[test_end:])

    train_indices = np.asarray(train_indices)
    val_indices = np.asarray(val_indices)
    test_indices = np.asarray(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    return {
        'train_indices': train_indices.tolist(),
        'val_indices': val_indices.tolist(),
        'test_indices': test_indices.tolist(),
    }


def apply_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[split['train_indices']].reset_index(drop=True)
    val_df = df.iloc[split['val_indices']].reset_index(drop=True)
    test_df = df.iloc[split['test_indices']].reset_index(drop=True)
    return train_df, val_df, test_df


def save_split(split: dict, path: str = 'data/processed/split_indices.json') -> None:
    Path(path).write_text(json.dumps(split, indent=2))
