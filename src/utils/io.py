from __future__ import annotations

import json
import pickle
from pathlib import Path


def read_json(path: str):
    return json.loads(Path(path).read_text())


def write_json(path: str, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def read_pickle(path: str):
    with Path(path).open('rb') as f:
        return pickle.load(f)
