from __future__ import annotations

import json
from pathlib import Path


def write_report(data: dict, path: str) -> None:
    Path(path).write_text(json.dumps(data, indent=2))
