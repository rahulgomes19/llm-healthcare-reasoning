from __future__ import annotations

from typing import Dict, Any


class BaseModel:
    def predict_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
