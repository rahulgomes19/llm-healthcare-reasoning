from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.llm.client import OllamaClient
from src.llm.parser import parse_yes_no
from src.llm.prompts import (
    build_icl_prompt,
    build_icl_train_context_prompt,
    build_icl_zero_shot_prompt,
)


class ICLModel:
    def __init__(
        self,
        mode: str = "few_shot",
        exemplar_path: str | None = None,
        train_context_path: str | None = None,
        model_name: str | None = None,
        num_shots: int | None = None,
    ) -> None:
        config = self._load_config()
        exemplar_config = config.get("exemplars", {})
        context_config = config.get("train_context", {})
        llm_config = config.get("llm", {})

        self.mode = mode
        self.num_shots = num_shots or int(exemplar_config.get("num_shots", 3))
        self.exemplar_path = Path(
            exemplar_path or exemplar_config.get("path", "artifacts/icl/exemplar_set.json")
        )
        self.train_context_path = Path(
            train_context_path or context_config.get("path", "artifacts/icl/train_context.json")
        )
        self.client = OllamaClient(model=model_name or llm_config.get("model", "deepseek-r1:8b"))
        self.exemplars = self._load_exemplars()
        self.train_context = self._load_train_context()

    def _load_config(self) -> dict:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "icl.yaml"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_exemplars(self) -> List[Dict[str, Any]]:
        if self.exemplar_path.exists():
            return json.loads(self.exemplar_path.read_text())
        return []

    def _load_train_context(self) -> Dict[str, Any]:
        if self.train_context_path.exists():
            return json.loads(self.train_context_path.read_text())
        return {}

    def _distance(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        keys = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
        return sum(abs(float(a.get(k, 0)) - float(b.get(k, 0))) for k in keys)

    def _select_exemplars(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.exemplars:
            return []
        ranked = sorted(self.exemplars, key=lambda ex: self._distance(case, ex["case"]))
        return ranked[: self.num_shots]

    def _vote_fallback(self, nearest: List[Dict[str, Any]]) -> tuple[str, float]:
        if not nearest:
            return "NO", 0.5
        yes_votes = sum(1 for ex in nearest if ex["label"] == "YES")
        probability = yes_votes / len(nearest)
        prediction = "YES" if probability >= 0.5 else "NO"
        return prediction, float(probability)

    def predict_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "zero_shot":
            prompt = build_icl_zero_shot_prompt(case)
            fallback_prediction, fallback_probability = "NO", 0.5
        elif self.mode == "train_context":
            prompt = build_icl_train_context_prompt(case, self.train_context)
            hba1c = float(case.get("HbA1c_level", 0))
            glucose = float(case.get("blood_glucose_level", 0))
            fallback_prediction = "YES" if hba1c >= 6.5 or glucose >= 126 else "NO"
            fallback_probability = 0.75 if fallback_prediction == "YES" else 0.55
        else:
            nearest = self._select_exemplars(case)
            if not nearest:
                return {
                    "prediction": "NO",
                    "probability": 0.5,
                    "raw_output": "No exemplars available.",
                }
            prompt = build_icl_prompt(case, nearest)
            fallback_prediction, fallback_probability = self._vote_fallback(nearest)

        try:
            raw_output = self.client.generate(prompt)
            prediction, probability = parse_yes_no(raw_output)
        except Exception as exc:
            prediction, probability = fallback_prediction, fallback_probability
            raw_output = (
                f"LLM unavailable, used fallback ICL mode '{self.mode}': {exc}\n\n"
                f"{prompt}"
            )

        return {
            "prediction": prediction,
            "probability": float(probability),
            "raw_output": raw_output,
        }

    def predict_batch(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict_one(case) for case in cases]
