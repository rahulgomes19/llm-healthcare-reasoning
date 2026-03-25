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
        batch_mode: str | None = None,
        exemplar_path: str | None = None,
        train_context_path: str | None = None,
        model_name: str | None = None,
        num_shots: int | None = None,
    ) -> None:
        config = self._load_config()
        exemplar_config = config.get("exemplars", {})
        context_config = config.get("train_context", {})
        batch_config = config.get("batch", {})
        llm_config = config.get("llm", {})

        self.mode = mode
        self.batch_mode = batch_mode or batch_config.get("mode", "heuristic")
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

    def _heuristic_decide(self, case: Dict[str, Any]) -> tuple[str, float]:
        hba1c = float(case.get("HbA1c_level", 0))
        glucose = float(case.get("blood_glucose_level", 0))
        bmi = float(case.get("bmi", 0))
        if hba1c >= 6.5 or glucose >= 126:
            return "YES", 0.80
        if hba1c < 5.7 and glucose < 100 and bmi < 25:
            return "NO", 0.75
        return "NO", 0.55

    def _build_prompt_and_fallback(self, case: Dict[str, Any]) -> tuple[str, str, float]:
        if self.mode == "zero_shot":
            prompt = build_icl_zero_shot_prompt(case)
            fallback_prediction, fallback_probability = self._heuristic_decide(case)
        elif self.mode == "train_context":
            prompt = build_icl_train_context_prompt(case, self.train_context)
            fallback_prediction, fallback_probability = self._heuristic_decide(case)
        else:
            nearest = self._select_exemplars(case)
            if not nearest:
                return "", "NO", 0.5
            prompt = build_icl_prompt(case, nearest)
            fallback_prediction, fallback_probability = self._vote_fallback(nearest)
        return prompt, fallback_prediction, fallback_probability

    def predict_one_llm(self, case: Dict[str, Any]) -> Dict[str, Any]:
        prompt, fallback_prediction, fallback_probability = self._build_prompt_and_fallback(case)
        if not prompt:
            return {
                "prediction": fallback_prediction,
                "probability": fallback_probability,
                "raw_output": "WARNING: ICL exemplar prompt could not be built because no exemplars were available.",
                "fallback_used": True,
            }

        try:
            raw_output = self.client.generate(prompt)
            prediction, probability = parse_yes_no(raw_output)
            return {
                "prediction": prediction,
                "probability": float(probability),
                "raw_output": raw_output,
                "fallback_used": False,
            }
        except Exception as exc:
            return {
                "prediction": fallback_prediction,
                "probability": float(fallback_probability),
                "raw_output": (
                    f"WARNING: ICL LLM mode failed and fallback was used. "
                    f"Mode='{self.mode}'. Reason: {exc}"
                ),
                "fallback_used": True,
            }

    def predict_one_heuristic(self, case: Dict[str, Any]) -> Dict[str, Any]:
        prompt, fallback_prediction, fallback_probability = self._build_prompt_and_fallback(case)
        if not prompt and self.mode == "few_shot":
            return {
                "prediction": fallback_prediction,
                "probability": fallback_probability,
                "raw_output": "WARNING: ICL heuristic mode used direct fallback because no exemplars were available.",
                "fallback_used": True,
            }

        return {
            "prediction": fallback_prediction,
            "probability": float(fallback_probability),
            "raw_output": (
                f"ICL heuristic batch mode used fallback logic for mode '{self.mode}' without calling the LLM."
            ),
            "fallback_used": False,
        }

    def predict_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if self.batch_mode == "heuristic":
            return self.predict_one_heuristic(case)
        return self.predict_one_llm(case)

    def predict_batch(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict_one(case) for case in cases]
