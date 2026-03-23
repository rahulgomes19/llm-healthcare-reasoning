from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from src.llm.client import OllamaClient
from src.llm.parser import parse_yes_no
from src.llm.prompts import build_rag_prompt


class RAGModel:
    def __init__(
        self,
        docs_dir: str = 'data/docs/mayo_guidelines',
        top_k: int = 3,
        model_name: str = 'deepseek-r1:8b',
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.top_k = top_k
        self.client = OllamaClient(model=model_name)
        self.documents = self._load_documents()

    def _load_documents(self) -> List[str]:
        if not self.docs_dir.exists():
            return []
        docs: List[str] = []
        for path in sorted(self.docs_dir.iterdir()):
            if path.suffix.lower() not in {'.txt', '.md', '.html'}:
                continue
            raw_text = path.read_text(encoding='utf-8', errors='ignore')
            cleaned = re.sub(r"<[^>]+>", " ", raw_text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                docs.append(cleaned)
        return docs

    def _retrieve(self, case: Dict[str, Any]) -> str:
        query_terms = {
            'diabetes',
            str(case.get('HbA1c_level', '')),
            str(case.get('blood_glucose_level', '')),
            'hba1c',
            'glucose',
        }
        scored = []
        for doc in self.documents:
            chunks = re.split(r'\n\s*\n', doc)
            for chunk in chunks:
                text = chunk.lower()
                score = sum(term.lower() in text for term in query_terms)
                if score:
                    scored.append((score, chunk.strip()))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for _, chunk in scored[: self.top_k]]
        return '\n\n'.join(top_chunks) if top_chunks else 'No guideline context found.'

    def _decide(self, case: Dict[str, Any], context: str) -> tuple[str, float]:
        hba1c = float(case.get('HbA1c_level', 0))
        glucose = float(case.get('blood_glucose_level', 0))
        if hba1c >= 6.5 or glucose >= 126:
            return 'YES', 0.80
        if hba1c < 5.7 and glucose < 100:
            return 'NO', 0.80
        return 'NO', 0.55

    def predict_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        context = self._retrieve(case)
        prompt = build_rag_prompt(case, context)

        try:
            raw_output = self.client.generate(prompt)
            prediction, probability = parse_yes_no(raw_output)
        except Exception as exc:
            prediction, probability = self._decide(case, context)
            raw_output = (
                f"LLM unavailable, used fallback RAG heuristic: {exc}\n\n"
                f"Retrieved context:\n{context[:1200]}"
            )

        return {
            'prediction': prediction,
            'probability': probability,
            'raw_output': raw_output,
        }
