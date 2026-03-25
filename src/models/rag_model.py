from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from src.llm.client import OllamaClient
from src.llm.parser import parse_yes_no
from src.llm.prompts import build_rag_prompt


class RAGModel:
    def __init__(
        self,
        docs_dir: str = 'data/docs/mayo_guidelines',
        top_k: int | None = None,
        model_name: str | None = None,
        embedding_model: str | None = None,
        index_dir: str | None = None,
    ) -> None:
        config = self._load_config()
        retrieval_config = config.get('retrieval', {})
        artifact_config = config.get('artifacts', {})
        llm_config = config.get('llm', {})

        self.docs_dir = Path(docs_dir)
        self.top_k = top_k or int(retrieval_config.get('top_k', 3))
        self.index_dir = Path(index_dir or artifact_config.get('index_dir', 'artifacts/rag/index'))
        self.model_name = model_name or llm_config.get('model', 'deepseek-r1:8b')
        self.embedding_model = embedding_model or llm_config.get('embedding_model', 'nomic-embed-text')
        self.client = OllamaClient(model=self.model_name)

        self.meta_path = self.index_dir / 'rag_index_meta.pkl'
        self.embedding_path = self.index_dir / 'rag_index_embeddings.npy'

        self.chunks: List[str] = []
        self.metadata: List[dict] = []
        self.embedding_matrix = np.zeros((0, 1), dtype='float32')
        self._load_index()

    def _load_config(self) -> dict:
        config_path = Path(__file__).resolve().parents[2] / 'configs' / 'rag.yaml'
        if not config_path.exists():
            return {}
        with config_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _load_index(self) -> None:
        if self.meta_path.exists() and self.embedding_path.exists():
            with self.meta_path.open('rb') as f:
                meta = pickle.load(f)
            self.chunks = meta.get('chunks', [])
            self.metadata = meta.get('metadata', [])
            self.embedding_matrix = np.load(self.embedding_path)

    def _has_index(self) -> bool:
        return bool(self.chunks) and self.embedding_matrix.shape[0] == len(self.chunks)

    def _case_query(self, case: Dict[str, Any]) -> str:
        return (
            f"diabetes diagnosis HbA1c {case.get('HbA1c_level')} "
            f"blood glucose {case.get('blood_glucose_level')} "
            f"age {case.get('age')} hypertension {case.get('hypertension')} "
            f"heart disease {case.get('heart_disease')}"
        )

    def _retrieve(self, case: Dict[str, Any]) -> tuple[str, List[dict]]:
        if not self._has_index():
            raise RuntimeError(
                "RAG index is missing or empty. Run 'python3 scripts/build_rag_index.py' first."
            )

        query = self._case_query(case)
        query_embedding = np.asarray(
            self.client.embed(query, model=self.embedding_model),
            dtype='float32',
        )
        if query_embedding.size == 0:
            raise RuntimeError("Embedding request returned no query embedding.")

        doc_norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        scores = (self.embedding_matrix @ query_embedding) / ((doc_norms * query_norm) + 1e-9)
        top_idx = scores.argsort()[::-1][: self.top_k]

        retrieved = []
        context_parts = []
        for idx in top_idx:
            item = {
                'text': self.chunks[int(idx)],
                'source': self.metadata[int(idx)].get('source', 'unknown'),
                'chunk_id': self.metadata[int(idx)].get('chunk_id', int(idx)),
                'score': float(scores[int(idx)]),
            }
            retrieved.append(item)
            context_parts.append(
                f"Source: {item['source']} | Chunk: {item['chunk_id']} | Score: {item['score']:.4f}\n"
                f"{item['text']}"
            )

        return "\n\n".join(context_parts)[:2500], retrieved

    def _decide(self, case: Dict[str, Any]) -> tuple[str, float]:
        hba1c = float(case.get('HbA1c_level', 0))
        glucose = float(case.get('blood_glucose_level', 0))
        if hba1c >= 6.5 or glucose >= 126:
            return 'YES', 0.80
        if hba1c < 5.7 and glucose < 100:
            return 'NO', 0.80
        return 'NO', 0.55

    def predict_one_llm(self, case: Dict[str, Any]) -> Dict[str, Any]:
        try:
            context, retrieved = self._retrieve(case)
            prompt = build_rag_prompt(case, context)
            raw_output = self.client.generate(prompt)
            prediction, probability = parse_yes_no(raw_output)
            return {
                'prediction': prediction,
                'probability': probability,
                'raw_output': raw_output,
                'retrieved_context': context,
                'retrieved_sources': retrieved,
                'fallback_used': False,
            }
        except Exception as exc:
            prediction, probability = self._decide(case)
            fallback_message = (
                "WARNING: True RAG failed and heuristic fallback was used. "
                f"Reason: {exc}"
            )
            return {
                'prediction': prediction,
                'probability': probability,
                'raw_output': fallback_message,
                'retrieved_context': '',
                'retrieved_sources': [],
                'fallback_used': True,
            }

    def predict_one_heuristic(self, case: Dict[str, Any]) -> Dict[str, Any]:
        try:
            context, retrieved = self._retrieve(case)
            prediction, probability = self._decide(case)
            return {
                'prediction': prediction,
                'probability': probability,
                'raw_output': (
                    "Heuristic RAG mode used deterministic thresholds over retrieved context."
                ),
                'retrieved_context': context,
                'retrieved_sources': retrieved,
                'fallback_used': False,
            }
        except Exception as exc:
            prediction, probability = self._decide(case)
            fallback_message = (
                "WARNING: Heuristic RAG retrieval failed and direct threshold fallback was used. "
                f"Reason: {exc}"
            )
            return {
                'prediction': prediction,
                'probability': probability,
                'raw_output': fallback_message,
                'retrieved_context': '',
                'retrieved_sources': [],
                'fallback_used': True,
            }

    def predict_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_one_llm(case)
