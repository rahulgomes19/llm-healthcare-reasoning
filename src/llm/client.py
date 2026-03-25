from __future__ import annotations

import os

import requests


class OllamaClient:
    def __init__(self, model: str = 'deepseek-r1:8b', host: str | None = None) -> None:
        self.model = model
        self.host = host or os.environ.get('OLLAMA_HOST', 'localhost:11434')

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f'http://{self.host}/api/generate',
            json={'model': self.model, 'prompt': prompt, 'stream': False},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Ollama generate request failed ({resp.status_code}): {resp.text[:500]}"
            )
        return resp.json().get('response', '')

    def embed(self, text: str, model: str) -> list[float]:
        resp = requests.post(
            f'http://{self.host}/api/embed',
            json={'model': model, 'input': text},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Ollama embed request failed ({resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()
        embeddings = data.get('embeddings', data.get('embedding', []))
        if isinstance(embeddings, list) and embeddings:
            return embeddings[0] if isinstance(embeddings[0], list) else embeddings
        return []
