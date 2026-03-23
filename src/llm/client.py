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
        resp.raise_for_status()
        return resp.json().get('response', '')
