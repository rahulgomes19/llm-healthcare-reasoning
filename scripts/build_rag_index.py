from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.client import OllamaClient


def load_config() -> dict:
    config_path = PROJECT_ROOT / 'configs' / 'rag.yaml'
    with config_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def read_html_like(path: Path) -> str:
    raw_text = path.read_text(encoding='utf-8', errors='ignore')
    raw_text = re.sub(r'<script\b[^>]*>.*?</script>', ' ', raw_text, flags=re.IGNORECASE | re.DOTALL)
    raw_text = re.sub(r'<style\b[^>]*>.*?</style>', ' ', raw_text, flags=re.IGNORECASE | re.DOTALL)
    raw_text = re.sub(r'<!--.*?-->', ' ', raw_text, flags=re.DOTALL)
    cleaned = re.sub(r'<[^>]+>', ' ', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {'.txt', '.md', '.html'}:
            continue
        text = read_html_like(path)
        if text:
            docs.append((path.name, text))
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk[:3000])
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build the vector index used by the true RAG pipeline.'
    )
    parser.parse_args()

    config = load_config()
    retrieval_config = config.get('retrieval', {})
    artifact_config = config.get('artifacts', {})
    llm_config = config.get('llm', {})

    docs_dir = PROJECT_ROOT / 'data' / 'docs' / 'mayo_guidelines'
    index_dir = PROJECT_ROOT / artifact_config.get('index_dir', 'artifacts/rag/index')
    index_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = int(retrieval_config.get('chunk_size', 800))
    chunk_overlap = int(retrieval_config.get('chunk_overlap', 100))
    embedding_model = llm_config.get('embedding_model', 'nomic-embed-text')

    client = OllamaClient(model=llm_config.get('model', 'deepseek-r1:8b'))
    documents = load_documents(docs_dir)

    all_chunks = []
    metadata = []
    for source_name, text in documents:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({'source': source_name, 'chunk_id': chunk_id})

    embeddings = []
    for idx, chunk in enumerate(all_chunks, start=1):
        emb = client.embed(chunk, model=embedding_model)
        if not emb:
            raise RuntimeError(f'Embedding failed for chunk {idx}')
        embeddings.append(emb)
        if idx % 25 == 0:
            print(f'Embedded {idx} chunks')

    emb_array = np.asarray(embeddings, dtype='float32') if embeddings else np.zeros((0, 1), dtype='float32')

    with (index_dir / 'rag_index_meta.pkl').open('wb') as f:
        pickle.dump({'metadata': metadata, 'chunks': all_chunks}, f)
    np.save(index_dir / 'rag_index_embeddings.npy', emb_array)

    print(f'Saved RAG index to {index_dir}')
    print(f'Documents: {len(documents)}')
    print(f'Chunks: {len(all_chunks)}')
    print(f'Embedding shape: {emb_array.shape}')


if __name__ == '__main__':
    main()
