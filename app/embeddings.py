"""Embedding backends behind one tiny interface."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from openai import OpenAI

from .config import get_settings
from .llm import get_client


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array."""
        ...


class OpenAIEmbedder:
    """OpenAI-compatible embeddings (LiteLLM proxy). Used in Docker / by graders."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        data = sorted(resp.data, key=lambda d: d.index)
        return np.asarray([d.embedding for d in data], dtype=np.float32)


class OllamaEmbedder:
    """Local embeddings via Ollama. Dev only: needs a running Ollama server."""

    def __init__(self, model: str):
        import ollama  # lazy import: the Docker image doesn't need this package

        self._ollama = ollama
        self.model = model
        self.name= f"ollama:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._ollama.embed(model=self.model, input=texts)
        embeddings = resp["embeddings"]
        if len(embeddings) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
        return np.asarray(embeddings, dtype=np.float32)


def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "ollama":
        return OllamaEmbedder(s.ollama_embedding_model)
    return OpenAIEmbedder(get_client(), s.embedding_model)