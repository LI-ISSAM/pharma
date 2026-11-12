"""In-memory semantic retriever: pluggable embeddings + numpy cosine similarity."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from .corpus import Chunk
from .embeddings import Embedder

log = logging.getLogger(__name__)
DRUG_BOOST = 0.10  # small bonus for chunks of a drug explicitly named in the query


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, embedder: Embedder, chunks: list[Chunk], batch_size: int = 32,
                 cache_dir: Path | None = None):
        self.embedder = embedder
        self.chunks = chunks
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.drugs = sorted({c.drug for c in chunks})
        self._drug_array = np.array([c.drug for c in chunks])
        self._matrix: np.ndarray | None = None

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(m, axis=-1, keepdims=True)
            return m / np.clip(norms, 1e-12, None)
        
    
    def search(
        self,
        query: str,
        k: int = 5,
        drug: str | None = None
    ) -> list[Hit]:

        if self._matrix is None:
            raise RuntimeError(
                "Retriever index not built; call build() first."
            )

        # Generate embedding for the question
        q = self._normalize(
            self.embedder.embed([query])
        )[0]

        # Calculate cosine similarity
        scores = self._matrix @ q

        # Give a small bonus to explicitly mentioned drugs
        mentioned = self.detect_drugs(query)

        if mentioned:
            scores = scores + DRUG_BOOST * np.isin(
                self._drug_array,
                list(mentioned)
            )

        # Optional filtering by drug
        if drug:
            scores = np.where(
                self._drug_array == drug.lower().strip(),
                scores,
                -np.inf
            )

        # Select the top K chunks
        top = np.argsort(-scores)[:k]

        return [
            Hit(self.chunks[i], float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    def detect_drugs(self, query: str) -> set[str]:
        q = query.lower()

        return {
            drug
            for drug in self.drugs
            if re.search(rf"\b{re.escape(drug)}\b", q)
        }
    def _cache_path(self, texts: list[str]) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256()
        digest.update(getattr(self.embedder, "name", "unknown").encode())
        for t in texts:
            digest.update(b"\x00" + t.encode())
        return Path(self.cache_dir) / f"embeddings-{digest.hexdigest()[:16]}.npy"

    def build(self) -> None:
        if not self.chunks:
            raise ValueError("The corpus contains no chunks.")
        texts = [c.embedding_text for c in self.chunks]

        cache = self._cache_path(texts)
        if cache and cache.exists():
            self._matrix = np.load(cache)
            log.info("Loaded embeddings from cache (%s)", cache.name)
            return

        parts = []
        for i in range(0, len(texts), self.batch_size):
            parts.append(self.embedder.embed(texts[i : i + self.batch_size]))
            log.info("Embedded %d/%d chunks", min(i + self.batch_size, len(texts)), len(texts))
        self._matrix = self._normalize(np.vstack(parts))

        if cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self._matrix)
            except OSError:
                log.warning("Could not write embedding cache", exc_info=True)
        log.info("Retriever ready: %d chunks indexed", len(self.chunks))