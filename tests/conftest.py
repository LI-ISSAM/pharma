import re
import zlib

import numpy as np
import pytest

from app.config import ROOT
from app.corpus import load_corpus
from app.retriever import Retriever
from app.tools import ToolBox


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings, for offline tests."""

    DIM = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(token.encode()) % self.DIM] += 1.0
        return out


@pytest.fixture(scope="session")
def toolbox_factory():
    chunks, glossary = load_corpus(ROOT / "corpus")
    retriever = Retriever(FakeEmbedder(), chunks)
    retriever.build()
    return lambda: ToolBox(retriever, glossary)