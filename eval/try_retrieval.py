
"""Evaluate the semantic retriever using Ollama embeddings."""

from app.config import get_settings
from app.corpus import load_corpus
from app.retriever import Retriever
import logging
from app.embeddings import get_embedder
from app.config import ROOT 
logging.basicConfig(level=logging.INFO)


CASES = [
    (
        "What are the main contraindications for warfarin?",
        {"dailymed-warfarin.md"}
    ),
    (
        "Compare the contraindications of warfarin and apixaban. "
        "What conditions are shared and which are unique to each drug?",
        {"dailymed-warfarin.md", "dailymed-apixaban.md"}
    ),
    (
        "A patient currently on warfarin is prescribed sertraline. "
        "What is the risk of bleeding and what monitoring is recommended?",
        {"dailymed-warfarin.md", "dailymed-sertraline.md"}
    ),
    (
        "A patient with an eGFR of 25 mL/min is being considered for metformin. "
        "Is metformin contraindicated at this level of renal function, "
        "and what is the threshold?",
        {"dailymed-metformin.md"}
    ),
]


if __name__ == "__main__":

    s = get_settings()

    chunks, glossary = load_corpus(s.corpus_dir)

    print(
        f"{len(chunks)} chunks, "
        f"{len(glossary.entries)} glossary terms, "
        f"drugs: {sorted({c.drug for c in chunks})}"
    )

    # Ollama local embeddings
    r = Retriever(get_embedder(), chunks, cache_dir=ROOT / ".cache")

    r.build()
    
    
    

    for question, expected in CASES:

        hits = r.search(question, k=6)

        files = {h.chunk.file for h in hits}

        status = "PASS" if expected <= files else "FAIL"

        print(f"\n{status}  {question[:80]}...")

        for h in hits:
            print(
                f"   {h.score:.3f}  "
                f"{h.chunk.file:<32} "
                f"{h.chunk.section}"
            )
