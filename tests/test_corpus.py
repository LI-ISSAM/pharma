from app.config import ROOT
from app.corpus import load_corpus, split_text

CHUNKS, GLOSSARY = load_corpus(ROOT / "corpus")


def test_chunks_are_non_empty():
    assert len(CHUNKS) > 50
    assert all(c.text.strip() for c in CHUNKS)


def test_warfarin_contraindications_chunk():
    hits = [c for c in CHUNKS if c.file == "dailymed-warfarin.md" and c.section.startswith("4 CONTRAINDICATIONS")]
    assert hits, "contraindications section not found"
    assert any("Hemorrhagic tendencies" in c.text for c in hits)


def test_source_url_line_is_dropped():
    assert not any(c.text.startswith("Source:") for c in CHUNKS)


def test_split_text_respects_limit():
    text = "\n\n".join(["word " * 100] * 20)
    assert all(len(p) <= 3000 for p in split_text(text, 3000))


def test_glossary_lookup():
    assert GLOSSARY.lookup("INR")
    assert GLOSSARY.lookup("contraindication")[0][0] == "Contraindication"