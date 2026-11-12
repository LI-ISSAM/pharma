"""Load the markdown corpus, chunk it by heading, and parse the glossary."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SOURCE_LINE_RE = re.compile(r"^Source:\s*https?://", re.IGNORECASE)
GLOSSARY_TERM_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.*)$")
MAX_CHARS = 3000

DOC_TYPE_LABELS = {
    "dailymed": "FDA label (DailyMed)",
    "medlineplus": "consumer info (MedlinePlus)",
}


@dataclass(frozen=True)
class Chunk:
    id: int
    file: str       # e.g. "dailymed-warfarin.md"  <- used for citations
    drug: str       # e.g. "warfarin"
    doc_type: str   # "dailymed" | "medlineplus" | ...
    section: str    # e.g. "4 CONTRAINDICATIONS"
    text: str

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model: context header + body."""
        label = DOC_TYPE_LABELS.get(self.doc_type, self.doc_type)
        return f"{self.drug.title()} - {label} - {self.section}\n{self.text}"


@dataclass
class Glossary:
    entries: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "Glossary":
        entries: dict[str, str] = {}
        current: str | None = None
        for line in text.splitlines():
            m = GLOSSARY_TERM_RE.match(line)
            if m:
                current = m.group(1).strip()
                entries[current] = m.group(2).strip()
            elif current and line.startswith((" ", "\t")) and line.strip():
                entries[current] += " " + line.strip()  # wrapped continuation line
            elif not line.strip() or line.startswith("#"):
                current = None
        return cls(entries)

    def lookup(self, term: str) -> list[tuple[str, str]]:
        """Case-insensitive lookup: exact match first, then substring either way."""
        q = term.strip().lower()
        if not q:
            return []
        exact = [(t, d) for t, d in self.entries.items() if t.lower() == q]
        if exact:
            return exact
        return [(t, d) for t, d in self.entries.items() if q in t.lower() or t.lower() in q]


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split long text on paragraph (then line) boundaries into <= max_chars parts."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for para in text.split("\n\n"):
        pieces.extend([para] if len(para) <= max_chars else para.split("\n"))
    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = piece
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_path, body) pairs. Level-1 title is excluded from the path."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            label = " > ".join(t for lvl, t in stack if lvl >= 2) or "Overview"
            sections.append((label, body))
        buf.clear()

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        elif SOURCE_LINE_RE.match(line):
            continue
        else:
            buf.append(line)
    flush()
    return sections


def _split_stem(stem: str) -> tuple[str, str]:
    doc_type, _, drug = stem.lower().partition("-")
    return (doc_type, drug) if drug else ("unknown", stem.lower())


def load_corpus(corpus_dir: Path) -> tuple[list[Chunk], Glossary]:
    files = sorted(p for p in Path(corpus_dir).iterdir() if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"No .md/.txt files found in {corpus_dir}")

    chunks: list[Chunk] = []
    glossary = Glossary()
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if path.stem.lower() == "glossary":
            glossary = Glossary.parse(text)
            continue
        doc_type, drug = _split_stem(path.stem)
        for section, body in _parse_sections(text):
            parts = split_text(body)
            for i, part in enumerate(parts, 1):
                label = section if len(parts) == 1 else f"{section} (part {i}/{len(parts)})"
                chunks.append(Chunk(len(chunks), path.name, drug, doc_type, label, part))
    return chunks, glossary