"""Local tools the agent can call via OpenAI function calling.

A ToolBox is created per request: it executes tool calls and records which
passages the agent actually saw (evidence), which later drives citations
and the critic step.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

from .corpus import Chunk, Glossary
from .retriever import Retriever

log = logging.getLogger(__name__)

MAX_SEARCH_K = 8
MAX_SECTION_CHARS = 6000
GLOSSARY_FILE = "GLOSSARY.md"
_NUMERIC_RE = re.compile(r"^\d+(\.\d+)*$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _base_section(label: str) -> str:
    """Strip the ' (part i/n)' suffix added when a long section was split."""
    return label.split(" (part ")[0]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List the documents in the corpus (drug, file, source type). "
                "If `drug` is given, also list that drug's section names so you can call read_section. "
                "Use this first when unsure which drugs or sections exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {"drug": {"type": "string", "description": "Optional drug name, e.g. 'warfarin'."}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Semantic search over drug label passages. Returns the most relevant passages with file and section. "
                "Filter to ONE drug with `drug`. For multi-drug questions, call this once per drug."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific search query, e.g. 'contraindications'."},
                    "drug": {"type": "string", "description": "Optional: restrict to one drug, e.g. 'apixaban'."},
                    "k": {"type": "integer", "description": f"Number of passages (1-{MAX_SEARCH_K}), default 5."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": (
                "Read the COMPLETE text of a section of a drug's document. Prefer this over search when you need an "
                "exhaustive list (e.g. all contraindications) or exact thresholds. `section` is a name or number, "
                "e.g. 'contraindications', '7 drug interactions', '4', 'renal impairment'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {"type": "string", "description": "Drug name, e.g. 'metformin'."},
                    "section": {"type": "string", "description": "Section name or number."},
                    "source": {
                        "type": "string",
                        "enum": ["dailymed", "medlineplus"],
                        "description": "Optional: FDA label (dailymed) or consumer info (medlineplus). Default: both.",
                    },
                },
                "required": ["drug", "section"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_glossary",
            "description": "Look up a pharmacology term (e.g. 'INR', 'contraindication', 'DOAC') in the glossary.",
            "parameters": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolBox:
    def __init__(self, retriever: Retriever, glossary: Glossary):
        self.retriever = retriever
        self.glossary = glossary
        self.chunks = retriever.chunks
        self.evidence: dict[int, Chunk] = {}
        self.glossary_used: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "list_documents": self.list_documents,
            "search_corpus": self.search_corpus,
            "read_section": self.read_section,
            "lookup_glossary": self.lookup_glossary,
        }

    # ---- public API -------------------------------------------------------

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def evidence_items(self) -> list[Chunk]:
        return [self.evidence[i] for i in sorted(self.evidence)]

    def sources(self) -> list[str]:
        files = {c.file for c in self.evidence.values()}
        if self.glossary_used:
            files.add(GLOSSARY_FILE)
        return sorted(files)

    def execute(self, name: str, raw_arguments: str | None) -> str:
        """Run one tool call. Never raises: errors are returned to the model as JSON."""
        started = time.perf_counter()
        args: dict[str, Any] = {}
        handler = self._handlers.get(name)
        try:
            if handler is None:
                result: dict[str, Any] = {"error": f"Unknown tool '{name}'. Available: {sorted(self._handlers)}"}
            else:
                parsed = json.loads(raw_arguments or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                args = parsed
                result = handler(**args)
        except json.JSONDecodeError:
            result = {"error": "Arguments are not valid JSON."}
        except (TypeError, ValueError) as exc:
            result = {"error": f"Invalid arguments: {exc}"}
        except Exception:  # noqa: BLE001 - tool failures must not crash the agent loop
            log.exception("Tool %s failed", name)
            result = {"error": "The tool failed unexpectedly."}

        self.calls.append(
            {
                "tool": name,
                "arguments": args,
                "count": result.get("count"),
                "error": result.get("error"),
                "ms": round((time.perf_counter() - started) * 1000),
            }
        )
        return json.dumps(result, ensure_ascii=False)

    # ---- helpers ----------------------------------------------------------

    def _record(self, chunk: Chunk) -> None:
        self.evidence[chunk.id] = chunk

    @staticmethod
    def _as_result(chunk: Chunk, score: float | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {"file": chunk.file, "section": chunk.section, "text": chunk.text}
        if score is not None:
            item["score"] = round(score, 3)
        return item

    def _check_drug(self, drug: str) -> tuple[str | None, dict[str, Any] | None]:
        name = _norm(drug)
        if name in self.retriever.drugs:
            return name, None
        return None, {"error": f"Drug '{drug}' is not in the corpus.", "available_drugs": self.retriever.drugs}

    # ---- tools ------------------------------------------------------------

    def list_documents(self, drug: str | None = None) -> dict[str, Any]:
        if drug:
            name, err = self._check_drug(drug)
            if err:
                return err
            docs: dict[str, dict[str, Any]] = {}
            for c in self.chunks:
                if c.drug != name:
                    continue
                d = docs.setdefault(c.file, {"file": c.file, "type": c.doc_type, "sections": []})
                base = _base_section(c.section)
                if base not in d["sections"]:
                    d["sections"].append(base)
            return {"count": len(docs), "documents": list(docs.values())}

        seen: dict[str, dict[str, str]] = {}
        for c in self.chunks:
            seen.setdefault(c.file, {"file": c.file, "drug": c.drug, "type": c.doc_type})
        return {"count": len(seen), "documents": list(seen.values()), "glossary_available": True}

    def search_corpus(self, query: str, drug: str | None = None, k: int = 5) -> dict[str, Any]:
        if not query or not query.strip():
            return {"error": "`query` must be a non-empty string."}
        name = None
        if drug:
            name, err = self._check_drug(drug)
            if err:
                return err
        k = max(1, min(int(k), MAX_SEARCH_K))
        hits = self.retriever.search(query.strip(), k=k, drug=name)
        for h in hits:
            self._record(h.chunk)
        return {"count": len(hits), "results": [self._as_result(h.chunk, h.score) for h in hits]}

    def read_section(self, drug: str, section: str, source: str | None = None) -> dict[str, Any]:
        name, err = self._check_drug(drug)
        if err:
            return err
        if not section or not section.strip():
            return {"error": "`section` must be a non-empty string."}
        if source and source not in {c.doc_type for c in self.chunks}:
            return {"error": f"Unknown source '{source}'.", "available_sources": sorted({c.doc_type for c in self.chunks})}

        candidates = [c for c in self.chunks if c.drug == name and (not source or c.doc_type == source)]
        q = _norm(section)
        if _NUMERIC_RE.match(q):
            def matches(c: Chunk) -> bool:
                for part in _base_section(c.section).split(" > "):
                    number = _norm(part).split(" ")[0]
                    if number == q or number.startswith(q + "."):
                        return True
                return False
        else:
            def matches(c: Chunk) -> bool:
                return q in _norm(_base_section(c.section))

        selected = [c for c in candidates if matches(c)]
        if not selected:
            available = sorted({f"{c.file} :: {_base_section(c.section)}" for c in candidates})
            return {
                "error": f"No section matching '{section}' for {name}.",
                "available_sections": available[:80],
            }

        out: list[dict[str, Any]] = []
        total = 0
        for i, c in enumerate(selected):
            if out and total + len(c.text) > MAX_SECTION_CHARS:
                omitted = len(selected) - i
                self._finish_read(out)
                return {
                    "count": len(out),
                    "sections": out,
                    "truncated": True,
                    "omitted_chunks": omitted,
                    "hint": "Output truncated. Call read_section again with a more specific section name.",
                }
            out.append(self._as_result(c))
            self._record(c)
            total += len(c.text)
        return {"count": len(out), "sections": out, "truncated": False}

    def _finish_read(self, out: list[dict[str, Any]]) -> None:  # kept for symmetry/logging
        log.debug("read_section returned %d chunk(s) before truncation", len(out))

    def lookup_glossary(self, term: str) -> dict[str, Any]:
        matches = self.glossary.lookup(term)
        if not matches:
            return {"count": 0, "message": f"'{term}' is not in the glossary."}
        for t, definition in matches:
            self.glossary_used[t] = definition
        return {"count": len(matches), "definitions": [{"term": t, "definition": d} for t, d in matches]}