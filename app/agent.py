"""Agent pipeline: plan -> research (tool loop) -> review (critic) -> revise -> answer."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import openai
from openai import OpenAI
from pydantic import ValidationError

from .config import ROOT, Settings, get_settings
from .corpus import Chunk, Glossary, load_corpus
from .embeddings import get_embedder
from .llm import get_client
from .prompts import (DISCLAIMER, NO_EVIDENCE_ANSWER, PLAN_SYSTEM, RESEARCH_SYSTEM, REVIEW_SYSTEM, REVISE_SYSTEM)
from .retriever import Retriever
from .schemas import AskResponse, Plan, Review
from .tools import GLOSSARY_FILE, ToolBox

log = logging.getLogger(__name__)

MAX_STEPS = 8
MAX_EVIDENCE_CHARS = 45_000
MAX_CAVEATS = 5


class AgentError(RuntimeError):
    pass


# ---- pure helpers (unit-tested) ---------------------------------------------

def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM output, tolerating code fences and surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")
    return data


def format_evidence(chunks: list[Chunk], glossary_used: dict[str, str], limit: int = MAX_EVIDENCE_CHARS) -> str:
    pieces = [f"[{c.file} § {c.section}]\n{c.text}" for c in chunks]
    pieces += [f"[{GLOSSARY_FILE} § {term}]\n{definition}" for term, definition in glossary_used.items()]
    out: list[str] = []
    total = 0
    for piece in pieces:
        if total + len(piece) > limit:
            out.append("[... further evidence omitted for length ...]")
            break
        out.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(out)


def cited_sources(text: str, box: ToolBox) -> list[str]:
    """Files the answer actually cites (limited to files the agent really retrieved)."""
    known = {c.file for c in box.evidence.values()}
    if box.glossary_used:
        known.add(GLOSSARY_FILE)
    cited = sorted(f for f in known if f in text)
    return cited or sorted(known)


# ---- agent ------------------------------------------------------------------

class Agent:
    def __init__(self, client: OpenAI, model: str, retriever: Retriever, glossary: Glossary):
        self.client = client
        self.model = model
        self.retriever = retriever
        self.glossary = glossary

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Agent":
        s = settings or get_settings()
        chunks, glossary = load_corpus(s.corpus_dir)
        retriever = Retriever(get_embedder(), chunks, cache_dir=ROOT / ".cache")
        retriever.build()
        return cls(get_client(), s.chat_model, retriever, glossary)

    # -- LLM helpers --

    def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            resp = self.client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except openai.BadRequestError:  # provider without JSON mode
            resp = self.client.chat.completions.create(**kwargs)
        return parse_json(resp.choices[0].message.content or "")

    def _chat_text(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    # -- pipeline stages --

    def _plan(self, question: str) -> Plan:
        user = f"Drugs available in the corpus: {', '.join(self.retriever.drugs)}\n\nQuestion: {question}"
        try:
            plan = Plan.model_validate(self._chat_json(PLAN_SYSTEM, user))
            plan.sub_questions = plan.sub_questions[:5]
            return plan
        except (ValueError, ValidationError):
            log.warning("Planner output invalid; falling back to the raw question", exc_info=True)
            return Plan(sub_questions=[question])

    def _research(self, question: str, plan: Plan, box: ToolBox) -> str:
        steps = "\n".join(f"- {q}" for q in plan.sub_questions)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nSuggested plan:\n{steps}"},
        ]
        for step in range(MAX_STEPS):
            last = step == MAX_STEPS - 1  # force a final answer on the last turn
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=box.schemas,
                tool_choice="none" if last else "auto",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return (msg.content or "").strip()
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                log.info("tool call: %s(%s)", call.function.name, call.function.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id,
                     "content": box.execute(call.function.name, call.function.arguments)}
                )
        raise AgentError("The agent did not produce an answer.")

    def _review(self, question: str, draft: str, evidence: str) -> Review:
        user = f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft}\n\nEVIDENCE:\n{evidence}"
        try:
            return Review.model_validate(self._chat_json(REVIEW_SYSTEM, user))
        except (ValueError, ValidationError):
            log.warning("Critic output invalid", exc_info=True)
            return Review(available=False, summary="Automated grounding review could not be completed.")

    def _revise(self, question: str, draft: str, review: Review, evidence: str) -> str:
        findings = {
            "problem_claims": [c.model_dump() for c in review.claims if c.verdict != "supported"],
            "issues": review.issues,
        }
        user = (f"QUESTION:\n{question}\n\nDRAFT:\n{draft}\n\n"
                f"REVIEW FINDINGS:\n{json.dumps(findings, ensure_ascii=False)}\n\nEVIDENCE:\n{evidence}")
        return self._chat_text(REVISE_SYSTEM, user) or draft

    @staticmethod
    def _caveats(review: Review) -> str:
        notes = [f"- {c.note or c.claim}" for c in review.claims if c.verdict == "ambiguous"][:MAX_CAVEATS]
        if not review.available:
            notes.append("- Automated grounding review was unavailable; verify every statement against the cited files.")
        return "\n\n**Points to verify with a clinician:**\n" + "\n".join(notes) if notes else ""

    # -- public API --

    def ask(self, question: str) -> AskResponse:
        started = time.perf_counter()
        box = ToolBox(self.retriever, self.glossary)

        plan = self._plan(question)
        log.info("plan: %s", plan.sub_questions)
        draft = self._research(question, plan, box)

        if not box.evidence and not box.glossary_used:
            answer, review, revised = NO_EVIDENCE_ANSWER, Review(available=False, summary="No passages retrieved."), False
        else:
            evidence = format_evidence(box.evidence_items(), box.glossary_used)
            review = self._review(question, draft, evidence)
            revised = review.available and review.needs_revision()
            answer = self._revise(question, draft, review, evidence) if revised else draft
            log.info("review: %d claims, revised=%s", len(review.claims), revised)
            answer += self._caveats(review)

        sources = cited_sources(answer, box)
        return AskResponse(
            answer=f"{answer}\n\n---\n_{DISCLAIMER}_",
            sources=sources,
            disclaimer=DISCLAIMER,
            plan=plan,
            tool_calls=box.calls,
            review=review,
            revised=revised,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )