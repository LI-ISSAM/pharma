"""Pydantic models shared by the agent and the API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1)
    drugs: list[str] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "ambiguous"]
    source: str | None = None
    note: str | None = None

    @field_validator("source", mode="before")
    @classmethod
    def _join_sources(cls, v: Any) -> Any:
        return ", ".join(map(str, v)) if isinstance(v, list) else v


class Review(BaseModel):
    available: bool = True
    claims: list[ClaimCheck] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    summary: str = ""

    def needs_revision(self) -> bool:
        return bool(self.issues) or any(c.verdict != "supported" for c in self.claims)


class AskRequest(BaseModel):
    question: str = Field(max_length=2000)

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 characters")
        return v

class AskResponse(BaseModel):
    answer: str                # final answer, citations + disclaimer included
    sources: list[str]
    disclaimer: str           # also returned separately for UIs
    plan: Plan
    tool_calls: list[dict[str, Any]]
    review: Review
    revised: bool
    elapsed_ms: int