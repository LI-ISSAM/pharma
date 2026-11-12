import pytest
from fastapi.testclient import TestClient

from app.agent import AgentError
from app.main import create_app
from app.prompts import DISCLAIMER
from app.schemas import AskResponse, Plan, Review


class FakeAgent:
    def ask(self, question: str) -> AskResponse:
        return AskResponse(
            answer=f"echo: {question}\n\n---\n_{DISCLAIMER}_",
            sources=["dailymed-warfarin.md"],
            disclaimer=DISCLAIMER,
            plan=Plan(sub_questions=[question]),
            tool_calls=[],
            review=Review(),
            revised=False,
            elapsed_ms=1,
        )


class BrokenAgent:
    def ask(self, question: str) -> AskResponse:
        raise AgentError("The agent did not produce an answer.")


def make_client(loader):
    return TestClient(create_app(loader))


def test_health_is_200_even_before_ready():
    with make_client(FakeAgent) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


def test_ask_returns_answer_sources_and_disclaimer():
    with make_client(FakeAgent) as client:
        res = client.post("/ask", json={"question": "What is warfarin used for?"})
        assert res.status_code == 200
        body = res.json()
        assert body["sources"] == ["dailymed-warfarin.md"]
        assert DISCLAIMER in body["answer"]
        assert body["plan"]["sub_questions"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "  "}, {"question": "ab"}, {"question": "x" * 2001}])
def test_ask_rejects_bad_input(payload):
    with make_client(FakeAgent) as client:
        assert client.post("/ask", json=payload).status_code == 422


def test_ask_returns_503_when_startup_failed():
    def failing_loader():
        raise RuntimeError("boom")

    with make_client(failing_loader) as client:
        assert client.post("/ask", json={"question": "valid question"}).status_code == 503
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["agent"] == "failed"


def test_ask_returns_502_on_agent_error():
    with make_client(BrokenAgent) as client:
        assert client.post("/ask", json={"question": "valid question"}).status_code == 502


def test_frontend_is_served():
    with make_client(FakeAgent) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "/ask" in res.text