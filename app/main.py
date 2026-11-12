"""FastAPI application: /health, /ask and the minimal frontend."""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Callable

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .agent import Agent, AgentError
from .config import ROOT, ConfigError
from .schemas import AskRequest, AskResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for noisy in ("httpx", "httpx2", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

FRONTEND_FILE = ROOT / "frontend" / "index.html"
STARTUP_WAIT_SECONDS = 120


class AgentService:
    """Builds the agent in a background thread so /health can answer immediately."""

    def __init__(self, loader: Callable[[], Agent]):
        self._loader = loader
        self._done = threading.Event()
        self._agent: Agent | None = None
        self.error: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._load, name="agent-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            self._agent = self._loader()
            log.info("Agent ready")
        except ConfigError as exc:
            log.exception("Agent configuration error")
            self.error = str(exc)
        except Exception:  # noqa: BLE001
            log.exception("Agent initialisation failed")
            self.error = "Initialisation failed; see server logs."
        finally:
            self._done.set()

    @property
    def status(self) -> str:
        if not self._done.is_set():
            return "loading"
        return "ready" if self._agent else "failed"

    def get(self, timeout: float) -> Agent:
        if not self._done.wait(timeout):
            raise HTTPException(503, "The agent is still starting up. Please retry in a few seconds.")
        if self._agent is None:
            raise HTTPException(503, f"The agent failed to initialise: {self.error}")
        return self._agent


def create_app(loader: Callable[[], Agent] | None = None) -> FastAPI:
    service = AgentService(loader or Agent.from_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.start()
        yield

    app = FastAPI(
        title="Pharma Research Assistant",
        description="Agentic research assistant over a local pharmacology corpus. Not medical advice.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        body = {"status": "ok", "agent": service.status}
        if service.error:
            body["detail"] = service.error
        return body

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        agent = service.get(STARTUP_WAIT_SECONDS)
        try:
            return agent.ask(request.question)
        except AgentError as exc:
            log.warning("Agent error: %s", exc)
            raise HTTPException(502, str(exc)) from exc
        except openai.APIError as exc:
            log.exception("LLM provider error")
            raise HTTPException(502, "The language model service returned an error. Please try again.") from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error while answering")
            raise HTTPException(500, "Unexpected internal error.") from exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_FILE)

    return app


app = create_app()