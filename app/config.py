"""Environment configuration. Fails fast with clear messages."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _env(name: str) -> str | None:
    """Read an env var, stripping whitespace and surrounding quotes.

    `docker run --env-file` keeps quotes literally (unlike python-dotenv),
    so we normalise here to behave the same in both places.
    """
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    corpus_dir: Path
    embedding_provider: str        # "openai" (default, used by graders) | "ollama" (local dev)
    ollama_embedding_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    missing = [v for v in ("OPENAI_API_KEY", "OPENAI_MODEL") if not _env(v)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    base_url = _env("OPENAI_BASE_URL")
    if base_url is None and _env("OPENAI_BASE_UR"):
        # The provided .env template ships with this typo.
        log.warning("Using OPENAI_BASE_UR (typo for OPENAI_BASE_URL) as fallback.")
        base_url = _env("OPENAI_BASE_UR")
    provider = (_env("EMBEDDING_PROVIDER") or "openai").lower()
    if provider not in {"openai", "ollama"}:
        raise ConfigError(f"EMBEDDING_PROVIDER must be 'openai' or 'ollama', got '{provider}'")

    return Settings(
        api_key=_env("OPENAI_API_KEY"),  # type: ignore[arg-type]
        base_url=base_url,
        chat_model=_env("OPENAI_MODEL"),  # type: ignore[arg-type]
        embedding_model=_env("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
        corpus_dir=Path(_env("CORPUS_DIR") or ROOT / "corpus"),
        embedding_provider=provider,
        ollama_embedding_model=_env("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text",
    )