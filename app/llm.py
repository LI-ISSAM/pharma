"""Single shared OpenAI-compatible client (works with the LiteLLM proxy)."""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from .config import get_settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.api_key, base_url=s.base_url, timeout=45.0, max_retries=2)