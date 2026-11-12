"""Time each stage separately so we can see exactly what is slow. Run: python -m eval.diagnose"""
import time
from urllib.parse import urlparse

from app.config import get_settings
from app.corpus import load_corpus
from app.embeddings import get_embedder
from app.llm import get_client
from app.tools import TOOL_SCHEMAS


def stage(label, fn):
    print(f"-> {label} ...", flush=True)
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED after {time.perf_counter() - start:.1f}s: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
    print(f"   ok in {time.perf_counter() - start:.1f}s", flush=True)
    return result


if __name__ == "__main__":
    s = get_settings()
    host = urlparse(s.base_url).netloc if s.base_url else "(default OpenAI)"
    print(f"embedding provider={s.embedding_provider} | ollama model={s.ollama_embedding_model}")
    print(f"chat model={s.chat_model} | base_url host={host}\n", flush=True)

    chunks, _ = stage("load corpus", lambda: load_corpus(s.corpus_dir))
    print(f"   {len(chunks)} chunks, longest = {max(len(c.text) for c in chunks)} chars", flush=True)

    embedder = get_embedder()
    stage("embed 1 short text (includes model load)", lambda: embedder.embed(["warfarin"]))
    stage("embed 1 batch of 16 chunks", lambda: embedder.embed([c.embedding_text for c in chunks[:16]]))

    client = get_client().with_options(timeout=30, max_retries=0)  # fail fast
    msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
    stage("chat, no tools", lambda: client.chat.completions.create(model=s.chat_model, messages=msgs))

    def with_tools():
        r = client.chat.completions.create(
            model=s.chat_model,
            messages=[{"role": "user", "content": "Call the list_documents tool to see what is in the corpus."}],
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        calls = r.choices[0].message.tool_calls
        print(f"   tool calls returned: {[c.function.name for c in calls] if calls else 'NONE'}", flush=True)

    stage("chat WITH tools", with_tools)