"""Prototype: let the LLM pick tools on its own. Usage: python -m eval.try_tool_calling "your question" """
import logging
import sys

from app.config import get_settings
from app.corpus import load_corpus
from app.embeddings import get_embedder
from app.llm import get_client
from app.config import ROOT
from app.retriever import Retriever
from app.tools import ToolBox

SYSTEM = (
    "You are a pharmacology research assistant. Use the tools to find evidence in the corpus. "
    "Never answer from memory. When you have enough evidence, answer briefly and cite file names."
)
DEFAULT_Q = (
    "Compare the contraindications of warfarin and apixaban. "
    "What conditions are shared and which are unique to each drug?"
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    question = " ".join(sys.argv[1:]) or DEFAULT_Q
    s = get_settings()
    chunks, glossary = load_corpus(s.corpus_dir)
    retriever = Retriever(get_embedder(), chunks, cache_dir=ROOT /".cache")
    retriever.build()
    box = ToolBox(retriever, glossary)
    client = get_client()

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(8):
        resp = client.chat.completions.create(
            model=s.chat_model, messages=messages, tools=box.schemas, tool_choice="auto"
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print("\n=== ANSWER ===\n", msg.content)
            break
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            print(f"[step {step}] {call.function.name}({call.function.arguments})")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": box.execute(call.function.name, call.function.arguments)}
            )
    print("\nSources used:", box.sources())