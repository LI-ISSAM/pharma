"""Python -m eval.diagnose_llm : teste le chat avec et sans tools, affiche le vrai message d'erreur."""
import openai

from app.config import get_settings
from app.llm import get_client
from app.tools import TOOL_SCHEMAS

s = get_settings()
print("base_url :", s.base_url)
print("model    :", repr(s.chat_model))
client = get_client().with_options(timeout=30, max_retries=0)


def attempt(label, **kwargs):
    try:
        r = client.chat.completions.create(
            model=s.chat_model,
            messages=[{"role": "user", "content": "Call list_documents, or reply ok."}],
            **kwargs,
        )
        calls = r.choices[0].message.tool_calls
        print(f"[OK]   {label} -> tool_calls={[c.function.name for c in calls] if calls else None}")
    except openai.APIStatusError as e:
        print(f"[FAIL] {label} -> {e.status_code}: {e.response.text[:500]}")


attempt("sans tools")
attempt("avec tools", tools=TOOL_SCHEMAS, tool_choice="auto")

try:
    ids = [m.id for m in client.models.list()][:40]
    print("modèles disponibles :", ids)
except openai.APIStatusError as e:
    print("liste des modèles impossible:", e.status_code)