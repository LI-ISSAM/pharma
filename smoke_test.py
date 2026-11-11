import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL"):
    if not os.getenv(var):
        raise SystemExit(f"Missing env var: {var}")

client = OpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL

emb = client.embeddings.create(
    model=os.environ["OPENAI_EMBEDDING_MODEL"],
    input=["warfarin"],
)
print("Embedding OK, dimensions:", len(emb.data[0].embedding))

chat = client.chat.completions.create(
    model=os.environ["OPENAI_MODEL"],
    messages=[{"role": "user", "content": "In one sentence, what is a contraindication?"}],
)
print("Chat OK:", chat.choices[0].message.content)