
import os
import ollama
from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    # 1. Vérifier les variables d'environnement
    required_vars = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    )

    for var in required_vars:
        if not os.getenv(var):
            raise SystemExit(f"Missing env var: {var}")

    # 2. Tester les embeddings avec Ollama
    print("\nTesting Ollama embeddings...")

    embedding_model = os.environ["OPENAI_EMBEDDING_MODEL"]

    emb = ollama.embed(
        model=embedding_model,
        input=["warfarin"]
    )

    vector = emb.embeddings[0]

    print("Embedding OK!")
    print("Dimensions:", len(vector))

    # 3. Tester le chat avec Gemini
    print("\nTesting Gemini chat...")

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    chat = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[
            {
                "role": "user",
                "content": "In one sentence, what is a contraindication?"
            }
        ],
    )

    print("Chat OK:", chat.choices[0].message.content)

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
