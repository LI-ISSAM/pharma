from app.llm import get_client

for m in get_client().models.list():
    print(m.id)