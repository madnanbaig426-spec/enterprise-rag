import os

import httpx
from dotenv import load_dotenv


load_dotenv()


QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "http://localhost:11434/api/chat",
)

QWEN_MODEL = os.getenv(
    "QWEN_MODEL",
    "qwen2.5:7b",
)


def generate_answer(
    question: str,
    context: str,
) -> str:

    prompt = f"""
You are an enterprise RAG assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }

    response = httpx.post(
        QWEN_API_URL,
        json=payload,
        timeout=120.0,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]
