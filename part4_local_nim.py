"""
Workshop 4 — Run NVIDIA NIM on Your Own GPU
Same OpenAI-compatible code as Workshops 1-3. Two changes only:
  - base_url points at http://localhost:8000/v1 instead of integrate.api.nvidia.com
  - api_key isn't validated by local NIM

This script reads NIM_BASE_URL from the environment so the same file works
against the hosted API Catalog (default) and a local NIM container.

Run against hosted (what we've done in Parts 1-3):
    python3 part4_local_nim.py

Run against local NIM (after starting the container — see the post):
    NIM_BASE_URL=http://localhost:8000/v1 python3 part4_local_nim.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
IS_LOCAL = NIM_BASE_URL.startswith("http://localhost")

# Local NIM doesn't validate the key, but the OpenAI client requires *some* string.
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY and not IS_LOCAL:
    raise SystemExit(
        "Set NVIDIA_API_KEY in a .env file (only needed for the hosted endpoint). "
        "Get a free key at https://build.nvidia.com/"
    )

client = OpenAI(
    base_url=NIM_BASE_URL,
    api_key=API_KEY or "not-needed-for-local-dev",
)

MODEL = "meta/llama-3.1-8b-instruct"


def ask(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content


print(f"Endpoint: {NIM_BASE_URL}  ({'local NIM container' if IS_LOCAL else 'hosted API Catalog'})")
print(f"Model:    {MODEL}\n")

# A small workload — proves the OpenAI-compatible API surface works against
# whichever endpoint we just configured. Swap NIM_BASE_URL and rerun to see
# identical behavior against a local NIM container.
demo_questions = [
    ("You are a concise USC campus assistant.",
     "In one sentence, why might a USC research lab choose to self-host a model on its own GPU?"),
    ("You are a helpful assistant. Answer in exactly two sentences.",
     "What is GPU acceleration and why does it matter for AI workloads?"),
]

for system_prompt, user_message in demo_questions:
    print(f"Q: {user_message}")
    print(f"A: {ask(system_prompt, user_message)}\n")
