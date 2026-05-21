"""
Workshop 1 — Build Your First AI App with NVIDIA NIM
Local Python version. For the workshop itself, students run notebook.ipynb in Colab.

What this file demonstrates:
  1. A single model call to an NVIDIA-hosted NIM via the OpenAI-compatible API.
  2. Changing behavior by changing the system prompt.
  3. A tiny "campus assistant" that answers from a knowledge base you provide.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Set NVIDIA_API_KEY in a .env file. "
        "Get a free key at https://build.nvidia.com/"
    )

# NVIDIA's API Catalog is OpenAI-compatible — point the OpenAI client at it.
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)

# Any model from https://build.nvidia.com/ works here.
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


# ── Step 1: A plain model call ────────────────────────────────────────────────
print("── Step 1: Plain model call ──")
print(ask(
    system_prompt="You are a helpful, concise assistant.",
    user_message="Explain GPU acceleration to a first-year CS student in 5 sentences.",
))

# ── Step 2: Change the system prompt, change the behavior ────────────────────
print("\n── Step 2: Same question, different persona ──")
print(ask(
    system_prompt="You are a sarcastic but accurate professor. Keep it under 5 sentences.",
    user_message="Explain GPU acceleration to a first-year CS student.",
))

# ── Step 3: Your first "AI app" — a campus assistant ─────────────────────────
campus_info = """
The AI Club meets every Thursday at 5 PM in the engineering building, room 204.
The GPU computing lab is open Monday to Friday from 10 AM to 6 PM.
Students can join the NVIDIA Developer Program for free to access tools and learning resources.
The next workshop will cover Retrieval Augmented Generation (RAG).
Office hours for the AI/ML faculty are Tuesdays 2-4 PM.
"""

assistant_system_prompt = f"""You are a campus assistant. Answer ONLY using the
information in CAMPUS INFO below. If the answer is not in there, say "I don't
have that information — check with the AI Club."

CAMPUS INFO:
{campus_info}
"""

print("\n── Step 3: Campus assistant ──")
for question in [
    "When does the AI Club meet?",
    "Is the GPU lab open on Saturday?",
    "What's the wifi password?",  # not in the data — should refuse
]:
    print(f"\nQ: {question}")
    print(f"A: {ask(assistant_system_prompt, question)}")
