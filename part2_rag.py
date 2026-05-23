"""
Workshop 2 — From Manual RAG to Real Retrieval: Embedding-Based RAG with NVIDIA NIM
Local Python version. The same code runs in Colab — see the dev.to tutorial.

What this file demonstrates:
  1. NVIDIA's nv-embedqa-e5-v5 embedding model with the query/passage distinction.
  2. Cosine similarity over a tiny knowledge base — no vector database needed.
  3. Retrieval-augmented ask() that swaps the hardcoded campus_info from Part 1
     for a real retrieve_context() call.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Set NVIDIA_API_KEY in a .env file. "
        "Get a free key at https://build.nvidia.com/"
    )

# Same client + ask() shape as Part 1 (app.py) ──────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)

CHAT_MODEL = "meta/llama-3.1-8b-instruct"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"


def ask(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content


# Step 2 — Knowledge base + passage embeddings ─────────────────────────────────
knowledge_base = [
    {"title": "USC AI Club meeting",
     "text": "The USC AI Club meets every Thursday at 5 PM in the engineering building, room 204."},
    {"title": "USC GPU lab hours",
     "text": "The USC GPU computing lab is open Monday to Friday from 10 AM to 6 PM."},
    {"title": "NVIDIA Developer Program",
     "text": "USC students can join the NVIDIA Developer Program for free."},
    {"title": "Next USC workshop",
     "text": "The next USC AI Club workshop will cover Retrieval Augmented Generation (RAG)."},
    {"title": "USC AI/ML office hours",
     "text": "Office hours for the USC AI/ML faculty are Tuesdays 2-4 PM."},
    {"title": "USC robotics lab",
     "text": "The USC robotics lab requires safety training before students can use the soldering station."},
    {"title": "USC tutoring",
     "text": "Peer tutoring for introductory Python at USC is available Wednesdays from 1 PM to 3 PM."},
]


def embed_texts(texts, input_type="passage"):
    """NVIDIA embedding model — note the query/passage distinction via extra_body."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type},
    )
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


print("Embedding knowledge base as passages...")
embeddings = embed_texts([item["text"] for item in knowledge_base], input_type="passage")
for item, embedding in zip(knowledge_base, embeddings):
    item["embedding"] = embedding
print(f"Embedded {len(knowledge_base)} chunks. Vector dim: {embeddings[0].shape[0]}")


# Step 3 — Retrieval ────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_context(question: str, k: int = 3) -> str:
    # User question goes in as a 'query' (not 'passage') — this is the part beginners miss.
    question_embedding = embed_texts([question], input_type="query")[0]

    scored = []
    for item in knowledge_base:
        score = cosine_similarity(question_embedding, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_items = [item for score, item in scored[:k]]

    return "\n".join(f"- {item['text']}" for item in top_items)


# Step 4 — Retrieval-augmented ask() ───────────────────────────────────────────
def ask_with_retrieval(question: str) -> str:
    context = retrieve_context(question)

    system_prompt = f"""You are a USC campus assistant. Answer ONLY using the
context below. If the answer is not in the context, say
"I don't have that information — check with the USC AI Club."

CONTEXT:
{context}
"""

    return ask(system_prompt, question)


# Demo run ─────────────────────────────────────────────────────────────────────
print("\n── Workshop 2: Embedding-based RAG over the USC knowledge base ──")
for question in [
    "Where does the USC AI Club meet?",
    "When can I get Python tutoring at USC?",
    "What is the wifi password?",  # not in the data — should refuse
]:
    print(f"\nQ: {question}")
    print(f"Context:\n{retrieve_context(question)}")
    print(f"A: {ask_with_retrieval(question)}")
