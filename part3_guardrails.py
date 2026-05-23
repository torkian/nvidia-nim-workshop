"""
Workshop 3 — Add Guardrails So Your AI App Doesn't Lie
Two practical guardrail layers with NVIDIA NIM:
  1. Scoped prompt with a fixed fallback line.
  2. A grounding check (second NIM call) on every answer.

Builds on the retriever from Workshop 2 (part2_rag.py).
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

# ── Setup carried from Workshops 1 and 2 ──────────────────────────────────────
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
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type},
    )
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_context(question: str, k: int = 3) -> str:
    q_emb = embed_texts([question], input_type="query")[0]
    scored = [(cosine_similarity(q_emb, item["embedding"]), item) for item in knowledge_base]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return "\n".join(f"- {item['text']}" for _, item in scored[:k])


# Embed the knowledge base once at startup.
print("Embedding knowledge base...")
for item, embedding in zip(
    knowledge_base,
    embed_texts([item["text"] for item in knowledge_base], input_type="passage"),
):
    item["embedding"] = embedding
print(f"Embedded {len(knowledge_base)} chunks.")


# ── Workshop 3 starts here ────────────────────────────────────────────────────
FALLBACK = "I don't have that information — check with the USC AI Club."

SCOPED_SYSTEM_PROMPT_TEMPLATE = """You are a USC campus assistant for AI Club,
GPU lab, NVIDIA program, workshop, office hour, robotics lab, and tutoring
questions only.

Rules:
- Answer ONLY using the CONTEXT below.
- If the user asks about anything outside this scope (e.g. weather, jokes,
  personal advice, code generation, general world knowledge), reply with
  exactly: "{fallback}"
- If the answer is not present in the context, reply with exactly: "{fallback}"
- Do not invent names, dates, room numbers, links, passwords, schedules,
  policies, or instructions that are not in the context.

CONTEXT:
{context}
"""


def answer_is_grounded(question: str, context: str, answer: str) -> bool:
    """Layer 2 — a second NIM call that checks the answer against the context."""
    verdict = ask(
        system_prompt=(
            "You are a strict grounding verifier. Read the CONTEXT and the "
            "ANSWER. Respond with only 'yes' or 'no'. Say 'yes' if every "
            "factual claim in the ANSWER is directly supported by the CONTEXT. "
            "Say 'no' otherwise — including if the ANSWER adds information not "
            "in the CONTEXT, even if that information sounds plausible."
        ),
        user_message=(
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Is every factual claim in the ANSWER supported by the CONTEXT?"
        ),
    )
    return verdict.strip().lower().startswith("yes")


def ask_guarded(question: str) -> str:
    context = retrieve_context(question)
    system_prompt = SCOPED_SYSTEM_PROMPT_TEMPLATE.format(
        fallback=FALLBACK, context=context,
    )
    answer = ask(system_prompt, question)
    if not answer_is_grounded(question, context, answer):
        return FALLBACK
    return answer


# Demo run ─────────────────────────────────────────────────────────────────────
print("\n── Workshop 3: guardrailed USC campus assistant ──")
for question in [
    "When does the USC AI Club meet?",          # in scope, in context
    "Can you write my breakup text?",           # OUT of scope
    "What is the wifi password?",               # in scope, NOT in context
    "What are the USC GPU lab Saturday hours?", # invites a hallucination
]:
    print(f"\nQ: {question}")
    print(f"A: {ask_guarded(question)}")
