"""
Workshop 5 — From Chatbot to Agent: Tool Calling with NVIDIA NIM
Self-contained — bundles everything from Workshops 1-3 so this script runs
on its own. Adds two tools (clock + USC knowledge-base search) and the
agent loop that lets the NIM model decide which to call.
"""

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
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

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)

MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"   # switched from 3.1-8b in workshops 1-4 —
                                         # this NVIDIA-tuned model is far more reliable
                                         # at tool calling, which matters once an agent
                                         # has to choose between tools instead of just chatting.
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"


def ask(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "/no_think\n\n" + system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content


# ── Carry the USC knowledge base + retriever from Workshop 2 ──────────────────
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


print("Embedding USC knowledge base...")
for item, emb in zip(knowledge_base, embed_texts([i["text"] for i in knowledge_base], "passage")):
    item["embedding"] = emb
print(f"Embedded {len(knowledge_base)} chunks.")


# ── Workshop 5 starts here ────────────────────────────────────────────────────
def get_current_time(timezone: str = "America/Los_Angeles") -> str:
    try:
        zone = ZoneInfo(timezone)
    except Exception:
        zone = ZoneInfo("UTC")
    return datetime.now(zone).strftime("%A, %B %d, %Y at %I:%M %p %Z")


def search_campus_info(query: str) -> str:
    return retrieve_context(query, k=3)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in an IANA time zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA time zone, e.g. America/Los_Angeles or UTC.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_campus_info",
            "description": "Search the USC campus assistant knowledge base for information about USC clubs (including AI Club), labs (GPU lab, robotics lab), workshops, faculty office hours, peer tutoring, and the NVIDIA Developer Program at USC. Always call this for any USC-related question — do not answer from your own knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The USC campus question or search phrase.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

available_tools = {
    "get_current_time": get_current_time,
    "search_campus_info": search_campus_info,
}


def ask_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\n\nYou are a USC campus assistant with two tools: "
                "get_current_time and search_campus_info. "
                "When the user asks something a tool can answer, call the tool, "
                "then write the final answer based on the tool's result. "
                "Do not call the same tool twice for the same question. "
                "If after using the tools you still cannot find the answer, "
                "reply exactly: I don't have that information — check with the USC AI Club."
            ),
        },
        {"role": "user", "content": question},
    ]

    for _ in range(3):  # hard cap on tool-call iterations
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=400,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or "I could not generate an answer. Please try again."

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")

            if name not in available_tools:
                result = f"Tool {name} is not available."
            else:
                result = available_tools[name](**arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": str(result),
            })

    return "I hit the tool loop limit."


# Demo run ─────────────────────────────────────────────────────────────────────
print("\n── Workshop 5: agent with clock + USC knowledge base ──")
for question in [
    "What time is it in Los Angeles?",          # → uses get_current_time
    "When does the USC AI Club meet?",          # → uses search_campus_info
    "Can I get the wifi password?",             # → searches, finds nothing, refuses
]:
    print(f"\nQ: {question}")
    print(f"A: {ask_agent(question)}")
