"""
Workshop 6 — Multi-Step Agent: Reason, Act, Observe with NVIDIA NIM
Self-contained — bundles the retriever from Workshop 2 and the tool-calling
setup from Workshop 5, then goes one level deeper.

Workshop 5 taught the model to pick ONE tool and answer. That's a chatbot with
hands. A real agent has to CHAIN tools: think, call a tool, read the result,
decide what to do next, and only then answer. That loop is the ReAct pattern
(Reason + Act). This script makes the loop visible so you can watch the agent
work through a question one step at a time.

Three tools — the retriever from Workshop 2, a clock, and a date calculator —
are enough to force genuine multi-step reasoning. A question like "how many
days until the next AI Club meeting?" can't be answered by any single tool:
the agent must search for the meeting day, then compute the days until it.
"""

import os
import json
from datetime import datetime, timedelta
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

MODEL = "meta/llama-3.3-70b-instruct"   # same as Workshop 5 — the 70B model is
                                         # far more reliable once the agent has to
                                         # choose AND sequence multiple tools.
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

LOCAL_TZ = "America/Los_Angeles"        # USC campus time zone — used by the clock
                                         # and date tools so "today" is consistent.


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
print(f"Embedded {len(knowledge_base)} chunks. Model: {MODEL}")


# ── Workshop 6 starts here — three tools, one of which forces chaining ────────
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_current_time(timezone: str = LOCAL_TZ) -> str:
    """Return the current date and time in an IANA time zone."""
    try:
        zone = ZoneInfo(timezone)
    except Exception:
        zone = ZoneInfo("UTC")
    return datetime.now(zone).strftime("%A, %B %d, %Y at %I:%M %p %Z")


def search_campus_info(query: str) -> str:
    """Reuse the Workshop 2 retriever — the agent gets semantic search for free."""
    return retrieve_context(query, k=3)


def days_until_weekday(weekday: str) -> str:
    """How many days from today until the next occurrence of a given weekday.

    This is the tool that forces multi-step reasoning. To answer "how many days
    until the AI Club meeting?", the agent first has to search the knowledge base
    to learn the meeting is on Thursday, THEN call this tool with weekday=Thursday.
    No single tool can do it alone.
    """
    target = weekday.strip().capitalize()
    if target not in WEEKDAYS:
        return (
            f"'{weekday}' is not a valid weekday. "
            f"Use one of: {', '.join(WEEKDAYS)}."
        )

    today = datetime.now(ZoneInfo(LOCAL_TZ))
    delta = (WEEKDAYS.index(target) - today.weekday()) % 7
    date_str = (today + timedelta(days=delta)).strftime("%B %d, %Y")

    if delta == 0:
        return f"Today is {target} ({date_str}) — that is 0 days away (it's today)."
    return f"The next {target} is in {delta} day(s), on {date_str}. Today is {today.strftime('%A')}."


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_campus_info",
            "description": (
                "Search the USC campus knowledge base for facts about USC clubs "
                "(including the AI Club), labs (GPU lab, robotics lab), workshops, "
                "faculty office hours, peer tutoring, and the NVIDIA Developer "
                "Program. Use this to find WHEN or WHERE something happens. Always "
                "call this for any USC-specific fact — do not answer from your own "
                "knowledge."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date, day of week, and time in an IANA time zone. "
                "Use this when the question depends on what day or time it is right now "
                "(e.g. 'is the lab open right now?')."
            ),
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
            "name": "days_until_weekday",
            "description": (
                "Calculate how many days from today until the next occurrence of a "
                "given weekday (e.g. 'Thursday'). Use this AFTER you know which day "
                "an event happens, to work out how far away it is. You usually have "
                "to call search_campus_info first to find the day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "weekday": {
                        "type": "string",
                        "description": "A weekday name, e.g. Monday, Thursday, Sunday.",
                    },
                },
                "required": ["weekday"],
            },
        },
    },
]

available_tools = {
    "search_campus_info": search_campus_info,
    "get_current_time": get_current_time,
    "days_until_weekday": days_until_weekday,
}

SYSTEM_PROMPT = (
    "You are a USC campus assistant that solves questions step by step using "
    "tools. You have three tools: search_campus_info (find USC facts), "
    "get_current_time (today's date and time), and days_until_weekday (days "
    "from today until a weekday).\n\n"
    "Work in a loop: think about what you still need, call ONE tool to get it, "
    "read the result, then decide whether you can answer or need another tool. "
    "Many questions need more than one tool — for example, to find how many days "
    "until an event, first search for the day it happens, then call "
    "days_until_weekday with that day.\n\n"
    "Only answer once you have gathered every fact you need from the tools. "
    "Base your final answer strictly on tool results, never on your own "
    "assumptions about USC. If the tools cannot give you the answer, reply "
    "exactly: I don't have that information — check with the USC AI Club."
)

MAX_STEPS = 5   # multi-step questions need more room than Workshop 5's cap of 3.


def run_agent(question: str, verbose: bool = True) -> str:
    """ReAct loop: the model reasons, acts (calls a tool), observes the result,
    and repeats until it can answer. Set verbose=True to print the trace."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
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

        # No tool call → the model is done and this is the final answer.
        if not message.tool_calls:
            return message.content

        # Otherwise: run every tool the model asked for, feed results back.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if name not in available_tools:
                result = f"Tool '{name}' is not available."
            else:
                try:
                    result = available_tools[name](**arguments)
                except Exception as exc:  # never let a bad tool call crash the loop
                    result = f"Tool '{name}' failed: {exc}"

            if verbose:
                print(f"  step {step} · acting  → {name}({json.dumps(arguments)})")
                print(f"  step {step} · observe ← {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": str(result),
            })

    return "I reached the step limit before finishing — try asking a narrower question."


# Demo run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Workshop 6: multi-step ReAct agent over USC campus info ──")
    for question in [
        # Two tools, in order: search (find "Thursday") → days_until_weekday.
        "How many days until the next USC AI Club meeting?",
        # Two tools: get_current_time (what day/hour is it) + search (lab hours), then reason.
        "Is the USC GPU lab open right now?",
        # One tool is enough — the agent should NOT over-call.
        "When does the USC AI Club meet?",
        # No tool can answer — the agent should search, find nothing, and refuse.
        "What is the campus wifi password?",
    ]:
        print(f"\nQ: {question}")
        answer = run_agent(question, verbose=True)
        print(f"A: {answer}")
