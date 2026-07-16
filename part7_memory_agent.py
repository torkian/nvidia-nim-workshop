"""
Workshop 7 — Memory: A Multi-Turn Conversational Agent with NVIDIA NIM
Self-contained — carries the retriever from Workshop 2 and the three tools from
Workshop 6, then adds the one thing every real assistant needs: memory.

Workshops 5 and 6 answered ONE question and forgot everything. Ask a follow-up
like "how many days until that?" and the agent had no idea what "that" meant.
This workshop fixes that by keeping the conversation history alive across turns.

The whole trick is small: in Workshop 6 the `messages` list lived inside the
agent function and was thrown away after each question. Here we lift that list
out of the function and into a ChatSession object, so it survives from one turn
to the next. The model sees the whole conversation and can resolve references
like "that", "those two", or "the second one" from what was already said.

One rule makes multi-turn tool calling safe: never split a turn when trimming
old history. An assistant message that asked for a tool MUST keep its matching
tool-result messages, or the API rejects the orphaned tool_call_id. We trim only
at user-message boundaries, which always removes whole turns.
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

MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"   # same as Workshops 5-6 — reliable tool calling,
                                         # and steadier across turns at low temperature.
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

LOCAL_TZ = "America/Los_Angeles"


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


# ── The three tools from Workshop 6 (unchanged) ───────────────────────────────
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_current_time(timezone: str = LOCAL_TZ) -> str:
    try:
        zone = ZoneInfo(timezone)
    except Exception:
        zone = ZoneInfo("UTC")
    return datetime.now(zone).strftime("%A, %B %d, %Y at %I:%M %p %Z")


def search_campus_info(query: str) -> str:
    return retrieve_context(query, k=3)


def days_until_weekday(weekday: str) -> str:
    target = weekday.strip().capitalize()
    if target not in WEEKDAYS:
        return f"'{weekday}' is not a valid weekday. Use one of: {', '.join(WEEKDAYS)}."
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
                "Use this when the question depends on what day or time it is right now."
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
                "an event happens, to work out how far away it is."
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
    "/no_think\n\nYou are a USC campus assistant having an ongoing conversation with a student. "
    "You remember everything said earlier in this conversation.\n\n"
    "When a question refers back to something already discussed — words like 'that', "
    "'those', 'then', 'it', or 'the second one' — resolve the reference from the "
    "conversation so far before doing anything else.\n\n"
    "You have three tools: search_campus_info (find USC facts), get_current_time "
    "(today's date and time), and days_until_weekday (days from today until a weekday). "
    "Work step by step: decide what you still need, call ONE tool, read the result, "
    "then continue. Before calling a tool, check whether the conversation ALREADY "
    "contains the fact you need — do not re-search for something you found a turn ago.\n\n"
    "To compare how soon two days are, call days_until_weekday for EACH day and compare "
    "the numbers it returns — never estimate the number of days yourself.\n\n"
    "Base every answer strictly on tool results and what was said in the conversation, "
    "never on your own assumptions about USC. If you cannot find the answer, reply "
    "exactly: I don't have that information — check with the USC AI Club."
)

MAX_STEPS = 5   # tool-call iterations allowed within a single turn


class ChatSession:
    """A conversation with memory. The messages list lives here, on the object,
    instead of inside a function — so it persists from one turn to the next.

    Each call to .chat() runs the Workshop 6 tool loop, but against the FULL
    history, then appends the result so the next turn can see it.
    """

    def __init__(self, max_turns: int = 8, verbose: bool = True):
        self.system = {"role": "system", "content": SYSTEM_PROMPT}
        self.messages = [self.system]
        self.max_turns = max_turns      # how many recent user turns to keep
        self.verbose = verbose

    def reset(self) -> None:
        """Forget the conversation — start fresh with only the system prompt."""
        self.messages = [self.system]

    def _trim(self) -> None:
        """Keep the system prompt + the last `max_turns` turns.

        A 'turn' starts at a user message and includes every assistant/tool message
        up to the next user message. We cut ONLY at a user-message boundary, which
        removes whole turns and can never orphan a tool result from its tool call.
        """
        user_indices = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if len(user_indices) <= self.max_turns:
            return
        cut = user_indices[-self.max_turns]          # first index we want to keep
        dropped = len(user_indices) - self.max_turns
        self.messages = [self.system] + self.messages[cut:]
        if self.verbose:
            print(f"  (memory: dropped {dropped} old turn(s), keeping last {self.max_turns})")

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for step in range(1, MAX_STEPS + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=400,
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:        # final answer for this turn
                self._trim()
                return message.content

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
                    except Exception as exc:
                        result = f"Tool '{name}' failed: {exc}"

                if self.verbose:
                    print(f"  step {step} · acting  -> {name}({json.dumps(arguments)})")
                    print(f"  step {step} · observe <- {result}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": str(result),
                })

        fallback = "I reached the step limit before finishing — try asking a narrower question."
        self.messages.append({"role": "assistant", "content": fallback})
        self._trim()
        return fallback


# Demo run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Workshop 7: a conversation the agent actually remembers ──")
    session = ChatSession(verbose=True)
    conversation = [
        "When does the USC AI Club meet?",          # turn 1: search → "Thursday"
        "How many days until that?",                # turn 2: "that" = Thursday (from memory)
        "And when are the AI/ML faculty office hours?",  # turn 3: search → "Tuesday"
        "Which of those two is sooner?",            # turn 4: compares both remembered facts
    ]
    for user_message in conversation:
        print(f"\nYou:       {user_message}")
        print(f"Assistant: {session.chat(user_message)}")

    # Memory is per-session. Clear it and the follow-up has nothing to refer back to.
    print("\n── After reset(): the same follow-up has no context to resolve ──")
    session.reset()
    print("\nYou:       How many days until that?")
    print(f"Assistant: {session.chat('How many days until that?')}")
