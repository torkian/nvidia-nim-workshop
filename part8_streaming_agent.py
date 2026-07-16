"""
Workshop 8 — Streaming: Make the Assistant Feel Real-Time with NVIDIA NIM
Self-contained — carries the multi-turn ChatSession from Workshop 7 and adds
ONE capability: streaming. Instead of waiting for the whole answer and printing
it at once, we print each token as the model generates it — the difference
between a script that hangs for three seconds and a product that feels alive.

The flag is trivial: stream=True on the create() call. The lesson is what the
stream gives you back. A streamed response arrives as many small CHUNKS, not one
message. Text comes in as `delta.content` pieces you concatenate. Tool calls are
trickier — the name and the arguments JSON arrive split across several chunks,
identified by an index — so you reassemble them by index before running them.

The agent loop itself does not change. Streaming is a data-accumulation layer on
top of the exact Workshop 7 loop: stream a turn, see whether it produced tool
calls or a final answer, and either run the tools or stop. Same control flow.
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

MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
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


def run_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool by name, defensively. Shared by chat() and stream()."""
    if name not in available_tools:
        return f"Tool '{name}' is not available."
    try:
        return available_tools[name](**arguments)
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"


class ChatSession:
    """A conversation with memory (Workshop 7) that can also stream (Workshop 8).

    .chat()   — returns the full answer at once (Workshop 7 behavior).
    .stream() — prints the answer token-by-token as the model generates it.

    Both share the same persistent history (self.messages) and the same loop.
    """

    def __init__(self, max_turns: int = 8, verbose: bool = True):
        self.system = {"role": "system", "content": SYSTEM_PROMPT}
        self.messages = [self.system]
        self.max_turns = max_turns
        self.verbose = verbose

    def reset(self) -> None:
        self.messages = [self.system]

    def _trim(self) -> None:
        # Keep system + last max_turns turns; cut only at a user-message boundary
        # so a tool result is never orphaned from its tool call (see Workshop 7).
        user_indices = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if len(user_indices) <= self.max_turns:
            return
        cut = user_indices[-self.max_turns]
        self.messages = [self.system] + self.messages[cut:]

    # ── Workshop 7: non-streaming ─────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for step in range(1, MAX_STEPS + 1):
            response = client.chat.completions.create(
                model=MODEL, messages=self.messages, tools=tools,
                tool_choice="auto", temperature=0.2, max_tokens=400,
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                self._trim()
                return message.content or "I could not generate an answer. Please try again."

            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = run_tool(tool_call.function.name, arguments)
                if self.verbose:
                    print(f"  step {step} · {tool_call.function.name}"
                          f"({json.dumps(arguments)}) -> {result}")
                self.messages.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "name": tool_call.function.name, "content": str(result),
                })

        fallback = "I reached the step limit before finishing — try asking a narrower question."
        self.messages.append({"role": "assistant", "content": fallback})
        self._trim()
        return fallback

    # ── Workshop 8: streaming ─────────────────────────────────────────────────
    def stream(self, user_message: str) -> str:
        """Same loop as chat(), but each turn is streamed. The model's text is
        printed token-by-token as it arrives; tool-call fragments are reassembled
        by index, then the turn ends exactly like chat()."""
        self.messages.append({"role": "user", "content": user_message})

        for step in range(1, MAX_STEPS + 1):
            stream_resp = client.chat.completions.create(
                model=MODEL, messages=self.messages, tools=tools,
                tool_choice="auto", temperature=0.2, max_tokens=400,
                stream=True,
            )

            text_parts = []
            tool_fragments = {}     # index -> {"id", "name", "arguments"}
            header_printed = False

            for chunk in stream_resp:
                if not chunk.choices:            # e.g. a trailing usage-only chunk
                    continue
                delta = chunk.choices[0].delta

                if delta.content:                # a piece of the visible answer
                    if not header_printed:
                        print("Assistant: ", end="", flush=True)
                        header_printed = True
                    print(delta.content, end="", flush=True)
                    text_parts.append(delta.content)

                for tc in (delta.tool_calls or []):   # a fragment of a tool call
                    slot = tool_fragments.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id and not slot["id"]:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name and not slot["name"]:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

            if header_printed:
                print()     # newline once the streamed answer finishes

            text = "".join(text_parts)
            tool_calls = [tool_fragments[i] for i in sorted(tool_fragments)]

            # Rebuild the assistant message from the streamed pieces, then store it.
            # Mirror chat()'s exclude_none shape: omit content when it's only tools.
            assistant_msg = {"role": "assistant"}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ]
                if text:
                    assistant_msg["content"] = text
            else:
                assistant_msg["content"] = text
            self.messages.append(assistant_msg)

            if not tool_calls:                  # final answer already streamed
                self._trim()
                return text or "I could not generate an answer. Please try again."

            for tc in tool_calls:               # run tools, then loop and stream again
                try:
                    arguments = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = run_tool(tc["name"], arguments)
                if self.verbose:
                    print(f"  step {step} · {tc['name']}({json.dumps(arguments)}) -> {result}")
                self.messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": tc["name"], "content": str(result),
                })

        fallback = "I reached the step limit before finishing — try asking a narrower question."
        print(f"Assistant: {fallback}")
        self.messages.append({"role": "assistant", "content": fallback})
        self._trim()
        return fallback


# Demo run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Without streaming (Workshop 7 behavior: answer arrives all at once) ──")
    session = ChatSession(verbose=True)
    print("You:       What are the USC GPU lab hours?")
    print(f"Assistant: {session.chat('What are the USC GPU lab hours?')}")

    print("\n── With streaming (Workshop 8: watch the answer appear token by token) ──")
    for user_message in [
        "When does the USC AI Club meet?",          # tool call, then the answer streams
        "How many days until that?",                # memory + tool, then streams
        "Which is sooner, that meeting or the AI/ML office hours?",  # multi-step, then streams
    ]:
        print(f"\nYou:       {user_message}")
        session.stream(user_message)                # prints "Assistant: ..." itself
