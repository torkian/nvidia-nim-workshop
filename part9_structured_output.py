"""
Workshop 9 — Structured Outputs: Make the Agent Return Data, Not Just Prose
Self-contained — carries the streaming ChatSession from Workshop 8 and adds ONE
capability: the final answer is now a validated JSON object instead of a
paragraph. That turns the assistant from something a human reads into something
other software can consume — the foundation the rest of the book builds on
(evals assert on fields, an API returns this body, logs store it).

The honest part: NVIDIA NIM exposes an OpenAI-compatible API, but strict schema
enforcement (response_format with a json_schema) is NOT reliable on hosted open
models like llama-3.3-70b, and it interacts badly with tool calling. So we don't
depend on it. We ask for JSON in the prompt, then do the robust thing ourselves:
parse it, validate it against our contract, and if it's wrong, make ONE repair
call. If even that fails, we return a deterministic error object. No framework.

The tools, memory, trim-by-turns, and streaming accumulator are unchanged
(live token printing is dropped: half-formed JSON is useless to display). Two
things are new in the code: streaming and non-streaming now share one loop
(_complete + _run_turn, instead of Workshop 8's two near-identical loops), and the
final-answer boundary parses/validates/repairs the reply into JSON.
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

MODEL = "meta/llama-3.3-70b-instruct"
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
                    "query": {"type": "string",
                              "description": "The USC campus question or search phrase."},
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
                    "timezone": {"type": "string",
                                 "description": "IANA time zone, e.g. America/Los_Angeles or UTC."},
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
                    "weekday": {"type": "string",
                                "description": "A weekday name, e.g. Monday, Thursday, Sunday."},
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


def run_tool(name: str, arguments: dict) -> str:
    if name not in available_tools:
        return f"Tool '{name}' is not available."
    try:
        return available_tools[name](**arguments)
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"


# ── Workshop 9: the JSON answer contract ──────────────────────────────────────
STATUSES = {"answered", "not_found", "needs_clarification"}
CATEGORIES = {"campus_event", "campus_hours", "campus_resource", "comparison", "refusal"}
REQUIRED_KEYS = ("status", "answer", "category", "items", "missing", "sources")

SYSTEM_PROMPT = (
    "You are a USC campus assistant having an ongoing conversation with a student. "
    "You remember everything said earlier in this conversation.\n\n"
    "When a question refers back to something already discussed — words like 'that', "
    "'those', 'then', 'it', or 'the second one' — resolve the reference from the "
    "conversation so far before doing anything else.\n\n"
    "You have three tools: search_campus_info (find USC facts), get_current_time "
    "(today's date and time), and days_until_weekday (days from today until a weekday). "
    "Work step by step: decide what you still need, call ONE tool, read the result, then "
    "continue. Before calling a tool, check whether the conversation ALREADY contains the "
    "fact you need — do not re-search for something you found a turn ago. To compare how "
    "soon two days are, call days_until_weekday for EACH day and compare the numbers it "
    "returns — never estimate the number of days yourself.\n\n"
    "FINAL ANSWER FORMAT. When you are done using tools, your final reply MUST be a single "
    "JSON object and NOTHING else — no prose before or after, no code fences. Use exactly "
    "these keys:\n"
    '  "status": one of "answered", "not_found", or "needs_clarification"\n'
    '  "answer": a one or two sentence plain-language answer for a human\n'
    '  "category": one of "campus_event", "campus_hours", "campus_resource", "comparison", "refusal"\n'
    '  "items": a list of objects holding the specific facts (name, day, time, location, days_until as relevant); may be empty\n'
    '  "missing": a list of things the user asked for that you could not find; may be empty\n'
    '  "sources": a list of the exact knowledge-base lines you used; may be empty\n'
    "Base everything strictly on tool results and the conversation. If you cannot find the "
    'answer, use status "not_found", category "refusal", put what is missing in "missing", '
    'and set "answer" to: I don\'t have that information — check with the USC AI Club.'
)

MAX_STEPS = 5


def parse_json_object(text: str) -> dict:
    """Pull a JSON object out of the model's text and parse it.

    Models sometimes wrap JSON in prose or fenced code blocks, so we take the span
    from the first '{' to the last '}'. Raises ValueError/JSONDecodeError if there
    is no parseable object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])


def validate_answer(data) -> list:
    """Return a list of human-readable problems. Empty list means valid."""
    if not isinstance(data, dict):
        return ["response is not a JSON object"]
    errors = []
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"missing required key: {key}")
    if data.get("status") not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    if data.get("category") not in CATEGORIES:
        errors.append(f"category must be one of {sorted(CATEGORIES)}")
    if "answer" in data and not isinstance(data["answer"], str):
        errors.append("answer must be a string")
    for key in ("items", "missing", "sources"):
        if key in data and not isinstance(data[key], list):
            errors.append(f"{key} must be a list")
    if isinstance(data.get("items"), list) and not all(isinstance(it, dict) for it in data["items"]):
        errors.append("each entry in items must be an object")
    return errors


def repair_answer_json(raw_text: str, errors: list) -> dict | None:
    """One bounded repair attempt: ask the model to fix its own JSON. Returns a
    valid dict, or None if the call fails or the result still doesn't pass."""
    try:
        fix = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=800,
            messages=[
                {"role": "system", "content": (
                    "You fix malformed JSON. Return ONLY a single valid JSON object — "
                    "no prose, no code fences.")},
                {"role": "user", "content": (
                    f"This response was supposed to match a schema but had these problems: "
                    f"{errors}.\n\nRequired keys: status (one of {sorted(STATUSES)}), answer "
                    f"(string), category (one of {sorted(CATEGORIES)}), items (list), missing "
                    f"(list), sources (list).\n\nPreserve all facts. Here is the original:\n"
                    f"{raw_text}\n\nReturn the corrected JSON object only.")},
            ],
        )
        data = parse_json_object(fix.choices[0].message.content or "")
    except Exception:           # API error, bad JSON — fall back deterministically
        return None
    return data if not validate_answer(data) else None


def format_error(missing: str = "valid_json", *, answer: str | None = None) -> dict:
    """Deterministic fallback when the model can't produce a usable answer.
    Pass `answer` for a cause-specific message (e.g. the step-limit case)."""
    return {
        "status": "needs_clarification",
        "answer": answer or "I couldn't produce a valid structured response. Please ask again.",
        "category": "refusal",
        "items": [],
        "missing": [missing],
        "sources": [],
    }


class ChatSession:
    """Workshop 8's streaming, multi-turn session — but chat() and stream() now
    return a validated JSON dict instead of a string. Tools, memory, and
    trim-by-turns are identical; Workshop 8's two loops are factored into a shared
    _complete + _run_turn, and the final-answer boundary now produces validated JSON."""

    def __init__(self, max_turns: int = 8, verbose: bool = True):
        self.system = {"role": "system", "content": SYSTEM_PROMPT}
        self.messages = [self.system]
        self.max_turns = max_turns
        self.verbose = verbose

    def reset(self) -> None:
        self.messages = [self.system]

    def _trim(self) -> None:
        user_indices = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if len(user_indices) <= self.max_turns:
            return
        cut = user_indices[-self.max_turns]
        self.messages = [self.system] + self.messages[cut:]

    def _complete(self, stream: bool):
        """One model call. Returns (text, tool_calls, assistant_msg) where
        tool_calls is a normalized list of {id, name, arguments} and assistant_msg
        is the dict to append to history. Streaming and non-streaming converge here."""
        if not stream:
            resp = client.chat.completions.create(
                model=MODEL, messages=self.messages, tools=tools,
                tool_choice="auto", temperature=0.2, max_tokens=800,
            )
            msg = resp.choices[0].message
            tool_calls = [{"id": tc.id, "name": tc.function.name,
                           "arguments": tc.function.arguments or "{}"}
                          for tc in (msg.tool_calls or [])]
            return (msg.content or ""), tool_calls, msg.model_dump(exclude_none=True)

        stream_resp = client.chat.completions.create(
            model=MODEL, messages=self.messages, tools=tools,
            tool_choice="auto", temperature=0.2, max_tokens=800, stream=True,
        )
        text_parts, fragments = [], {}
        for chunk in stream_resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
            for tc in (delta.tool_calls or []):
                slot = fragments.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id and not slot["id"]:
                    slot["id"] = tc.id
                if tc.function and tc.function.name and not slot["name"]:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        text = "".join(text_parts)
        tool_calls = [fragments[i] for i in sorted(fragments)]
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
        return text, tool_calls, assistant_msg

    def _finalize_json(self, raw_text: str) -> dict:
        """Turn the model's final text into a validated dict: parse -> validate ->
        repair once -> deterministic fallback."""
        if not raw_text.strip():
            # An empty reply is a real llama failure mode. Don't send it to repair —
            # with nothing to preserve, the repair model would invent facts.
            return format_error("empty_model_response")
        try:
            data = parse_json_object(raw_text)
            errors = validate_answer(data)
        except (ValueError, json.JSONDecodeError):
            data, errors = None, ["response was not valid JSON"]
        if errors:
            if self.verbose:
                print(f"  (final JSON invalid: {errors} — attempting one repair)")
            repaired = repair_answer_json(raw_text, errors)
            data = repaired if repaired is not None else format_error()
        return data

    def _run_turn(self, user_message: str, stream: bool) -> dict:
        checkpoint = len(self.messages)     # so a failed turn can be rolled back
        self.messages.append({"role": "user", "content": user_message})
        try:
            return self._run_turn_inner(stream)
        except BaseException:
            # Roll back the half-finished turn: an assistant tool_calls message
            # with no tool replies leaves the history invalid for the next call.
            del self.messages[checkpoint:]
            raise

    def _run_turn_inner(self, stream: bool) -> dict:
        for step in range(1, MAX_STEPS + 1):
            text, tool_calls, assistant_msg = self._complete(stream)

            if tool_calls:                       # still gathering facts
                self.messages.append(assistant_msg)
                for tc in tool_calls:
                    try:
                        arguments = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = run_tool(tc["name"], arguments)
                    if self.verbose:
                        print(f"  step {step} · {tc['name']}({json.dumps(arguments)}) -> {result}")
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                          "name": tc["name"], "content": str(result)})
                continue

            # Final answer → validated JSON. Store the canonical JSON in history.
            data = self._finalize_json(text)
            self.messages.append({"role": "assistant", "content": json.dumps(data)})
            self._trim()
            return data

        data = format_error(
            "step_limit_reached",
            answer="I reached the step limit before finishing — try asking a narrower question.",
        )
        self.messages.append({"role": "assistant", "content": json.dumps(data)})
        self._trim()
        return data

    def chat(self, user_message: str) -> dict:
        return self._run_turn(user_message, stream=False)

    def stream(self, user_message: str) -> dict:
        # Streaming still works and composes — but partial JSON is useless to a
        # caller, so we accumulate it and parse only when the stream completes.
        return self._run_turn(user_message, stream=True)


# Demo run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Workshop 9: the agent now returns structured JSON ──")
    session = ChatSession(verbose=True)
    for user_message in [
        "When does the USC AI Club meet?",                          # answered, campus_event
        "How many days until that?",                                # memory + tool
        "Which is sooner, that meeting or the AI/ML office hours?",  # comparison
        "What is the campus wifi password?",                        # not_found / refusal
    ]:
        print(f"\nYou: {user_message}")
        result = session.chat(user_message)
        print("Structured response:")
        print(json.dumps(result, indent=2))
