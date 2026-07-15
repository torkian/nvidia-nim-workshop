"""
Workshop 10 — Traces: See What Your Agent Did (and Why)
Self-contained — carries the structured-output ChatSession from Workshop 9 and
adds ONE capability: every turn writes a trace record to a JSONL file. When an
agent misbehaves, the question is always "why did it do that?" — and the trace
answers it after the fact: which tools it called, with what arguments, what came
back, how long each step took, and whether the JSON had to be repaired.

One line per turn, with a `steps` array inside. That shape is deliberate:
Workshop 11 (evals) will load one line and get the input, the tool path, the
validation outcome, and the final six-key answer together — everything an
assertion needs, no regrouping required. `cat` one line into json.tool and the
whole turn reads like a story.

Two honesty notes baked in:
  - Token usage is best-effort. Non-streaming NIM responses usually include
    response.usage; streaming often doesn't. When it's absent we log null —
    never fake zeros. This is accounting, not billing-grade metering.
  - App-side traces work everywhere — hosted API Catalog and local NIM alike.
    Server-side Prometheus metrics (:8000/v1/metrics) exist ONLY on
    self-hosted NIM containers; the hosted endpoint doesn't expose them.

The privacy rule this workshop teaches: traces contain user messages, tool
results, and answers — real deployments hold names, emails, IDs. Never log API
keys, headers, or the full messages array. Keep trace files out of git.
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
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
TRACE_PATH = Path(__file__).resolve().parent / "traces" / "campus_assistant.jsonl"
# ^ anchored to this script so traces always land in the repo's gitignored traces/


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


# ── The Workshop 9 answer contract (unchanged) ────────────────────────────────
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
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])


def validate_answer(data) -> list:
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
    except Exception:
        return None
    return data if not validate_answer(data) else None


def format_error(missing: str = "valid_json", *, answer: str | None = None) -> dict:
    return {
        "status": "needs_clarification",
        "answer": answer or "I couldn't produce a valid structured response. Please ask again.",
        "category": "refusal",
        "items": [],
        "missing": [missing],
        "sources": [],
    }


# ── Workshop 10 starts here: the tracer ───────────────────────────────────────
class JsonlTracer:
    """Writes one JSON line per completed turn. Plain file I/O — no framework.

    The privacy rule: a trace holds the user's message, tool results, and the
    final answer — in a real deployment that's personal data. Never log API
    keys, request headers, or the whole self.messages array. And keep the
    trace directory out of git (see .gitignore).
    """

    SCHEMA_VERSION = "ws10.turn.v1"

    def __init__(self, path: Path = TRACE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_id = uuid.uuid4().hex[:12]     # one id per session
        self.turn_id = 0
        self._turn = None                          # the record being built

    def begin_turn(self, user_message: str, mode: str) -> None:
        if self._turn is not None:          # a prior turn never closed — keep the evidence
            self.abort_turn("turn abandoned without end_turn")
        self.turn_id += 1
        self._turn = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "turn_id": self.turn_id,
            "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
            "model": MODEL,
            "mode": mode,                          # "chat" or "stream"
            "user_message": user_message,
            "steps": [],
            "final": None,
            "total_latency_ms": None,
        }
        self._start = time.perf_counter()

    def record_model_call(self, step: int, latency_ms: int, usage, tool_calls) -> None:
        self._turn["steps"].append({
            "type": "model_call",
            "step": step,
            "latency_ms": latency_ms,
            # usage is best-effort: null when the endpoint didn't report it.
            "usage": usage,
            "tool_calls": [
                {"id": tc["id"], "name": tc["name"], "arguments": tc["arguments"]}
                for tc in tool_calls
            ],
        })

    def record_tool_call(self, step: int, tool_call_id: str, name: str,
                         arguments: dict, result: str, latency_ms: int) -> None:
        self._turn["steps"].append({
            "type": "tool_call",
            "step": step,
            "tool_call_id": tool_call_id,
            "name": name,
            "arguments": arguments,
            "result": result,
            "latency_ms": latency_ms,
        })

    def record_validation(self, parse_ok: bool, errors: list,
                          repair_attempted: bool, repair_succeeded: bool) -> None:
        self._turn["steps"].append({
            "type": "validation",
            "parse_ok": parse_ok,
            "errors": errors,
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
        })

    def end_turn(self, final: dict) -> None:
        self._turn["final"] = final
        self._turn["total_latency_ms"] = int((time.perf_counter() - self._start) * 1000)
        with self.path.open("a") as f:
            f.write(json.dumps(self._turn) + "\n")
        self._turn = None

    def abort_turn(self, error: str) -> None:
        """A crash mid-turn still leaves a trace — that's when you need one most."""
        if self._turn is None:
            return
        self._turn["error"] = error
        self.end_turn(final=None)


class ChatSession:
    """Workshop 9's session, now traced. The loop is unchanged — the tracer is
    bolted onto its five seams: turn start, each model call, each tool call,
    validation, and the final answer."""

    def __init__(self, max_turns: int = 8, verbose: bool = True,
                 trace_path: Path = TRACE_PATH):
        self.system = {"role": "system", "content": SYSTEM_PROMPT}
        self.messages = [self.system]
        self.max_turns = max_turns
        self.verbose = verbose
        self.tracer = JsonlTracer(trace_path)      # ← new in Workshop 10

    def reset(self) -> None:
        self.messages = [self.system]

    def _trim(self) -> None:
        user_indices = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if len(user_indices) <= self.max_turns:
            return
        cut = user_indices[-self.max_turns]
        self.messages = [self.system] + self.messages[cut:]

    def _complete(self, stream: bool):
        """One model call. New in Workshop 10: also returns best-effort usage
        (a dict with prompt/completion/total tokens, or None when the endpoint
        didn't report it — never fake zeros)."""
        if not stream:
            resp = client.chat.completions.create(
                model=MODEL, messages=self.messages, tools=tools,
                tool_choice="auto", temperature=0.2, max_tokens=800,
            )
            msg = resp.choices[0].message
            tool_calls = [{"id": tc.id, "name": tc.function.name,
                           "arguments": tc.function.arguments or "{}"}
                          for tc in (msg.tool_calls or [])]
            usage = None
            if getattr(resp, "usage", None):
                usage = {"prompt_tokens": resp.usage.prompt_tokens,
                         "completion_tokens": resp.usage.completion_tokens,
                         "total_tokens": resp.usage.total_tokens}
            return (msg.content or ""), tool_calls, msg.model_dump(exclude_none=True), usage

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
        # Streamed responses usually don't carry usage — log null, don't guess.
        return text, tool_calls, assistant_msg, None

    def _finalize_json(self, raw_text: str):
        """New in Workshop 10: returns (data, validation_meta) so the tracer can
        record what happened without tangling the parsing logic."""
        meta = {"parse_ok": True, "errors": [], "repair_attempted": False,
                "repair_succeeded": False}
        if not raw_text.strip():
            # An empty reply is a real llama failure mode. Don't send it to repair —
            # with nothing to preserve, the repair model would invent facts.
            meta["parse_ok"] = False
            meta["errors"] = ["empty model response"]
            return format_error("empty_model_response"), meta
        try:
            data = parse_json_object(raw_text)
            meta["errors"] = validate_answer(data)
        except (ValueError, json.JSONDecodeError):
            data, meta["parse_ok"] = None, False
            meta["errors"] = ["response was not valid JSON"]
        if meta["errors"]:
            if self.verbose:
                print(f"  (final JSON invalid: {meta['errors']} — attempting one repair)")
            meta["repair_attempted"] = True
            repaired = repair_answer_json(raw_text, meta["errors"])
            meta["repair_succeeded"] = repaired is not None
            data = repaired if repaired is not None else format_error()
        return data, meta

    def _run_turn(self, user_message: str, stream: bool) -> dict:
        checkpoint = len(self.messages)     # so a failed turn can be rolled back
        self.messages.append({"role": "user", "content": user_message})
        self.tracer.begin_turn(user_message, mode="stream" if stream else "chat")
        try:
            return self._run_turn_traced(stream)
        except BaseException as exc:        # BaseException: Ctrl-C must leave a trace too
            # Write the partial trace before the exception propagates — a crash
            # mid-turn is exactly the moment you'll want the record. Then roll the
            # history back to the checkpoint so the session stays usable: a half
            # turn (assistant tool_calls with no tool replies) breaks later calls.
            self.tracer.abort_turn(f"{type(exc).__name__}: {exc}")
            del self.messages[checkpoint:]
            raise

    def _run_turn_traced(self, stream: bool) -> dict:
        for step in range(1, MAX_STEPS + 1):
            t0 = time.perf_counter()
            text, tool_calls, assistant_msg, usage = self._complete(stream)
            self.tracer.record_model_call(
                step, int((time.perf_counter() - t0) * 1000), usage, tool_calls)

            if tool_calls:
                self.messages.append(assistant_msg)
                for tc in tool_calls:
                    try:
                        arguments = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    t1 = time.perf_counter()
                    result = run_tool(tc["name"], arguments)
                    self.tracer.record_tool_call(
                        step, tc["id"], tc["name"], arguments, str(result),
                        int((time.perf_counter() - t1) * 1000))
                    if self.verbose:
                        print(f"  step {step} · {tc['name']}({json.dumps(arguments)}) -> {result}")
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                          "name": tc["name"], "content": str(result)})
                continue

            data, vmeta = self._finalize_json(text)
            self.tracer.record_validation(vmeta["parse_ok"], vmeta["errors"],
                                          vmeta["repair_attempted"], vmeta["repair_succeeded"])
            self.messages.append({"role": "assistant", "content": json.dumps(data)})
            self._trim()
            self.tracer.end_turn(data)
            return data

        data = format_error(
            "step_limit_reached",
            answer="I reached the step limit before finishing — try asking a narrower question.",
        )
        # Keep every trace line the same shape: this turn gets a validation event too.
        self.tracer.record_validation(parse_ok=False, errors=["step_limit_reached"],
                                      repair_attempted=False, repair_succeeded=False)
        self.messages.append({"role": "assistant", "content": json.dumps(data)})
        self._trim()
        self.tracer.end_turn(data)
        return data

    def chat(self, user_message: str) -> dict:
        return self._run_turn(user_message, stream=False)

    def stream(self, user_message: str) -> dict:
        return self._run_turn(user_message, stream=True)


# ── A 15-line trace analysis — the payoff ─────────────────────────────────────
def analyze_traces(path: Path = TRACE_PATH) -> None:
    """The trace answers 'why did the agent do that?' without rerunning it."""
    if not Path(path).exists() or not Path(path).read_text().strip():
        print(f"No traces at {path} yet — run the demo first.")
        return
    turns = [json.loads(line) for line in Path(path).read_text().splitlines()]
    slowest = max(turns, key=lambda t: t["total_latency_ms"] or 0)
    tool_counts, repairs = {}, 0
    for t in turns:
        for e in t["steps"]:
            if e["type"] == "tool_call":
                tool_counts[e["name"]] = tool_counts.get(e["name"], 0) + 1
            if e["type"] == "validation" and e["repair_attempted"]:
                repairs += 1
    print(f"turns:       {len(turns)}")
    print(f"slowest:     turn {slowest['turn_id']} ({slowest['total_latency_ms']} ms) — {slowest['user_message']!r}")
    print(f"tool calls:  {tool_counts}")
    print(f"repair rate: {repairs}/{len(turns)}")


# Demo run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start each demo run with a fresh trace file so the analysis reads one session.
    if TRACE_PATH.exists():
        TRACE_PATH.unlink()

    print("\n── Workshop 10: every turn leaves a trace ──")
    session = ChatSession(verbose=True)
    for user_message in [
        "When does the USC AI Club meet?",
        "How many days until that?",
        "Which is sooner, that meeting or the AI/ML office hours?",
        "What is the campus wifi password?",
    ]:
        print(f"\nYou: {user_message}")
        result = session.chat(user_message)
        print(f"Answer: {result['answer']}")

    print(f"\n── The trace file ({TRACE_PATH}) — one line per turn ──")
    first_line = TRACE_PATH.read_text().splitlines()[0]
    print(json.dumps(json.loads(first_line), indent=2)[:1500] + "\n  ...")

    print("\n── Trace analysis ──")
    analyze_traces()
