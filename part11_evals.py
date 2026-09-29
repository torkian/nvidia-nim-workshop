"""
Workshop 11 — Evals: Test Your Agent Like Software
Self-contained around one new idea: a hand-rolled regression eval harness. A
golden set of cases, each pairing an input with STRUCTURAL assertions on the
Workshop 9 six-key JSON contract, run through the real ChatSession, so a
regression fails red on your machine before a student ever hits it.

Workshop 9 built the assertable surface (the validated {status, answer, category,
items, missing, sources} dict). Workshop 10 recorded the evidence (one JSONL
trace line per turn). Workshop 11 turns both into a gate.

The whole lesson in one line: your agent is now importable, testable software.
This file *imports* Workshop 10 — `from part10_traces import ChatSession, ...` —
instead of redefining it. (The Colab notebook can't import a sibling file, so it
carries the same ChatSession inline, as every notebook in this series does.)

What makes an eval good, and what breaks it:
  - Assert on DECISIONS, not diction. status/category/missing/items shape come
    from constrained enums and structure — they're stable across sampled runs.
    Prose changes every run, so the only text check allowed is a loose,
    case-insensitive substring. Asserting on exact prose is THE pitfall: it fails
    tomorrow on an equally-correct paraphrase, and an ignored red suite is worse
    than no suite at all.
  - Never hardcode clock-dependent truth. "5 days until Thursday" is wrong
    tomorrow. Compute expected day counts in the harness, the same way the tool
    does.
  - Nemotron is non-deterministic (and temp 0 on a hosted endpoint isn't a fix —
    batching makes it non-deterministic anyway). So each case runs once and
    retries once on failure: first-try pass = PASS, pass-on-retry = FLAKY
    (a prompt bug to investigate, not an assertion to loosen), fail twice = FAIL.

Why not pytest / DeepEval / NeMo Evaluator? See the sidebar at the bottom.
"""

import sys
import re
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# The payoff of Workshops 1-10: the agent is a library now. Importing it also
# embeds the knowledge base once (you'll see the "Embedding..." line from WS10).
from part10_traces import (
    ChatSession, validate_answer, STATUSES, CATEGORIES, WEEKDAYS, LOCAL_TZ,
)

EVAL_TRACE = Path(__file__).resolve().parent / "traces" / "evals.jsonl"
# ^ a SEPARATE trace file from the chat one, so eval evidence stays clean and the
#   tool-path assertions have a predictable per-run file to read back.


# ── Ground truth that depends on the clock is COMPUTED, never hardcoded ────────
def days_until(weekday: str) -> int:
    """Same arithmetic the days_until_weekday tool uses — so the eval's expected
    value tracks the calendar instead of rotting overnight."""
    today = datetime.now(ZoneInfo(LOCAL_TZ))
    return (WEEKDAYS.index(weekday) - today.weekday()) % 7


# ── Reading the tool path back from this case's trace ─────────────────────────
def case_steps(path: Path) -> list:
    """All step events across this case's turns. Each attempt clears the file
    first, so everything in it belongs to the current run — safe to aggregate,
    and it catches tools called in earlier turns of a multi-turn case."""
    steps = []
    for line in (Path(path).read_text().splitlines() if Path(path).exists() else []):
        steps.extend(json.loads(line).get("steps", []))
    return steps


def _tool_calls(steps: list) -> list:
    """[(name, arguments_dict), ...] pulled from the trace's tool_call steps."""
    return [(s.get("name"), s.get("arguments", {}))
            for s in steps if s.get("type") == "tool_call"]


# ── The assertion vocabulary: expect-key -> checker(final, steps, expected) ────
# Each checker returns a failure message, or None if it passes.
def _lc(x) -> str:
    return str(x).lower() if x is not None else ""


def _lst(f, key) -> list:
    """A field that must be a list, coerced safely so a malformed type can't crash a checker."""
    v = f.get(key)
    return v if isinstance(v, list) else []


def _int(x):
    """First integer in a value, so a numeric assertion survives 2, '2', or '2 days'."""
    m = re.search(r"-?\d+", str(x))
    return int(m.group()) if m else None


CHECKERS = {
    "status": lambda f, s, e: None if f.get("status") == e
        else f"status: expected {e!r}, got {f.get('status')!r}",
    "category": lambda f, s, e: None if f.get("category") == e
        else f"category: expected {e!r}, got {f.get('category')!r}",
    "category_in": lambda f, s, e: None if f.get("category") in e
        else f"category: expected one of {e}, got {f.get('category')!r}",
    "min_items": lambda f, s, e: None if len(_lst(f, "items")) >= e
        else f"min_items: expected >= {e}, got {len(_lst(f, 'items'))}",
    "items_empty": lambda f, s, e: None if (len(_lst(f, "items")) == 0) == e
        else f"items_empty: expected {e}, got {len(_lst(f, 'items'))} item(s)",
    "missing_empty": lambda f, s, e: None if (len(_lst(f, "missing")) == 0) == e
        else f"missing_empty: expected {e}, missing={f.get('missing')}",
    "missing_nonempty": lambda f, s, e: None if (len(_lst(f, "missing")) > 0) == e
        else f"missing_nonempty: expected {e}, missing={f.get('missing')}",
    "sources_nonempty": lambda f, s, e: None if (len(_lst(f, "sources")) > 0) == e
        else f"sources_nonempty: expected {e}, got {len(_lst(f, 'sources'))} source(s)",
    "answer_contains_any": lambda f, s, e: None
        if any(_lc(sub) in _lc(f.get("answer")) for sub in e)
        else f"answer_contains_any: none of {e} in answer {_lc(f.get('answer'))[:80]!r}",
    "answer_not_contains": lambda f, s, e: None
        if not any(_lc(sub) in _lc(f.get("answer")) for sub in e)
        else f"answer_not_contains: found one of {e} in answer",
    "item_has_keys": lambda f, s, e: None
        if any(all(k in it for k in e) for it in _lst(f, "items") if isinstance(it, dict))
        else f"item_has_keys: no items entry has all of {e}",
    # Assert a STRUCTURED value, not prose: for each field, every expected value must
    # appear as that field on some item — numeric-tolerant (2 == '2' == '2 days').
    "item_field_equals": lambda f, s, e: None
        if all(any(isinstance(it, dict) and _int(it.get(field)) is not None
                   and _int(it.get(field)) == _int(val)
                   for it in _lst(f, "items"))
               for field, vals in e.items()
               for val in (vals if isinstance(vals, list) else [vals]))
        else f"item_field_equals: no item satisfies {e}; items={_lst(f, 'items')}",
    "tools_called": lambda f, s, e: None
        if set(e) <= {name for name, _ in _tool_calls(s)}
        else f"tools_called: expected {e}, saw {sorted({n for n,_ in _tool_calls(s)})}",
    "tool_args_contain": lambda f, s, e: None
        if all(any(name == tool and _lc(sub) in _lc(json.dumps(args))
                   for name, args in _tool_calls(s))
               for tool, sub in e.items())
        else f"tool_args_contain: expected {e} in the tool calls",
}


def check_case(final, steps: list, expect: dict) -> list:
    """Run every checker named in `expect`; collect ALL failures (no short-circuit)."""
    unknown = set(expect) - set(CHECKERS)
    if unknown:                       # a typo'd assertion must never silently pass — even on a crashed turn
        raise ValueError(f"unknown expect key(s): {sorted(unknown)}")
    if not isinstance(final, dict):        # a crashed/empty turn is a failure, not a traceback
        return [f"final is not a dict (got {type(final).__name__})"]
    failures = []
    schema_errors = validate_answer(final)   # always implied, free, deterministic
    if schema_errors:
        failures.append(f"schema invalid: {schema_errors}")
    for key, expected in expect.items():
        msg = CHECKERS[key](final, steps, expected)
        if msg:
            failures.append(msg)
    return failures


# ── The golden set — every case grounded in the real USC knowledge base ───────
# Built at RUN time (a function, not a module constant) so clock-dependent
# expectations can't go stale between import and execution.
def build_cases():
    return [
    {
        "name": "ai_club_answered", "regression": False,
        "turns": ["When does the USC AI Club meet?"],
        "expect": {
            "status": "answered", "category": "campus_event",
            "min_items": 1, "missing_empty": True, "sources_nonempty": True,
            "answer_contains_any": ["thursday"],
            "tools_called": ["search_campus_info"],
        },
    },
    {
        "name": "gpu_lab_hours_answered", "regression": False,
        "turns": ["What are the USC GPU lab hours?"],
        "expect": {
            "status": "answered", "category": "campus_hours",
            "min_items": 1, "sources_nonempty": True,
            "answer_contains_any": ["10", "6", "monday", "friday"],
        },
    },
    {
        # REGRESSION: this exact question broke during the Nemotron migration
        # (reasoning-on ate the token budget -> empty content). Pin the refusal
        # AND guard against a future prompt tweak inventing a password.
        "name": "wifi_stays_refusal", "regression": True,
        "turns": ["What is the campus wifi password?"],
        "expect": {
            "status": "not_found", "category": "refusal",
            "items_empty": True, "missing_nonempty": True,
            "answer_contains_any": ["check with the usc ai club"],
            "answer_not_contains": ["the password is", "password:"],
        },
    },
    {
        # Clock-dependent: assert the DECISION structurally — both events' day counts,
        # computed at run time (never hardcoded, never a prose substring). The stated
        # "sooner" winner lives only in prose; verifying the two numbers the conclusion
        # rests on is the decision-level check the JSON contract can actually support.
        "name": "sooner_comparison", "regression": False,
        "turns": ["Which is sooner, the AI Club meeting or the AI/ML office hours?"],
        "expect": {
            "status": "answered", "category": "comparison",
            "min_items": 2,
            "tools_called": ["days_until_weekday"],
            "item_field_equals": {"days_until": [days_until("Thursday"), days_until("Tuesday")]},
        },
    },
    {
        # Multi-turn: memory is ChatSession's signature capability; nothing else
        # tests it. "that" must resolve to Thursday THROUGH the conversation.
        "name": "memory_days_until", "regression": False,
        "turns": ["When does the USC AI Club meet?", "How many days until that?"],
        "expect": {
            "status": "answered",
            "category_in": ["campus_event", "campus_hours", "comparison"],
            "tool_args_contain": {"days_until_weekday": "thursday"},
            "item_field_equals": {"days_until": days_until("Thursday")},
        },
    },
    ]


# ── The runner: run once, retry once on failure -> PASS / FLAKY / FAIL ─────────
def run_case(case: dict):
    def attempt():
        if EVAL_TRACE.exists():
            EVAL_TRACE.unlink()          # clean slate: reader sees only this run's turns
        session = ChatSession(verbose=False, trace_path=EVAL_TRACE)  # fresh: no memory bleed
        final = None
        try:
            for turn in case["turns"]:
                final = session.chat(turn)
        except Exception as exc:          # a network/API blip shouldn't abort the whole suite
            return [f"agent raised: {type(exc).__name__}: {exc}"]
        return check_case(final, case_steps(EVAL_TRACE), case["expect"])

    failures = attempt()
    if not failures:
        return "PASS", []
    retry = attempt()
    if not retry:
        return "FLAKY", failures        # passed on retry — a prompt bug to investigate
    return "FAIL", retry


def main():
    if EVAL_TRACE.exists():
        EVAL_TRACE.unlink()             # fresh evidence for this run

    print("\n── Workshop 11: eval suite ──  (temp=0.2, run-once + retry-on-fail)")
    passed = flaky = failed = 0
    t_all = time.perf_counter()
    for case in build_cases():
        t0 = time.perf_counter()
        verdict, failures = run_case(case)
        secs = time.perf_counter() - t0
        tag = "REGRESSION " if case.get("regression") else ""
        print(f"{verdict:5s}  {tag}{case['name']:28s} {secs:5.1f}s")
        if verdict == "PASS":
            passed += 1
        elif verdict == "FLAKY":
            flaky += 1
            print(f"         (passed only on retry — investigate the prompt)")
            for msg in failures:
                print(f"         first-try miss: {msg}")
        else:
            failed += 1
            for msg in failures:
                print(f"         - {msg}")

    print(f"\neval summary: {passed} passed, {flaky} flaky, {failed} failed "
          f"— {time.perf_counter() - t_all:.0f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()


# ── Sidebar: why not pytest / DeepEval / NeMo Evaluator? ──────────────────────
# pytest — the cases port to @pytest.mark.parametrize almost mechanically (a good
#   exercise), but its binary pass/fail has no room for PASS/FLAKY/FAIL, and its
#   discovery model fights the data-driven `for case in CASES` loop you want in
#   cron/CI. Hand-rolling the ~60-line runner shows there's no magic: an eval is
#   run-agent + assert-on-dict + count.
# DeepEval / Ragas / promptfoo — add LLM-as-judge metrics. Powerful, but a judge
#   is a second non-deterministic model grading the first; the six-key contract
#   lets us make deterministic assertions, so we don't pay that cost yet.
# NVIDIA NeMo Evaluator — the production graduation path. It runs custom datasets
#   and benchmarks at scale against NIM endpoints, with judge models and results
#   tracked over time. This harness is a working miniature of exactly that
#   pipeline (cases + assertions + a scored, exit-coded run) — so when you need
#   thousands of cases, judge models, or CI dashboards, you swap the runner for
#   NeMo Evaluator, keep the golden cases, and stay inside the NVIDIA ecosystem.
