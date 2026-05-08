"""
LLM Council — five advisor models debate, peer-review each other anonymously,
then a chairman model synthesizes a final verdict.

Edit the three values below, then run: python council.py
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import httpx
import markdown2

import history


# Raised when the user clicks Stop / cancels mid-run. Defined here (not in
# app.py) so council-level helpers — particularly interrupt waiters — can
# propagate cancellation cleanly.
class Cancelled(Exception):
    pass

# =============================================================================
# USER EDITS ONLY THESE THREE
# =============================================================================

MODE = "scout"  # "scout" (quick) | "verdict" (real decisions) | "deep" (career-level)

topic = "CA_Finals_vs_RL_Project"  # short, filename-safe, no spaces

question = """Council this: should I focus next 2 months on CA finals prep or split time with Mario Kart RL project?

Context:
- CA finals are critical for career trajectory
- RL project has firm team commitments and deadlines
- Both interest me long-term (finance + ML)
- ACCA exams (PM, AAA) also coming up
- Time is the real constraint

What's the smartest prioritization?"""

# =============================================================================
# CONFIG
# =============================================================================

MODEL_SETS = {
    "scout": {
        "chairman": "anthropic/claude-haiku-4.5",
        "advisors": [
            "google/gemini-3-flash-preview",
            "openai/gpt-5-mini",
            "x-ai/grok-4.1-fast",
            "deepseek/deepseek-v4-flash",
            "anthropic/claude-haiku-4.5",
        ],
    },
    "verdict": {
        "chairman": "anthropic/claude-sonnet-4.6",
        "advisors": [
            "openai/gpt-5.4",
            "google/gemini-3-pro-preview",
            "x-ai/grok-4.3",
            "deepseek/deepseek-v4-flash",
            "anthropic/claude-sonnet-4.6",
        ],
    },
    "deep": {
        "chairman": "anthropic/claude-opus-4.7",
        "advisors": [
            "openai/gpt-5.5",
            "google/gemini-3.1-pro-preview",
            "x-ai/grok-4.3",
            "deepseek/deepseek-v4-pro",
            "anthropic/claude-opus-4.7",
        ],
    },
}

PERSONAS = {
    "Contrarian": (
        "You are the Contrarian advisor. Actively look for what's wrong, missing, "
        "or will fail. Assume the idea has a fatal flaw and find it. Stress-test "
        "by asking the questions the user is avoiding. Be direct. Don't soften."
    ),
    "First Principles Thinker": (
        "You are the First Principles Thinker. Ignore the surface question. Ask "
        "what they're actually trying to solve. Strip assumptions. Rebuild the "
        "problem from the ground up. If they're asking the wrong question, say so."
    ),
    "Expansionist": (
        "You are the Expansionist advisor. Hunt for upside everyone else is missing. "
        "What could be bigger? What adjacent opportunity is hiding? Care about what "
        "happens if this works better than expected."
    ),
    "Outsider": (
        "You are the Outsider advisor. You have zero context about this person, "
        "their field, or history. Respond purely to what's in front of you. Catch "
        "the curse of knowledge — things obvious to insiders but invisible to outsiders."
    ),
    "Executor": (
        "You are the Executor advisor. Only one thing matters: can this be done, "
        "and what's the fastest path? Ignore theory and big-picture thinking. Look "
        "at every idea through 'OK, but what do you do Monday morning?'"
    ),
}

ADVISOR_NAMES = list(PERSONAS.keys())

# Short role labels for UI subtitles (interrupt cards).
PERSONA_ROLES = {
    "Contrarian": "stress-tests for failure modes",
    "First Principles Thinker": "rebuilds the problem from scratch",
    "Expansionist": "hunts for hidden upside",
    "Outsider": "catches the curse of knowledge",
    "Executor": "asks 'what do you do Monday morning?'",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 120.0

# =============================================================================
# INTERRUPT PROTOCOL (Feature B)
# =============================================================================

INTERRUPT_INSTRUCTION = """

If — and only if — you genuinely need a clarification before you can give useful advice, output ONLY this JSON object as your entire response (nothing before, nothing after):
{"interrupt": true, "question": "<your question to the user>", "options": ["<option 1>", "<option 2>", "<option 3>"], "allow_freetext": true}

Use this only when the answer would materially change your advice. Most questions don't need it. If you can give competent advice with what you already have, do that instead. Do not interrupt for trivial reasons. You get only one interrupt per response."""


def parse_interrupt(response: str) -> dict | None:
    """Detect and parse an interrupt JSON block. Returns a normalized dict
    {question, options, allow_freetext} or None if not an interrupt."""
    if not response:
        return None
    text = response.strip()
    # Strip optional code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{") or '"interrupt"' not in text:
        return None
    # Walk to the matching closing brace (handles trailing junk).
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        obj = json.loads(text[:end])
    except Exception:
        return None
    if not isinstance(obj, dict) or not obj.get("interrupt"):
        return None
    q = obj.get("question")
    if not isinstance(q, str) or not q.strip():
        return None
    options = obj.get("options")
    if not isinstance(options, list):
        options = []
    options = [str(o).strip() for o in options if str(o).strip()][:6]
    return {
        "question": q.strip(),
        "options": options,
        "allow_freetext": bool(obj.get("allow_freetext", True)),
    }


# =============================================================================
# CLARIFICATION GENERATOR (Feature A)
# =============================================================================

CLARIFICATION_SYSTEM = (
    "You generate clarifying questions to ask before a 5-person advisory "
    "council deliberates on a user's decision. You output strict JSON only — "
    "no preamble, no explanation, no code fences."
)

CLARIFICATION_USER_TMPL = """Read the user's question below and identify 3 to 5 clarifying questions whose answers would materially change the advice the council can give. Probe stakes, hard constraints, the alternatives the user hasn't shared, or hidden assumptions. Don't ask trivial demographic or restating questions.

Output ONLY a JSON array of 3-5 objects in this exact format:
[
  {{"question": "...", "options": ["...", "..."], "allow_freetext": true}},
  ...
]

Rules:
- "options" is 2-4 short, distinct possible answers (or [] if free-form is the only sensible response)
- "allow_freetext" is true if a custom answer is likely; false if the options cover the space
- Each question must be specific to this user's question — not generic
- 3 to 5 items, no more, no less

User's question:
\"\"\"
{question}
\"\"\"

JSON:"""


def _parse_clarification_questions(raw: str) -> list[dict]:
    if not raw:
        return []
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    # Find first '[' through matching ']'
    start = text.find("[")
    if start < 0:
        return []
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return []
    try:
        arr = json.loads(text[start:end])
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    out: list[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        opts = item.get("options")
        if not isinstance(opts, list):
            opts = []
        opts = [str(o).strip() for o in opts if str(o).strip()][:6]
        out.append({
            "question": q.strip(),
            "options": opts,
            "allow_freetext": bool(item.get("allow_freetext", True)),
        })
    return out[:5]


async def _generate_clarification_questions_async(
    api_key: str, question: str
) -> list[dict]:
    async with httpx.AsyncClient() as client:
        raw = await call_openrouter(
            client, api_key,
            model="anthropic/claude-haiku-4.5",
            system=CLARIFICATION_SYSTEM,
            user=CLARIFICATION_USER_TMPL.format(question=question[:3000]),
            temperature=0.3,
            max_tokens=1500,
            label="clarify-gen",
        )
    return _parse_clarification_questions(raw)


def generate_clarification_questions(api_key: str, question: str) -> list[dict]:
    """Sync wrapper for use from Streamlit's main thread."""
    return asyncio.run(_generate_clarification_questions_async(api_key, question))


def inject_clarifications(question: str, qa_pairs: list[tuple[str, str]]) -> str:
    """Build the augmented prompt sent into the pipeline. The original
    question stays first; clarification Q&A appears as labelled context."""
    if not qa_pairs:
        return question
    lines = ["", "--- USER CLARIFICATIONS (asked before deliberation) ---"]
    for i, (q, a) in enumerate(qa_pairs, 1):
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a}")
    lines.append("--- END CLARIFICATIONS ---")
    return question + "\n".join(["\n"] + lines)

# =============================================================================
# OPENROUTER CALL
# =============================================================================


async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    label: str,
    max_retries: int = 2,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/llm-council",
        "X-Title": "LLM Council",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = f"[ERROR from {label} ({model}) — exhausted retries]"
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT
            )
            if resp.status_code != 200:
                last_error = f"[ERROR from {label} ({model}) — HTTP {resp.status_code}: {resp.text[:300]}]"
                if resp.status_code in (408, 425, 429, 500, 502, 503, 504) and attempt < max_retries:
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                return last_error
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                last_error = f"[ERROR from {label} ({model}) — empty choices: {str(data)[:300]}]"
                if attempt < max_retries:
                    await asyncio.sleep(0.5)
                    continue
                return last_error
            content = choices[0].get("message", {}).get("content", "")
            if not content or not content.strip():
                last_error = f"[ERROR from {label} ({model}) — empty content]"
                if attempt < max_retries:
                    # Bump max_tokens on retry — empty content often means
                    # reasoning models burned all tokens internally.
                    payload["max_tokens"] = int(max_tokens * 1.5)
                    await asyncio.sleep(0.5)
                    continue
                return last_error
            return content.strip()
        except httpx.TimeoutException:
            last_error = f"[ERROR from {label} ({model}) — timed out after {TIMEOUT}s]"
            if attempt < max_retries:
                continue
            return last_error
        except Exception as e:
            last_error = f"[ERROR from {label} ({model}) — {type(e).__name__}: {e}]"
            return last_error
    return last_error


# =============================================================================
# AUTO TOPIC SLUG
# =============================================================================


async def topic_from_question(
    client: httpx.AsyncClient,
    api_key: str,
    question: str,
) -> str:
    """Cheap Haiku call → short underscore-joined label naming the decision."""
    system = "You produce short topic labels for a decision-making app. Output only the label, nothing else."
    user = f"""Read the question and output a short topic label (2-4 words) that names the decision being asked about.

Format rules:
- 2 to 4 words connected with underscores
- ASCII letters and digits only
- Title Case for normal words (Battery, Calculation, Strategy)
- ALL CAPS for acronyms and short technical abbreviations (ML, AI, CA, US, API, RL, BMS, LLM)
- lowercase for English connectors only: "of", "vs", "and", "the", "or"
- No quotes, no explanation, just the label

Examples:
Q: "Council this: should I take the Microsoft offer or stay at my startup?"
A: Microsoft_Offer_vs_Startup

Q: "Can ML help fix battery degradation calculation in real-time?"
A: ML_Battery_Degradation_Calculation

Q: "Should I focus on CA finals or my RL project?"
A: CA_Finals_vs_RL_Project

Q: "Should I marry Priya or wait another year?"
A: Marry_Priya_vs_Wait

Q: "Should I quit my job and start a startup?"
A: Quit_Job_for_Startup

Question:
{question[:2000]}

Label:"""
    raw = await call_openrouter(
        client, api_key,
        model="anthropic/claude-haiku-4.5",
        system=system,
        user=user,
        temperature=0.0,
        max_tokens=30,
        label="topic-slug",
    )
    if raw.startswith("[ERROR"):
        return "council_decision"
    slug = raw.strip().split("\n")[0].strip()
    slug = re.sub(r'^["\'`*_]+|["\'`*_.]+$', '', slug)
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', slug)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return (slug[:60] or "council_decision")


# =============================================================================
# COUNCIL STAGES
# =============================================================================


async def _call_with_interrupt(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    model: str,
    system: str,
    user: str,
    label: str,
    temperature: float,
    max_tokens: int,
    holder: dict | None,
    advisor_name: str,
    stage: str,
) -> str:
    """Make a call to the advisor/reviewer; if it returns an interrupt JSON
    block AND holder is provided, register the interrupt, wait for the
    user's answer, then re-run with the answer injected (max 1 interrupt
    per call). Other coroutines continue running while we await the user."""
    sys_for_call = system + (INTERRUPT_INSTRUCTION if holder is not None else "")
    response = await call_openrouter(
        client, api_key, model, sys_for_call, user,
        temperature=temperature, max_tokens=max_tokens, label=label,
    )
    if holder is None:
        return response

    interrupt = parse_interrupt(response)
    if not interrupt:
        return response

    answer_event = threading.Event()
    holder.setdefault("interrupts_pending", {})
    holder.setdefault("interrupts_order", [])
    holder.setdefault("interrupts_log", [])

    key = label  # unique: advisor names are unique, "Name (review)" too
    holder["interrupts_pending"][key] = {
        "key": key,
        "advisor_name": advisor_name,
        "stage": stage,
        "question": interrupt["question"],
        "options": interrupt["options"],
        "allow_freetext": interrupt["allow_freetext"],
        "answer": None,
        "event": answer_event,
    }
    holder["interrupts_order"].append(key)

    cancel_event = holder.get("cancel_event")
    while not answer_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        await asyncio.sleep(0.4)

    answer_text = (
        holder["interrupts_pending"][key].get("answer") or "(user skipped)"
    )
    holder["interrupts_log"].append({
        "advisor": advisor_name,
        "stage": stage,
        "question": interrupt["question"],
        "answer": answer_text,
    })
    holder["interrupts_pending"].pop(key, None)
    if key in holder["interrupts_order"]:
        holder["interrupts_order"].remove(key)

    # One-shot: do NOT pass INTERRUPT_INSTRUCTION on the resume call.
    augmented_user = (
        user.rstrip()
        + "\n\n--- YOUR EARLIER CLARIFICATION ---\n"
        + f"You asked: {interrupt['question']}\n"
        + f"User answered: {answer_text}\n"
        + "Now provide your full advice."
    )
    return await call_openrouter(
        client, api_key, model, system, augmented_user,
        temperature=temperature, max_tokens=max_tokens,
        label=f"{label} (post-interrupt)",
    )


async def run_advisors(
    client: httpx.AsyncClient,
    api_key: str,
    advisor_models: dict,
    question: str,
    holder: dict | None = None,
) -> dict:
    """advisor_models: {advisor_name: model_id}.  Returns {advisor_name: response}.
    If `holder` is provided, advisors may interrupt with a clarifying question."""
    tasks = []
    for name in ADVISOR_NAMES:
        model = advisor_models[name]
        system = PERSONAS[name] + (
            "\n\nGive a focused response of 2-4 short paragraphs. No filler, no "
            "throat-clearing. Speak directly to the user about their decision."
        )
        tasks.append(
            _call_with_interrupt(
                client, api_key,
                model=model, system=system, user=question, label=name,
                temperature=0.7, max_tokens=1200,
                holder=holder, advisor_name=name, stage="advisor",
            )
        )
    results = await asyncio.gather(*tasks)
    return dict(zip(ADVISOR_NAMES, results))


def anonymize(advisor_responses: dict) -> tuple[dict, dict]:
    """Shuffle advisor names to letters A-E. Returns (letter->response, letter->advisor_name)."""
    names = list(advisor_responses.keys())
    random.shuffle(names)
    letters = ["A", "B", "C", "D", "E"]
    letter_to_name = dict(zip(letters, names))
    letter_to_response = {L: advisor_responses[letter_to_name[L]] for L in letters}
    return letter_to_response, letter_to_name


def build_anonymized_block(letter_to_response: dict) -> str:
    parts = []
    for L in ["A", "B", "C", "D", "E"]:
        parts.append(f"=== Response {L} ===\n{letter_to_response[L]}")
    return "\n\n".join(parts)


async def run_peer_review(
    client: httpx.AsyncClient,
    api_key: str,
    advisor_models: dict,
    advisor_responses: dict,
    question: str,
    holder: dict | None = None,
) -> tuple[dict, dict]:
    """Each advisor reviews all 5 anonymized responses. Returns (reviews, letter_map).
    If `holder` is provided, reviewers may interrupt with a clarifying question."""
    letter_to_response, letter_to_name = anonymize(advisor_responses)
    anon_block = build_anonymized_block(letter_to_response)

    review_prompt = f"""The original question was:

{question}

Five advisors gave the responses below (anonymized as A-E). Read all five, then answer:

1. Which response is strongest, and why?
2. Which response has the biggest blind spot, and what is it?
3. What did all five of them miss?

Be concise. Aim for 3 short paragraphs total — one per question.

{anon_block}"""

    tasks = []
    for name in ADVISOR_NAMES:
        model = advisor_models[name]
        system = (
            "You are a member of an advisory council, peer-reviewing the work of "
            "four colleagues plus your own (you don't know which is which). Be "
            "honest, specific, and brief. Reference responses by their letter (A-E)."
        )
        tasks.append(
            _call_with_interrupt(
                client, api_key,
                model=model, system=system, user=review_prompt,
                label=f"{name} (review)",
                temperature=0.7, max_tokens=1000,
                holder=holder, advisor_name=name, stage="review",
            )
        )
    results = await asyncio.gather(*tasks)
    reviews = dict(zip(ADVISOR_NAMES, results))
    return reviews, letter_to_name


async def run_chairman(
    client: httpx.AsyncClient,
    api_key: str,
    chairman_model: str,
    question: str,
    advisor_responses: dict,
    reviews: dict,
) -> str:
    advisor_block = "\n\n".join(
        f"### {name}'s response\n{advisor_responses[name]}"
        for name in ADVISOR_NAMES
    )
    review_block = "\n\n".join(
        f"### {name}'s peer review\n{reviews[name]}"
        for name in ADVISOR_NAMES
    )

    system = (
        "You are the Chairman of an advisory council. Five advisors have given "
        "their views, and each has peer-reviewed the others anonymously. Your job "
        "is to synthesize a clear, decisive verdict for the user. Do not hedge. "
        "Do not say 'it depends'. Give a real recommendation."
    )

    user = f"""ORIGINAL QUESTION:
{question}

============================================================
ADVISOR RESPONSES:
{advisor_block}

============================================================
PEER REVIEWS:
{review_block}

============================================================
Now write the final synthesis. Use EXACTLY these section headers and order, in markdown:

## Where the Council Agrees
[High-confidence points multiple advisors converged on]

## Where the Council Clashes
[Genuine disagreements. Present both sides. Explain why reasonable advisors disagree.]

## Blind Spots the Council Caught
[Things that only emerged through peer review]

## The Recommendation
[A clear, direct recommendation. Not "it depends." A real answer with reasoning.]

## The One Thing to Do First
[A single concrete next step. Not a list. One thing.]
"""

    return await call_openrouter(
        client, api_key, chairman_model, system, user,
        temperature=0.5, max_tokens=2500, label="Chairman",
    )


# =============================================================================
# OUTPUT — TERMINAL
# =============================================================================

DIVIDER = "=" * 72


def print_header(title: str):
    print()
    print(DIVIDER)
    print(title)
    print(DIVIDER)


def print_terminal_report(
    mode: str,
    advisor_models: dict,
    chairman_model: str,
    question: str,
    advisor_responses: dict,
    reviews: dict,
    chairman_verdict: str,
):
    print_header("CHAIRMAN'S VERDICT")
    print(chairman_verdict)

    print_header("ADVISOR RESPONSES")
    for name in ADVISOR_NAMES:
        print()
        print(f"--- {name}  ({advisor_models[name]}) ---")
        print(advisor_responses[name])

    print_header("PEER REVIEWS")
    for name in ADVISOR_NAMES:
        print()
        print(f"--- {name} reviews the council ---")
        print(reviews[name])


# =============================================================================
# OUTPUT — PDF
# =============================================================================

PDF_CSS = """
@page {
    size: A4;
    margin: 2cm;
    @bottom-center {
        content: "LLM Council  ·  page " counter(page) " of " counter(pages);
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size: 9pt;
        color: #888;
    }
}
body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
}
h1, h2, h3, h4 {
    font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
    color: #111;
    line-height: 1.25;
}
h1.cover-title {
    font-size: 30pt;
    margin-top: 1.2cm;
    margin-bottom: 0.2cm;
    border-bottom: 3px solid #b8860b;
    padding-bottom: 0.3cm;
}
.cover-meta {
    font-family: 'Helvetica', sans-serif;
    font-size: 11pt;
    color: #555;
    margin-bottom: 1.2cm;
}
.section-label {
    font-family: 'Helvetica', sans-serif;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 9pt;
    color: #b8860b;
    margin-top: 1.2cm;
    margin-bottom: 0.2cm;
}
.question-box {
    background: #f7f4ec;
    border-left: 4px solid #b8860b;
    padding: 0.7cm 0.9cm;
    margin: 0.4cm 0 0.8cm 0;
    font-style: italic;
    white-space: pre-wrap;
}
h2.verdict-h2 {
    font-size: 16pt;
    margin-top: 0.9cm;
    margin-bottom: 0.2cm;
    color: #2a2a2a;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.15cm;
}
.advisor-card {
    margin: 0.6cm 0;
    page-break-inside: avoid;
}
.advisor-name {
    font-family: 'Helvetica Neue', 'Helvetica', sans-serif;
    font-size: 14pt;
    font-weight: 600;
    color: #b8860b;
    margin-bottom: 0;
}
.advisor-model {
    font-family: 'Helvetica', sans-serif;
    font-size: 9pt;
    color: #888;
    margin-top: 2px;
    margin-bottom: 0.3cm;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.advisor-body {
    margin-top: 0.2cm;
}
hr.section-divider {
    border: none;
    border-top: 2px solid #b8860b;
    margin: 1cm 0 0.6cm 0;
    width: 30%;
}
.page-break {
    page-break-before: always;
}
.footer {
    margin-top: 1.5cm;
    padding-top: 0.4cm;
    border-top: 1px solid #ccc;
    font-family: 'Helvetica', sans-serif;
    font-size: 9pt;
    color: #888;
    text-align: center;
}
p { margin: 0.25cm 0; }
strong { color: #111; }

/* PDF outline (the bookmark sidebar in PDF readers) */
h1.cover-title       { bookmark-level: 1; bookmark-label: content(); }
.section-label       { bookmark-level: 2; bookmark-label: content(); }
.advisor-name        { bookmark-level: 3; bookmark-label: content(); }

/* Clickable Table of Contents */
.toc-block {
    background: #fbfaf5;
    border: 1px solid #e8e1cc;
    border-radius: 4px;
    padding: 0.5cm 0.8cm 0.6cm 0.8cm;
    margin: 0.4cm 0 1.2cm 0;
}
.toc-title {
    font-family: 'Helvetica', sans-serif;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 9pt;
    color: #b8860b;
    margin-bottom: 0.3cm;
}
.toc, .toc ul {
    list-style: none;
    padding-left: 0;
    margin: 0;
}
.toc li {
    font-family: 'Helvetica', 'Arial', sans-serif;
    font-size: 10.5pt;
    margin: 0.12cm 0;
}
.toc ul {
    padding-left: 0.7cm;
    margin-top: 0.1cm;
}
.toc ul li {
    font-size: 9.5pt;
    color: #555;
}
.toc a {
    color: #1a1a1a;
    text-decoration: none;
    border-bottom: 1px dotted #c7bd9c;
}
.toc a:hover { color: #b8860b; }

/* Clarifications & interrupts (Feature A & B) */
.qa-block {
    margin: 0.4cm 0 0.8cm 0;
    background: #fbfaf5;
    border: 1px solid #e8e1cc;
    border-radius: 4px;
    padding: 0.5cm 0.7cm;
}
.qa-block-title {
    font-family: 'Helvetica', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 8.5pt;
    color: #b8860b;
    margin-bottom: 0.25cm;
}
.qa-pair {
    margin: 0.25cm 0;
    padding: 0.15cm 0;
    border-bottom: 1px dotted #e8e1cc;
}
.qa-pair:last-child { border-bottom: none; }
.qa-meta {
    font-family: 'Helvetica', sans-serif;
    font-size: 8.5pt;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.1cm;
}
.qa-q {
    font-weight: 600;
    color: #2a2a2a;
    margin: 0.1cm 0;
}
.qa-a {
    color: #444;
    margin: 0.1cm 0 0 0.4cm;
}
"""


def md_to_html(md: str) -> str:
    return markdown2.markdown(md, extras=["fenced-code-blocks", "break-on-newline"])


_LOWERCASE_CONNECTORS = {
    "of", "the", "and", "a", "an", "or", "to", "vs", "in", "on",
    "for", "by", "at",
}


def topic_to_title(topic: str) -> str:
    """Underscore-joined slug → display title.

    Haiku is asked to produce slugs with proper casing already (Title Case
    for words, ALL CAPS for acronyms, lowercase connectors). We trust any
    word that contains uppercase letters and only Title-case words that
    came back fully lowercase. Connector words stay lowercase if not first.
    """
    raw = topic.replace("_", " ").replace("-", " ").strip()
    if not raw:
        return ""
    out = []
    for i, w in enumerate(raw.split()):
        if any(c.isupper() for c in w):
            out.append(w)
        elif i > 0 and w in _LOWERCASE_CONNECTORS:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _slug(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9]+', '-', s).strip('-').lower()
    return s or "section"


def _build_clarifications_block(clarifications: list[dict]) -> str:
    """HTML for Feature A's pre-deliberation clarifications."""
    if not clarifications:
        return ""
    items = []
    for c in clarifications:
        q = (c.get("question") or "").strip()
        a = (c.get("answer") or "").strip()
        if not q:
            continue
        items.append(
            f"<div class=\"qa-pair\"><div class=\"qa-q\">Q. {q}</div>"
            f"<div class=\"qa-a\">A. {a}</div></div>"
        )
    if not items:
        return ""
    return (
        '<div class="qa-block"><div class="qa-block-title">'
        'Pre-deliberation clarifications</div>'
        + "".join(items) + "</div>"
    )


def _build_interrupts_block(interrupts: list[dict]) -> str:
    """HTML for Feature B's mid-deliberation advisor interrupts."""
    if not interrupts:
        return ""
    items = []
    for i in interrupts:
        advisor = (i.get("advisor") or "").strip() or "An advisor"
        stage = (i.get("stage") or "advisor").strip()
        q = (i.get("question") or "").strip()
        a = (i.get("answer") or "").strip()
        if not q:
            continue
        stage_label = "advisor" if stage == "advisor" else "peer reviewer"
        items.append(
            f"<div class=\"qa-pair\">"
            f"<div class=\"qa-meta\">{advisor} · {stage_label}</div>"
            f"<div class=\"qa-q\">Q. {q}</div>"
            f"<div class=\"qa-a\">A. {a}</div></div>"
        )
    if not items:
        return ""
    return (
        '<div class="qa-block"><div class="qa-block-title">'
        'Mid-deliberation clarifications</div>'
        + "".join(items) + "</div>"
    )


def build_pdf_html(
    mode: str,
    chairman_model: str,
    advisor_models: dict,
    question: str,
    advisor_responses: dict,
    reviews: dict,
    chairman_verdict: str,
    timestamp: datetime,
    topic: str,
    clarifications: list[dict] | None = None,
    interrupts: list[dict] | None = None,
) -> str:
    title = topic_to_title(topic)
    date_str = timestamp.strftime("%B %d, %Y · %H:%M")

    chairman_html = md_to_html(chairman_verdict)
    chairman_html = chairman_html.replace("<h2>", '<h2 class="verdict-h2">')

    advisor_cards = []
    for name in ADVISOR_NAMES:
        body_html = md_to_html(advisor_responses[name])
        advisor_cards.append(f"""
        <div class="advisor-card" id="advisor-{_slug(name)}">
            <div class="advisor-name">{name}</div>
            <div class="advisor-model">{advisor_models[name]}</div>
            <div class="advisor-body">{body_html}</div>
        </div>""")

    review_cards = []
    for name in ADVISOR_NAMES:
        body_html = md_to_html(reviews[name])
        review_cards.append(f"""
        <div class="advisor-card" id="review-{_slug(name)}">
            <div class="advisor-name">{name}</div>
            <div class="advisor-model">peer review · {advisor_models[name]}</div>
            <div class="advisor-body">{body_html}</div>
        </div>""")

    advisor_toc = "\n".join(
        f'      <li><a href="#advisor-{_slug(n)}">{n}</a></li>' for n in ADVISOR_NAMES
    )
    review_toc = "\n".join(
        f'      <li><a href="#review-{_slug(n)}">{n}</a></li>' for n in ADVISOR_NAMES
    )

    clarifications_html = _build_clarifications_block(clarifications or [])
    interrupts_html = _build_interrupts_block(interrupts or [])

    toc_block = f"""
<div class="toc-block">
  <div class="toc-title">Contents</div>
  <ul class="toc">
    <li><a href="#question">Question &amp; Context</a></li>
    <li><a href="#verdict">Chairman's Verdict</a></li>
    <li><a href="#advisors">Advisor Responses</a>
      <ul>
{advisor_toc}
      </ul>
    </li>
    <li><a href="#reviews">Peer Reviews</a>
      <ul>
{review_toc}
      </ul>
    </li>
  </ul>
</div>"""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{PDF_CSS}</style></head>
<body>

<h1 class="cover-title">{title}</h1>
<div class="cover-meta">{date_str}  ·  Mode: <strong>{mode.upper()}</strong>  ·  Chairman: {chairman_model}</div>

{toc_block}

<div class="page-break"></div>

<div class="section-label" id="question">Question &amp; Context</div>
<div class="question-box">{question}</div>
{clarifications_html}
{interrupts_html}

<div class="section-label" id="verdict">Chairman's Verdict</div>
{chairman_html}

<hr class="section-divider"/>

<div class="page-break"></div>
<div class="section-label" id="advisors">Advisor Responses</div>
{''.join(advisor_cards)}

<hr class="section-divider"/>

<div class="page-break"></div>
<div class="section-label" id="reviews">Peer Reviews</div>
{''.join(review_cards)}

<div class="footer">Generated {timestamp.strftime("%Y-%m-%d %H:%M:%S")} · LLM Council ({mode.upper()})</div>

</body></html>"""


def save_pdf(html: str, filename: str) -> str:
    """Try weasyprint; fallback to reportlab if it fails."""
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(filename)
        return "weasyprint"
    except Exception as e:
        print(f"[weasyprint failed: {e} — falling back to reportlab]")
        save_pdf_reportlab(html, filename)
        return "reportlab"


def save_pdf_reportlab(html: str, filename: str):
    """Crude fallback: strip tags and render plain text."""
    import re
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    for para in text.split("\n\n"):
        flow.append(Paragraph(para.replace("\n", "<br/>"), styles["Normal"]))
        flow.append(Spacer(1, 8))
    doc.build(flow)


# =============================================================================
# MAIN
# =============================================================================


async def main():
    api_key = os.environ.get("OPENROUTER_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_KEY environment variable.")
        sys.exit(1)

    if MODE not in MODEL_SETS:
        print(f"ERROR: MODE must be one of {list(MODEL_SETS.keys())}, got {MODE!r}")
        sys.exit(1)

    config = MODEL_SETS[MODE]
    chairman_model = config["chairman"]
    advisor_models = dict(zip(ADVISOR_NAMES, config["advisors"]))

    print()
    print(DIVIDER)
    print(f"  LLM COUNCIL  ·  MODE: {MODE.upper()}")
    print(DIVIDER)
    print(f"  Topic:    {topic}")
    print(f"  Chairman: {chairman_model}")
    print(f"  Advisors:")
    for name in ADVISOR_NAMES:
        print(f"    · {name:<26} {advisor_models[name]}")
    print(DIVIDER)

    started = datetime.now()

    async with httpx.AsyncClient() as client:
        print("\n→ Running 5 advisors in parallel...")
        advisor_responses = await run_advisors(client, api_key, advisor_models, question)
        print("✓ Advisors done")

        print("\n→ Running peer review (anonymized)...")
        reviews, letter_map = await run_peer_review(
            client, api_key, advisor_models, advisor_responses, question
        )
        print("✓ Peer review done")
        print(f"  (anonymization map this run: {letter_map})")

        print("\n→ Chairman synthesizing...")
        chairman_verdict = await run_chairman(
            client, api_key, chairman_model, question, advisor_responses, reviews
        )
        print("✓ VERDICT READY")

    elapsed = (datetime.now() - started).total_seconds()

    print_terminal_report(
        MODE, advisor_models, chairman_model, question,
        advisor_responses, reviews, chairman_verdict,
    )

    timestamp = datetime.now()

    try:
        new_id = history.save_council(
            {
                "question": question,
                "mode": MODE,
                "topic": topic,
                "chairman_model": chairman_model,
                "advisor_models": advisor_models,
                "advisor_responses": advisor_responses,
                "reviews": reviews,
                "letter_map": letter_map,
                "chairman_verdict": chairman_verdict,
            },
            timestamp,
        )
        print(f"\n💾 Saved to history as #{new_id}  (council_history.db)")
    except Exception as e:
        print(f"[history save failed: {e}]")

    filename = f"{topic}_{MODE}_{timestamp.strftime('%Y%m%d_%H%M')}.pdf"
    pdf_path = Path(__file__).parent / filename
    html = build_pdf_html(
        MODE, chairman_model, advisor_models, question,
        advisor_responses, reviews, chairman_verdict, timestamp, topic,
    )
    engine = save_pdf(html, str(pdf_path))

    print()
    print(DIVIDER)
    print(f"  Run took {elapsed:.1f}s  ·  PDF engine: {engine}")
    print(DIVIDER)
    print(f"\n✅ PDF SAVED: {filename}")
    print(f'📥 To download: Open the file panel on the left, find {filename}, '
          f'tap the three-dot menu, select "Download"')
    print(f"\n   Full path: {pdf_path}\n")


# =============================================================================
# CLI — HISTORY SUB-COMMANDS
# =============================================================================


def cli_history(limit: int = 10) -> None:
    rows = history.list_councils(limit=limit)
    if not rows:
        print("No councils saved yet. Run one to start your archive.")
        return
    print()
    print(DIVIDER)
    print(f"  COUNCIL HISTORY  ·  last {len(rows)}")
    print(DIVIDER)
    for r in rows:
        title = topic_to_title(r["topic_slug"])
        print(
            f"  [{r['id']:>3}]  {r['timestamp']}  "
            f"{r['mode'].upper():<7}  ₹{r['cost_estimate']:>5.1f}  ·  {title}"
        )
    print(DIVIDER)
    print(f"  Recall a run with:  python council.py --recall <id>")
    print()


def cli_recall(council_id: int) -> None:
    row = history.get_council(council_id)
    if not row:
        print(f"No council with id {council_id}.")
        sys.exit(1)
    print()
    print(DIVIDER)
    print(f"  COUNCIL #{row['id']}  ·  {row['timestamp']}  ·  {row['mode'].upper()}")
    print(DIVIDER)
    print(f"  Topic:    {row['topic_slug']}")
    print(f"  Chairman: {row['chairman_model']}")
    print(f"  Cost:     ₹{row['cost_estimate']:.1f}  (mode estimate)")
    print(DIVIDER)
    print("\nQUESTION:")
    print(row["question"])
    print_terminal_report(
        row["mode"],
        row["advisor_models"],
        row["chairman_model"],
        row["question"],
        row["advisor_responses"],
        row["reviews"],
        row["chairman_verdict"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Council — five advisors debate, chairman decides.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print the last 10 councils from local history and exit.",
    )
    parser.add_argument(
        "--recall",
        type=int,
        metavar="ID",
        help="Print a specific past council in full and exit.",
    )
    args = parser.parse_args()

    if args.history:
        cli_history()
    elif args.recall is not None:
        cli_recall(args.recall)
    else:
        asyncio.run(main())
