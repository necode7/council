"""
LLM Council — five advisor models debate, peer-review each other anonymously,
then a chairman model synthesizes a final verdict.

Edit the three values below, then run: python council.py
"""

import asyncio
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import httpx
import markdown2

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

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 120.0

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
    try:
        resp = await client.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT
        )
        if resp.status_code != 200:
            return f"[ERROR from {label} ({model}) — HTTP {resp.status_code}: {resp.text[:300]}]"
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return f"[ERROR from {label} ({model}) — empty choices: {str(data)[:300]}]"
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            return f"[ERROR from {label} ({model}) — empty content]"
        return content.strip()
    except httpx.TimeoutException:
        return f"[ERROR from {label} ({model}) — timed out after {TIMEOUT}s]"
    except Exception as e:
        return f"[ERROR from {label} ({model}) — {type(e).__name__}: {e}]"


# =============================================================================
# COUNCIL STAGES
# =============================================================================


async def run_advisors(
    client: httpx.AsyncClient,
    api_key: str,
    advisor_models: dict,
    question: str,
) -> dict:
    """advisor_models: {advisor_name: model_id}.  Returns {advisor_name: response}."""
    tasks = []
    for name in ADVISOR_NAMES:
        model = advisor_models[name]
        system = PERSONAS[name] + (
            "\n\nGive a focused response of 2-4 short paragraphs. No filler, no "
            "throat-clearing. Speak directly to the user about their decision."
        )
        tasks.append(
            call_openrouter(
                client, api_key, model, system, question,
                temperature=0.7, max_tokens=600, label=name,
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
) -> tuple[dict, dict]:
    """Each advisor reviews all 5 anonymized responses. Returns (reviews, letter_map)."""
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
            call_openrouter(
                client, api_key, model, system, review_prompt,
                temperature=0.7, max_tokens=400, label=f"{name} (review)",
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
        temperature=0.5, max_tokens=1500, label="Chairman",
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
"""


def md_to_html(md: str) -> str:
    return markdown2.markdown(md, extras=["fenced-code-blocks", "break-on-newline"])


def topic_to_title(topic: str) -> str:
    return topic.replace("_", " ").replace("-", " ").strip()


def build_pdf_html(
    mode: str,
    chairman_model: str,
    advisor_models: dict,
    question: str,
    advisor_responses: dict,
    reviews: dict,
    chairman_verdict: str,
    timestamp: datetime,
) -> str:
    title = topic_to_title(topic)
    date_str = timestamp.strftime("%B %d, %Y · %H:%M")

    chairman_html = md_to_html(chairman_verdict)
    chairman_html = chairman_html.replace("<h2>", '<h2 class="verdict-h2">')

    advisor_cards = []
    for name in ADVISOR_NAMES:
        body_html = md_to_html(advisor_responses[name])
        advisor_cards.append(f"""
        <div class="advisor-card">
            <div class="advisor-name">{name}</div>
            <div class="advisor-model">{advisor_models[name]}</div>
            <div class="advisor-body">{body_html}</div>
        </div>""")

    review_cards = []
    for name in ADVISOR_NAMES:
        body_html = md_to_html(reviews[name])
        review_cards.append(f"""
        <div class="advisor-card">
            <div class="advisor-name">{name}</div>
            <div class="advisor-model">peer review · {advisor_models[name]}</div>
            <div class="advisor-body">{body_html}</div>
        </div>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{PDF_CSS}</style></head>
<body>

<h1 class="cover-title">{title}</h1>
<div class="cover-meta">{date_str}  ·  Mode: <strong>{mode.upper()}</strong>  ·  Chairman: {chairman_model}</div>

<div class="section-label">The Question</div>
<div class="question-box">{question}</div>

<div class="section-label">Chairman's Verdict</div>
{chairman_html}

<hr class="section-divider"/>

<div class="page-break"></div>
<div class="section-label">Advisor Responses</div>
{''.join(advisor_cards)}

<hr class="section-divider"/>

<div class="page-break"></div>
<div class="section-label">Peer Reviews</div>
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
    filename = f"{topic}_{MODE}_{timestamp.strftime('%Y%m%d_%H%M')}.pdf"
    pdf_path = Path(__file__).parent / filename
    html = build_pdf_html(
        MODE, chairman_model, advisor_models, question,
        advisor_responses, reviews, chairman_verdict, timestamp,
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


if __name__ == "__main__":
    asyncio.run(main())
