"""
LLM Council — Streamlit web app.

Usage:
    Local:  streamlit run app.py
    Cloud:  push to GitHub → deploy on share.streamlit.io with secrets:
              OPENROUTER_KEY      = "sk-or-v1-..."
              APP_PASSWORD        = "<your password>"
              TURSO_DATABASE_URL  = "libsql://..."   # optional, enables durable history
              TURSO_AUTH_TOKEN    = "..."            # optional
"""

import asyncio
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

import council
from council import Cancelled
import history


# =============================================================================
# SECRETS — hydrate into os.environ early so history.py picks up Turso creds
# regardless of where it's imported from (Streamlit context vs CLI).
# =============================================================================


def _early_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


for _k in ("TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN"):
    if not os.environ.get(_k):
        _v = _early_secret(_k)
        if _v:
            os.environ[_k] = _v


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="LLM Council",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# SECRETS / AUTH
# =============================================================================

def _secret(key: str, default: str = "") -> str:
    """st.secrets raises if no secrets.toml exists; this just returns ''."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def require_password() -> bool:
    if st.session_state.get("auth_ok"):
        return True

    expected = _secret("APP_PASSWORD")
    if not expected:
        st.session_state.auth_ok = True
        return True

    st.title("🏛️ LLM Council")
    st.caption("Private. Enter the password to continue.")
    pw = st.text_input("Password", type="password", label_visibility="collapsed")
    if st.button("Enter", use_container_width=True):
        if pw == expected:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


if not require_password():
    st.stop()


# =============================================================================
# COUNCIL PIPELINE (with progress + cancellation)
# =============================================================================


async def _run_pipeline(
    api_key: str,
    mode: str,
    original_question: str,
    augmented_question: str,
    clarifications: list[dict],
    holder: dict,
):
    config = council.MODEL_SETS[mode]
    chairman_model = config["chairman"]
    advisor_models = dict(zip(council.ADVISOR_NAMES, config["advisors"]))
    cancel_event: threading.Event = holder["cancel_event"]

    def check_cancel():
        if cancel_event.is_set():
            raise Cancelled()

    async with httpx.AsyncClient() as client:
        holder["stage"] = "Reading your question…"
        # Topic is derived from the ORIGINAL user question — we don't want
        # the clarification block bleeding into the slug.
        topic = await council.topic_from_question(client, api_key, original_question)
        holder["topic"] = topic
        check_cancel()

        holder["stage"] = "5 advisors deliberating in parallel…"
        advisors = await council.run_advisors(
            client, api_key, advisor_models, augmented_question, holder=holder
        )
        check_cancel()

        holder["stage"] = "Anonymous peer review…"
        reviews, letter_map = await council.run_peer_review(
            client, api_key, advisor_models, advisors, augmented_question,
            holder=holder,
        )
        check_cancel()

        holder["stage"] = "Chairman delivering verdict…"
        verdict = await council.run_chairman(
            client, api_key, chairman_model, augmented_question, advisors, reviews
        )

    return {
        "topic": topic,
        "mode": mode,
        "chairman_model": chairman_model,
        "advisor_models": advisor_models,
        "advisor_responses": advisors,
        "reviews": reviews,
        "letter_map": letter_map,
        "chairman_verdict": verdict,
        # Display/PDF/history use the ORIGINAL question; clarifications are
        # rendered as a separate, structured Q&A list.
        "question": original_question,
        "clarifications": clarifications,
        "interrupts": list(holder.get("interrupts_log", [])),
    }


def start_run(
    api_key: str,
    mode: str,
    original_question: str,
    clarifications: list[dict] | None = None,
):
    """Spawn a thread that runs the pipeline and updates session_state['holder'].
    `clarifications` is a list of {question, answer} from Feature A."""
    clarifications = clarifications or []
    qa_pairs = [(c["question"], c["answer"]) for c in clarifications]
    augmented_question = council.inject_clarifications(original_question, qa_pairs)

    holder = {
        "stage": "Starting…",
        "started_at": time.time(),
        "done": False,
        "cancelled": False,
        "result": None,
        "error": None,
        "cancel_event": threading.Event(),
        "topic": None,
        # Feature B — pending advisor/reviewer interrupts. The pipeline thread
        # writes here; the UI thread renders + answers.
        "interrupts_pending": {},
        "interrupts_order": [],
        "interrupts_log": [],
    }
    st.session_state["holder"] = holder

    def runner():
        try:
            result = asyncio.run(_run_pipeline(
                api_key, mode, original_question, augmented_question,
                clarifications, holder,
            ))
            holder["result"] = result
            try:
                holder["saved_id"] = history.save_council(result, datetime.now())
            except Exception as e:
                holder["history_save_error"] = f"{type(e).__name__}: {e}"
        except Cancelled:
            holder["cancelled"] = True
        except Exception as e:
            holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            holder["done"] = True
            holder["finished_at"] = time.time()

    t = threading.Thread(target=runner, daemon=True)
    add_script_run_ctx(t)
    t.start()


def build_pdf_bytes(result: dict, timestamp: datetime) -> bytes | None:
    html = council.build_pdf_html(
        result["mode"],
        result["chairman_model"],
        result["advisor_models"],
        result["question"],
        result["advisor_responses"],
        result["reviews"],
        result["chairman_verdict"],
        timestamp,
        result["topic"],
        clarifications=result.get("clarifications", []),
        interrupts=result.get("interrupts", []),
    )
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        council.save_pdf(html, tmp_path)
        return Path(tmp_path).read_bytes()
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return None


# =============================================================================
# INTERACTIVE QUESTION CARD (Feature A & B share this)
# =============================================================================


def _combine_answer(chosen: list[str], freetext: str) -> str:
    """Merge selected options + freetext into a single answer string."""
    parts = []
    if chosen:
        parts.append(" / ".join(chosen))
    if freetext.strip():
        parts.append(freetext.strip())
    return " — ".join(parts) if parts else ""


def _render_question_card(
    *,
    asker: str,
    asker_role: str | None,
    question_text: str,
    options: list[str],
    allow_freetext: bool,
    key_prefix: str,
    on_submit,           # callable(answer: str) -> None
    on_skip,             # callable() -> None
    show_skip_remaining: bool = False,
    on_skip_remaining=None,
):
    """Render a Claude-style question card (used by Feature A and Feature B)."""
    with st.container(border=True):
        st.markdown(f"**{asker} asks:**")
        if asker_role:
            st.caption(asker_role)
        st.markdown(f"### {question_text}")

        col1, col2 = st.columns(2)
        with col1:
            if options:
                chosen = st.multiselect(
                    "Pick one or more options",
                    options,
                    key=f"{key_prefix}_opts",
                )
            else:
                chosen = []
                st.caption("_No predefined options — please type your answer._")
        with col2:
            free = st.text_area(
                "Your own answer" if allow_freetext else "Add nuance (optional)",
                key=f"{key_prefix}_free",
                height=120,
            )

        bcols = st.columns([2, 1, 2] if show_skip_remaining else [2, 1])
        with bcols[0]:
            if st.button(
                "Submit answer",
                type="primary",
                key=f"{key_prefix}_submit",
                use_container_width=True,
            ):
                answer = _combine_answer(chosen, free)
                if not answer:
                    st.warning("Pick an option or write something before submitting.")
                else:
                    on_submit(answer)
        with bcols[1]:
            if st.button(
                "Skip",
                key=f"{key_prefix}_skip",
                use_container_width=True,
            ):
                on_skip()
        if show_skip_remaining and on_skip_remaining is not None:
            with bcols[2]:
                if st.button(
                    "Skip remaining questions",
                    key=f"{key_prefix}_skip_all",
                    use_container_width=True,
                ):
                    on_skip_remaining()


def _render_answer_card(idx: int, q_text: str, answer: str):
    """Small card showing a previously-answered question."""
    with st.container(border=True):
        st.caption(f"Q{idx + 1}: {q_text}")
        st.markdown(f"_{answer}_")


# =============================================================================
# UI — TITLE + TABS
# =============================================================================

st.title("🏛️ LLM Council")
st.markdown("`Metamorphosis • Phase 3 Architecture Active`")
st.divider()
st.caption(
    "Five advisors debate your decision on different models. "
    "They peer-review each other anonymously. A chairman delivers the verdict."
)

tab_council, tab_history = st.tabs(["Council", "History"])


# =============================================================================
# TAB — COUNCIL
# =============================================================================

with tab_council:
    holder = st.session_state.get("holder")
    running = bool(holder and not holder["done"])
    have_result = bool(holder and holder["done"] and holder["result"])
    clarify_state = st.session_state.get("clarify_state")
    in_clarify = clarify_state is not None and not running and not have_result

    # ---- CLARIFICATION PHASE (Feature A) ---------------------------------
    if in_clarify:
        # Step 1: generate questions if we don't have them yet.
        if clarify_state["phase"] == "generating":
            with st.spinner("The council is preparing some questions for you…"):
                try:
                    qs = council.generate_clarification_questions(
                        clarify_state["api_key"], clarify_state["original_question"]
                    )
                except Exception as e:
                    st.error(
                        f"Couldn't generate clarifying questions: {e}. "
                        f"Running without clarification."
                    )
                    qs = []
                clarify_state["questions"] = qs
                clarify_state["phase"] = "asking" if qs else "done"
            st.rerun()

        elif clarify_state["phase"] == "asking":
            questions = clarify_state["questions"]
            answers = clarify_state["answers"]
            idx = clarify_state["current_idx"]

            st.markdown("### 🤔 The council has a few questions first")
            st.caption(
                f"Question {min(idx + 1, len(questions))} of {len(questions)}"
            )

            # Trail of prior answers
            for i in range(len(answers)):
                _render_answer_card(i, questions[i]["question"], answers[i])

            current_q = questions[idx]

            def _submit(answer: str):
                clarify_state["answers"].append(answer)
                clarify_state["current_idx"] += 1
                if clarify_state["current_idx"] >= len(questions):
                    clarify_state["phase"] = "done"
                st.rerun()

            def _skip():
                clarify_state["answers"].append("(user skipped)")
                clarify_state["current_idx"] += 1
                if clarify_state["current_idx"] >= len(questions):
                    clarify_state["phase"] = "done"
                st.rerun()

            def _skip_all():
                clarify_state["phase"] = "done"
                st.rerun()

            _render_question_card(
                asker="The Council",
                asker_role=None,
                question_text=current_q["question"],
                options=current_q["options"],
                allow_freetext=current_q["allow_freetext"],
                key_prefix=f"clarify_{idx}",
                on_submit=_submit,
                on_skip=_skip,
                show_skip_remaining=True,
                on_skip_remaining=_skip_all,
            )

            if st.button("Cancel clarification", key="clarify_cancel"):
                del st.session_state["clarify_state"]
                st.rerun()

        elif clarify_state["phase"] == "done":
            # Build the clarifications record (only Q&As that were actually asked
            # — if user skip-all'd early, only the answered ones are kept).
            clarifications = []
            for i, ans in enumerate(clarify_state["answers"]):
                clarifications.append({
                    "question": clarify_state["questions"][i]["question"],
                    "answer": ans,
                })
            api_key = clarify_state["api_key"]
            mode = clarify_state["mode"]
            original_q = clarify_state["original_question"]
            del st.session_state["clarify_state"]
            start_run(api_key, mode, original_q, clarifications=clarifications)
            st.rerun()

    # ---- INPUT FORM ------------------------------------------------------
    elif not running and not have_result:
        # If a previous run set this flag, clear the question textarea state
        # BEFORE the widget renders.
        if st.session_state.get("_clear_question"):
            st.session_state["question_text"] = ""
            del st.session_state["_clear_question"]
        st.session_state.setdefault("question_text", "")

        mode = st.selectbox(
            "Mode",
            options=["scout", "verdict", "deep"],
            index=1,
            key="mode_select",
            help="scout ≈ ₹2-3 · verdict ≈ ₹7-8 · deep ≈ ₹15-17",
        )

        question = st.text_area(
            "Question",
            height=280,
            key="question_text",
            placeholder=(
                "Council this: [your decision in one sentence]\n\n"
                "Context:\n"
                "- [stake 1]\n"
                "- [stake 2]\n"
                "- [hard constraint]\n"
                "- [what's pulling you toward A]\n"
                "- [what's pulling you toward B]\n\n"
                "[The real question, sharpened.]"
            ),
            help="Richer context = sharper verdict. Don't pre-load the answer.",
        )

        # Similarity check (best-effort — never let a backend failure block the form)
        q_stripped = question.strip()
        if q_stripped and len(q_stripped) > 30:
            try:
                matches = history.find_similar(q_stripped, threshold=0.6)
            except Exception:
                matches = []
            if matches:
                sim, top_row = matches[0]
                st.warning(
                    f"💡 You've councilled something similar before — "
                    f"**{council.topic_to_title(top_row['topic_slug'])}** "
                    f"(#{top_row['id']}, {sim * 100:.0f}% match, "
                    f"{top_row['timestamp']}). Check the History tab."
                )

        # Feature A — clarification toggle
        clarify_choice = st.radio(
            "How would you like to proceed?",
            options=["Jump straight in", "Let the council ask me first"],
            horizontal=True,
            key="clarify_mode",
            help=(
                "‘Jump straight in’ runs the council immediately. "
                "‘Let the council ask me first’ has a fast clarifier ask 3-5 questions "
                "before the deliberation, so the advisors get sharper context."
            ),
        )

        generate_pdf = st.checkbox("Generate PDF", value=True, key="gen_pdf")
        submitted = st.button(
            "Run council", type="primary", use_container_width=True, key="run_btn"
        )

        with st.expander(f"Models in **{mode}** mode", expanded=False):
            cfg = council.MODEL_SETS[mode]
            st.markdown(f"**Chairman:** `{cfg['chairman']}`")
            st.markdown("**Advisors:**")
            for name, model in zip(council.ADVISOR_NAMES, cfg["advisors"]):
                st.markdown(f"- {name} → `{model}`")

        # Persistence backend indicator (small, non-intrusive)
        backend = history.backend()
        if backend == "turso":
            st.caption("📡 History: Turso (durable across restarts)")
        else:
            st.caption(
                "💾 History: local SQLite "
                "(ephemeral on Streamlit Cloud — set TURSO_* secrets for durability)"
            )

        if submitted:
            if not question.strip():
                st.error("Question is required.")
                st.stop()

            api_key = (
                _secret("OPENROUTER_KEY")
                or os.environ.get("OPENROUTER_KEY")
                or st.session_state.get("OPENROUTER_KEY")
            )
            if not api_key:
                st.error(
                    "OpenRouter key not configured. Set `OPENROUTER_KEY` in Streamlit "
                    "secrets (Cloud) or env (local)."
                )
                st.stop()

            st.session_state["last_generate_pdf"] = generate_pdf

            if clarify_choice == "Let the council ask me first":
                # Kick off Feature A's pre-deliberation clarification flow.
                st.session_state["clarify_state"] = {
                    "phase": "generating",
                    "original_question": question,
                    "mode": mode,
                    "api_key": api_key,
                    "questions": [],
                    "answers": [],
                    "current_idx": 0,
                }
            else:
                start_run(api_key, mode, question)
            st.rerun()

    # ---- RUNNING ---------------------------------------------------------
    if running:
        elapsed = time.time() - holder["started_at"]

        # Feature B — mid-deliberation interrupts. Surface the FIRST pending
        # interrupt (FIFO). Other advisors continue running while the user
        # answers; once submitted, the interrupted advisor's coroutine wakes,
        # re-runs with the answer injected, and rejoins the gather.
        pending_order = list(holder.get("interrupts_order", []))
        if pending_order:
            key = pending_order[0]
            interrupt = holder["interrupts_pending"].get(key)
            if interrupt:
                advisor_name = interrupt["advisor_name"]
                stage_label = (
                    "advisor" if interrupt["stage"] == "advisor" else "peer reviewer"
                )
                role = council.PERSONA_ROLES.get(advisor_name)
                role_subtitle = (
                    f"{stage_label} · {role}" if role else stage_label
                )

                st.markdown("### ⏸ The council needs a clarification")
                st.caption(
                    f"⏳ {holder['stage']}  ·  {elapsed:0.0f}s elapsed  ·  "
                    f"other advisors still running"
                )

                # Trail of any interrupts already answered this run
                log = holder.get("interrupts_log", [])
                if log:
                    with st.expander(
                        f"Earlier clarifications this run ({len(log)})",
                        expanded=False,
                    ):
                        for i, item in enumerate(log):
                            _render_answer_card(
                                i,
                                f"{item['advisor']} ({item['stage']}): {item['question']}",
                                item["answer"],
                            )

                def _submit_interrupt(answer: str):
                    interrupt["answer"] = answer
                    interrupt["event"].set()
                    st.rerun()

                def _skip_interrupt():
                    interrupt["answer"] = "(user skipped)"
                    interrupt["event"].set()
                    st.rerun()

                _render_question_card(
                    asker=advisor_name,
                    asker_role=role_subtitle,
                    question_text=interrupt["question"],
                    options=interrupt["options"],
                    allow_freetext=interrupt["allow_freetext"],
                    key_prefix=f"int_{key}",
                    on_submit=_submit_interrupt,
                    on_skip=_skip_interrupt,
                )

                if st.button(
                    "⏹ Stop the whole run",
                    type="secondary",
                    use_container_width=True,
                    key="stop_during_interrupt",
                ):
                    holder["cancel_event"].set()
                    interrupt["event"].set()  # unstick the waiter
                    st.toast("Stopping after the current stage finishes…", icon="⏹")
                    st.rerun()

                # Refresh more often while waiting on the user — keeps the
                # "other advisors still running" feel responsive.
                time.sleep(0.8)
                st.rerun()

        # No pending interrupts — show the normal running banner.
        st.markdown("### 🏛️ Council in session")
        st.info(f"⏳ {holder['stage']}  ·  {elapsed:0.0f}s elapsed")

        # Show any interrupts already resolved this run (small status line)
        log = holder.get("interrupts_log", [])
        if log:
            st.caption(
                f"📝 {len(log)} clarification{'s' if len(log) != 1 else ''} "
                f"answered this run"
            )

        if st.button("⏹ Stop", type="secondary", use_container_width=True):
            holder["cancel_event"].set()
            # Unblock any (just-arrived) interrupt waiters too
            for k, it in list(holder.get("interrupts_pending", {}).items()):
                it["event"].set()
            st.toast("Stopping after the current stage finishes…", icon="⏹")
            st.rerun()
        st.caption(
            "Cancellation takes effect between stages, so you may wait up to ~30s "
            "for the in-flight stage to finish."
        )
        time.sleep(1.0)
        st.rerun()

    # ---- CANCELLED / ERROR -----------------------------------------------
    if holder and holder["done"] and holder.get("cancelled"):
        st.warning("Run cancelled.")
        if st.button("Clear and start over", use_container_width=True):
            st.session_state.pop("holder", None)
            st.session_state["_clear_question"] = True
            st.rerun()

    if holder and holder["done"] and holder.get("error"):
        st.error(f"Run failed: {holder['error']}")
        if st.button("Clear and try again", use_container_width=True):
            st.session_state.pop("holder", None)
            st.session_state["_clear_question"] = True
            st.rerun()

    # ---- RESULT ----------------------------------------------------------
    if have_result:
        result = holder["result"]
        elapsed = holder.get("finished_at", time.time()) - holder["started_at"]
        timestamp = datetime.now()

        topic = result["topic"]
        mode = result["mode"]
        chairman_model = result["chairman_model"]
        advisor_models = result["advisor_models"]
        advisor_responses = result["advisor_responses"]
        reviews = result["reviews"]
        chairman_verdict = result["chairman_verdict"]
        letter_map = result["letter_map"]

        st.markdown("---")
        meta_cols = st.columns([3, 2])
        with meta_cols[0]:
            saved_id = holder.get("saved_id")
            history_line = (
                f"  \n**Saved to history:** #{saved_id}" if saved_id else ""
            )
            save_err = holder.get("history_save_error")
            if save_err:
                history_line = f"  \n**History save failed:** {save_err}"
            st.markdown(
                f"**Topic:** {council.topic_to_title(topic)}  \n"
                f"**Mode:** {mode.upper()}  \n"
                f"**Chairman:** `{chairman_model}`  \n"
                f"**Run took:** {elapsed:.1f}s"
                f"{history_line}"
            )
        with meta_cols[1]:
            if st.session_state.get("last_generate_pdf", True):
                with st.spinner("Building interactive PDF…"):
                    pdf_bytes = build_pdf_bytes(result, timestamp)
                if pdf_bytes:
                    filename = f"{topic}_{mode}_{timestamp.strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        "📥 Download PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True,
                    )

            if st.button("🔄 New question", use_container_width=True):
                st.session_state.pop("holder", None)
                st.session_state["_clear_question"] = True
                st.rerun()

        clarifications = result.get("clarifications", [])
        interrupts = result.get("interrupts", [])
        if clarifications or interrupts:
            with st.expander(
                f"Question & Context "
                f"({len(clarifications)} pre-deliberation, "
                f"{len(interrupts)} mid-deliberation)",
                expanded=False,
            ):
                if clarifications:
                    st.markdown("**Pre-deliberation clarifications**")
                    for i, c in enumerate(clarifications, 1):
                        st.markdown(f"**Q{i}.** {c['question']}")
                        st.markdown(f"_{c['answer']}_")
                if interrupts:
                    if clarifications:
                        st.markdown("---")
                    st.markdown("**Mid-deliberation clarifications**")
                    for i, it in enumerate(interrupts, 1):
                        stage_lbl = (
                            "advisor" if it.get("stage") == "advisor"
                            else "peer reviewer"
                        )
                        st.caption(f"{it['advisor']} · {stage_lbl}")
                        st.markdown(f"**Q{i}.** {it['question']}")
                        st.markdown(f"_{it['answer']}_")

        st.markdown("## 🏛️ Chairman's Verdict")
        st.markdown(chairman_verdict)

        st.markdown("---")
        st.markdown("## 📋 Advisor Responses")
        for name in council.ADVISOR_NAMES:
            with st.expander(f"**{name}** · `{advisor_models[name]}`", expanded=False):
                st.markdown(advisor_responses[name])

        st.markdown("---")
        st.markdown("## 🔍 Peer Reviews")
        st.caption(
            "Anonymization map this run: "
            + " · ".join(f"`{L}` = {n}" for L, n in letter_map.items())
        )
        for name in council.ADVISOR_NAMES:
            with st.expander(f"**{name}** reviews the council", expanded=False):
                st.markdown(reviews[name])


# =============================================================================
# TAB — HISTORY
# =============================================================================

with tab_history:
    try:
        rows = history.list_councils()
    except history.TursoError as e:
        st.error(
            "**History backend (Turso) error**\n\n```\n"
            f"{e}\n"
            "```\n\n"
            "Common causes: wrong DB URL, expired/revoked token, or DB region "
            "mismatch. Check your Streamlit secrets against the values in your "
            "Turso dashboard."
        )
        rows = []
    except Exception as e:
        st.error(f"Couldn't load history: `{type(e).__name__}: {e}`")
        rows = []

    if not rows:
        st.info("No past councils yet. Run one to start your archive.")
    else:
        st.caption(
            f"{len(rows)} council{'s' if len(rows) != 1 else ''} archived locally · "
            f"newest first"
        )

        for row in rows:
            title = council.topic_to_title(row["topic_slug"])
            ts = row["timestamp"]
            mode_badge = row["mode"].upper()
            cost = row["cost_estimate"]
            rid = row["id"]

            st.markdown(f"### {title}")
            st.caption(
                f"`{mode_badge}` · {ts} · ₹{cost:.1f} · #{rid} · "
                f"chairman `{row['chairman_model']}`"
            )

            with st.expander("Show verdict"):
                st.markdown(row["chairman_verdict"])
                st.caption("— original question —")
                st.markdown(f"> {row['question']}")

                row_clar = row.get("clarifications", []) or []
                row_int = row.get("interrupts", []) or []
                if row_clar:
                    st.caption("— pre-deliberation clarifications —")
                    for i, c in enumerate(row_clar, 1):
                        st.markdown(f"**Q{i}.** {c['question']}")
                        st.markdown(f"_{c['answer']}_")
                if row_int:
                    st.caption("— mid-deliberation clarifications —")
                    for i, it in enumerate(row_int, 1):
                        stage_lbl = (
                            "advisor" if it.get("stage") == "advisor"
                            else "peer reviewer"
                        )
                        st.markdown(
                            f"**{it['advisor']}** ({stage_lbl}) — "
                            f"_{it['question']}_"
                        )
                        st.markdown(f"→ {it['answer']}")

            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("📋 Load into Council", key=f"load_{rid}"):
                    st.session_state["question_text"] = row["question"]
                    st.session_state["mode_select"] = row["mode"]
                    # Clear any in-flight run / displayed result so the
                    # form re-renders fresh on the Council tab.
                    st.session_state.pop("holder", None)
                    st.toast(
                        "Loaded into Council tab. Switch tabs to run.",
                        icon="📋",
                    )
                    st.rerun()
            with c2:
                if st.button("🗑️ Delete", key=f"del_{rid}"):
                    history.delete_council(rid)
                    st.toast(f"Deleted #{rid}.", icon="🗑️")
                    st.rerun()

            st.markdown("---")

        with st.expander("⚠️ Danger zone — clear all history"):
            confirm = st.checkbox(
                "I understand this will permanently delete all council history.",
                key="clear_all_confirm",
            )
            if st.button(
                "Clear All History",
                type="secondary",
                disabled=not confirm,
                key="clear_all_btn",
            ):
                history.clear_all()
                st.session_state.pop("clear_all_confirm", None)
                st.toast("History cleared.", icon="🗑️")
                st.rerun()

st.markdown("---")
st.caption("LLM Council Refactor: Phases 1–3 Complete")
