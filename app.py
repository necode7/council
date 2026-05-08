"""
LLM Council — Streamlit web app.

Usage:
    Local:  streamlit run app.py
    Cloud:  push to GitHub → deploy on share.streamlit.io with secrets:
              OPENROUTER_KEY = "sk-or-v1-..."
              APP_PASSWORD   = "<your password>"
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
import history


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

class Cancelled(Exception):
    pass


async def _run_pipeline(api_key: str, mode: str, question: str, holder: dict):
    config = council.MODEL_SETS[mode]
    chairman_model = config["chairman"]
    advisor_models = dict(zip(council.ADVISOR_NAMES, config["advisors"]))
    cancel_event: threading.Event = holder["cancel_event"]

    def check_cancel():
        if cancel_event.is_set():
            raise Cancelled()

    async with httpx.AsyncClient() as client:
        holder["stage"] = "Reading your question…"
        topic = await council.topic_from_question(client, api_key, question)
        holder["topic"] = topic
        check_cancel()

        holder["stage"] = "5 advisors deliberating in parallel…"
        advisors = await council.run_advisors(client, api_key, advisor_models, question)
        check_cancel()

        holder["stage"] = "Anonymous peer review…"
        reviews, letter_map = await council.run_peer_review(
            client, api_key, advisor_models, advisors, question
        )
        check_cancel()

        holder["stage"] = "Chairman delivering verdict…"
        verdict = await council.run_chairman(
            client, api_key, chairman_model, question, advisors, reviews
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
        "question": question,
    }


def start_run(api_key: str, mode: str, question: str):
    """Spawn a thread that runs the pipeline and updates session_state['holder']."""
    holder = {
        "stage": "Starting…",
        "started_at": time.time(),
        "done": False,
        "cancelled": False,
        "result": None,
        "error": None,
        "cancel_event": threading.Event(),
        "topic": None,
    }
    st.session_state["holder"] = holder

    def runner():
        try:
            result = asyncio.run(_run_pipeline(api_key, mode, question, holder))
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
# UI — TITLE + TABS
# =============================================================================

st.title("🏛️ LLM Council")
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

    # Hide the form while a run is in flight or after a result is shown.
    #
    # We're NOT using st.form here on purpose — forms batch their values until
    # submit, which means the "Models in this mode" expander wouldn't update
    # when the user changes the Mode dropdown. Outside of a form, every widget
    # change reruns the script and keeps the expander in sync.
    if not running and not have_result:
        # If a previous run set this flag, clear the question textarea state
        # BEFORE the widget renders. Streamlit only honours session_state writes
        # to a widget's key when the write happens before the widget is created.
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

        # Similarity check — runs on every rerun (cheap; local SQLite, ≤100s of rows).
        q_stripped = question.strip()
        if q_stripped and len(q_stripped) > 30:
            matches = history.find_similar(q_stripped, threshold=0.6)
            if matches:
                sim, top_row = matches[0]
                st.warning(
                    f"💡 You've councilled something similar before — "
                    f"**{council.topic_to_title(top_row['topic_slug'])}** "
                    f"(#{top_row['id']}, {sim * 100:.0f}% match, "
                    f"{top_row['timestamp']}). Check the History tab."
                )

        generate_pdf = st.checkbox("Generate PDF", value=True, key="gen_pdf")
        submitted = st.button(
            "Run council", type="primary", use_container_width=True, key="run_btn"
        )

        # Live-updates whenever `mode` changes, because we're outside a form.
        with st.expander(f"Models in **{mode}** mode", expanded=False):
            cfg = council.MODEL_SETS[mode]
            st.markdown(f"**Chairman:** `{cfg['chairman']}`")
            st.markdown("**Advisors:**")
            for name, model in zip(council.ADVISOR_NAMES, cfg["advisors"]):
                st.markdown(f"- {name} → `{model}`")

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

            start_run(api_key, mode, question)
            st.rerun()

    # ---- RUNNING ---------------------------------------------------------
    if running:
        elapsed = time.time() - holder["started_at"]
        st.markdown("### 🏛️ Council in session")
        st.info(f"⏳ {holder['stage']}  ·  {elapsed:0.0f}s elapsed")
        if st.button("⏹ Stop", type="secondary", use_container_width=True):
            holder["cancel_event"].set()
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
    rows = history.list_councils()

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
