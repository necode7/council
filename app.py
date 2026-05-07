"""
LLM Council — Streamlit web app.

Usage:
    Local:  streamlit run app.py
    Cloud:  push to GitHub → deploy on share.streamlit.io with secrets:
              OPENROUTER_KEY = "sk-or-v1-..."
              APP_PASSWORD   = "<your password>"
"""

import asyncio
import io
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

import council


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
# SIMPLE PASSWORD GATE
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
    # If no password is configured, allow (for local dev convenience).
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
# RUN PIPELINE
# =============================================================================

async def run_council(api_key: str, mode: str, question: str):
    config = council.MODEL_SETS[mode]
    chairman_model = config["chairman"]
    advisor_models = dict(zip(council.ADVISOR_NAMES, config["advisors"]))

    async with httpx.AsyncClient() as client:
        advisor_responses = await council.run_advisors(
            client, api_key, advisor_models, question
        )
        reviews, letter_map = await council.run_peer_review(
            client, api_key, advisor_models, advisor_responses, question
        )
        chairman_verdict = await council.run_chairman(
            client, api_key, chairman_model, question, advisor_responses, reviews
        )

    return {
        "chairman_model": chairman_model,
        "advisor_models": advisor_models,
        "advisor_responses": advisor_responses,
        "reviews": reviews,
        "letter_map": letter_map,
        "chairman_verdict": chairman_verdict,
    }


def build_pdf_bytes(
    mode: str,
    chairman_model: str,
    advisor_models: dict,
    question: str,
    advisor_responses: dict,
    reviews: dict,
    chairman_verdict: str,
    timestamp: datetime,
) -> bytes | None:
    html = council.build_pdf_html(
        mode, chairman_model, advisor_models, question,
        advisor_responses, reviews, chairman_verdict, timestamp,
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
# UI
# =============================================================================

st.title("🏛️ LLM Council")
st.caption(
    "Five advisors debate your decision on different models. "
    "They peer-review each other anonymously. A chairman delivers the verdict."
)

with st.form("council_form", clear_on_submit=False):
    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.selectbox(
            "Mode",
            options=["scout", "verdict", "deep"],
            index=0,
            help="scout ≈ ₹2-3 · verdict ≈ ₹7-8 · deep ≈ ₹15-17",
        )
    with col2:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. Quit_job_for_startup",
            help="Short, filename-safe — used in the PDF name.",
        )

    question = st.text_area(
        "Question",
        height=260,
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

    generate_pdf = st.checkbox("Generate PDF", value=True)

    submitted = st.form_submit_button("Run council", type="primary", use_container_width=True)

# Show models used (just below the form so the user always sees what they'd get)
with st.expander(f"Models in **{mode}** mode", expanded=False):
    cfg = council.MODEL_SETS[mode]
    st.markdown(f"**Chairman:** `{cfg['chairman']}`")
    st.markdown("**Advisors:**")
    for name, model in zip(council.ADVISOR_NAMES, cfg["advisors"]):
        st.markdown(f"- {name} → `{model}`")


# =============================================================================
# EXECUTE
# =============================================================================

if submitted:
    # --- validation ---
    if not topic.strip():
        st.error("Topic is required.")
        st.stop()
    if not question.strip():
        st.error("Question is required.")
        st.stop()

    safe_topic = topic.strip()
    if not all(c.isalnum() or c in "_-" for c in safe_topic):
        st.error("Topic must be filename-safe: letters, digits, `_` or `-` only.")
        st.stop()

    import os
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

    started = time.time()

    with st.status("Running council…", expanded=True) as status:
        st.write(f"→ 5 advisors running in parallel on **{mode}** models…")
        try:
            result = asyncio.run(run_council(api_key, mode, question))
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            st.stop()

        st.write("✓ Advisors done")
        st.write("✓ Peer review done (anonymized)")
        st.write("✓ Chairman synthesizing…")
        elapsed = time.time() - started
        status.update(label=f"✓ Verdict ready ({elapsed:.0f}s)", state="complete")

    chairman_model = result["chairman_model"]
    advisor_models = result["advisor_models"]
    advisor_responses = result["advisor_responses"]
    reviews = result["reviews"]
    chairman_verdict = result["chairman_verdict"]

    # --- header strip with metadata + PDF download ---
    timestamp = datetime.now()
    st.markdown("---")
    meta_cols = st.columns([3, 2])
    with meta_cols[0]:
        st.markdown(
            f"**Topic:** {safe_topic}  \n"
            f"**Mode:** {mode.upper()}  \n"
            f"**Chairman:** `{chairman_model}`  \n"
            f"**Run took:** {elapsed:.1f}s"
        )
    with meta_cols[1]:
        if generate_pdf:
            with st.spinner("Building PDF…"):
                pdf_bytes = build_pdf_bytes(
                    mode, chairman_model, advisor_models, question,
                    advisor_responses, reviews, chairman_verdict, timestamp,
                )
            if pdf_bytes:
                filename = f"{safe_topic}_{mode}_{timestamp.strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    "📥 Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )

    # --- chairman's verdict ---
    st.markdown("## 🏛️ Chairman's Verdict")
    st.markdown(chairman_verdict)

    # --- advisor responses ---
    st.markdown("---")
    st.markdown("## 📋 Advisor Responses")
    for name in council.ADVISOR_NAMES:
        with st.expander(f"**{name}** · `{advisor_models[name]}`", expanded=False):
            st.markdown(advisor_responses[name])

    # --- peer reviews ---
    st.markdown("---")
    st.markdown("## 🔍 Peer Reviews")
    st.caption(
        f"Anonymization map this run: "
        + " · ".join(f"`{L}` = {n}" for L, n in result["letter_map"].items())
    )
    for name in council.ADVISOR_NAMES:
        with st.expander(f"**{name}** reviews the council", expanded=False):
            st.markdown(reviews[name])
