"""
app.py — Streamlit UI for AI Resume Screener

Key design decisions that prevent Streamlit re-execution bugs:
  1. ALL imports at top level — never inside button handlers
  2. @st.cache_resource on heavy model loaders — loaded once, reused forever
  3. Pipeline result stored in st.session_state — survives reruns without re-running
  4. clear_collections() called only when the button is actually clicked
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

# ── All imports at top level ──────────────────────────────────────────────────
# Moving imports inside the button handler was the root cause of triple execution.
# Streamlit reruns the full script on every widget interaction; importing inside
# the handler caused module-level side effects (model loading, DB init) to fire
# multiple times. Top-level imports are cached by Python's module system.

sys.path.insert(0, str(Path(__file__).parent))

import agent_3_validator
import agent_2_evaluator
import agent_1_interviewer
from main import _extract_jd_requirements, _extract_resume_skill_items
from extractor import extract_resume
from vector_store import (
    store_resume_chunks,
    store_jd_requirements_tagged,
    clear_collections,
)


# ── Cache heavy models — loaded ONCE per process, never again ─────────────────
# Without this, CrossEncoder and SentenceTransformer reload on every rerun.

@st.cache_resource(show_spinner="Loading NLP models…")
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner=False)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# Warm up at startup so first click is instant
_load_cross_encoder()
_load_sentence_transformer()


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("AI Resume Screener")
    st.markdown("---")
    match_threshold   = st.slider("Match threshold",   0.40, 0.90, 0.65, 0.01)
    partial_threshold = st.slider("Partial threshold", 0.25, 0.65, 0.38, 0.01)
    st.markdown("---")
    if st.button("🗑️ Clear results"):
        for key in ["ats_result", "eval_result", "questions", "run_error"]:
            st.session_state.pop(key, None)
        st.rerun()

st.title("🎯 AI Resume Screener")

uploaded_pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
jd_title     = st.text_input("Job title")
jd_text      = st.text_area("Paste job description", height=200)

run_btn = st.button("🚀 Run Screener", type="primary")


# ── Pipeline — runs ONLY when button is clicked ───────────────────────────────
# Results are stored in session_state so the display block below can render
# them on subsequent reruns without re-running the expensive pipeline.

if run_btn and uploaded_pdf and jd_text.strip():
    # Clear stale results
    for key in ["ats_result", "eval_result", "questions", "run_error", "run_traceback"]:
        st.session_state.pop(key, None)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        # Apply sidebar thresholds
        agent_3_validator.MATCH_THRESHOLD   = match_threshold
        agent_3_validator.PARTIAL_THRESHOLD = partial_threshold

        with st.spinner("🔄 Running pipeline…"):
            clear_collections()

            chunks = extract_resume(tmp_path)
            store_resume_chunks(chunks)

            jd_items     = _extract_jd_requirements(jd_text)
            resume_items = _extract_resume_skill_items(chunks)
            store_jd_requirements_tagged(
                jd_items=jd_items,
                resume_items=resume_items,
                title=jd_title or "Target Role",
            )

            ats_result  = agent_3_validator.run(jd_text)
            eval_result = agent_2_evaluator.run(ats_result)
            questions   = agent_1_interviewer.run(ats_result, eval_result)

        st.session_state["ats_result"]  = ats_result
        st.session_state["eval_result"] = eval_result
        st.session_state["questions"]   = questions

    except Exception as e:
        import traceback
        st.session_state["run_error"]     = str(e)
        st.session_state["run_traceback"] = traceback.format_exc()

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    # Rerun once to render results from session_state cleanly
    # (prevents the display block from executing in the same pass as the pipeline)
    st.rerun()


# ── Display — reads from session_state, NEVER re-runs pipeline ───────────────

if "run_error" in st.session_state:
    st.error(f"❌ Error: {st.session_state['run_error']}")
    with st.expander("Traceback"):
        st.code(st.session_state.get("run_traceback", ""))

elif "ats_result" in st.session_state:
    ats = st.session_state["ats_result"]
    ev  = st.session_state["eval_result"]
    q   = st.session_state["questions"]

    st.success("✅ Analysis complete!")

    # Score metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ATS Score",       f"{ats.ats_score}%")
    col2.metric("✅ Matched",      len(ats.matching_skills))
    col3.metric("⚠️ Partial",      len(ats.partial_matches))
    col4.metric("❌ Missing",      len(ats.missing_skills))

    st.markdown("---")

    # Skill breakdown
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("✅ Matched Skills")
        if ats.matching_skills:
            for m in ats.matching_skills:
                st.markdown(f"- **{m.requirement}** `{m.similarity_score:.2f}`")
        else:
            st.write("None")

        st.subheader("⚠️ Partial Matches")
        if ats.partial_matches:
            for m in ats.partial_matches:
                st.markdown(f"- **{m.requirement}** `{m.similarity_score:.2f}`")
        else:
            st.write("None")

    with col_r:
        st.subheader("❌ Missing Skills")
        if ats.missing_skills:
            for m in ats.missing_skills:
                st.markdown(f"- **{m.requirement}** `{m.similarity_score:.2f}`")
        else:
            st.write("None")

    st.markdown("---")

    # Evaluation
    st.subheader("📋 Evaluation")
    fit_color = {"strong": "🟢", "moderate": "🟡", "weak": "🔴"}.get(ev.overall_fit, "⚪")
    st.markdown(f"**Overall Fit:** {fit_color} {ev.overall_fit.upper()}")
    st.markdown(f"_{ev.qualitative_feedback}_")

    col_s, col_g = st.columns(2)
    with col_s:
        st.markdown("**💪 Strengths**")
        for s in ev.strengths:
            st.markdown(f"- {s}")
    with col_g:
        st.markdown("**🔧 Gaps**")
        for g in ev.gaps:
            st.markdown(f"- {g}")

    st.markdown("---")

    # Interview Questions
    st.subheader("🎤 Interview Questions")
    tab1, tab2, tab3 = st.tabs(["🔧 Technical", "🤝 Behavioral", "📐 Scenario-Based"])
    with tab1:
        for i, qt in enumerate(q.technical, 1):
            st.markdown(f"**Q{i}.** {qt}")
    with tab2:
        for i, qt in enumerate(q.behavioral, 1):
            st.markdown(f"**Q{i}.** {qt}")
    with tab3:
        for i, qt in enumerate(q.scenario_based, 1):
            st.markdown(f"**Q{i}.** {qt}")

elif run_btn and (not uploaded_pdf or not jd_text.strip()):
    st.warning("⚠️ Please upload a resume PDF and paste a job description.")
