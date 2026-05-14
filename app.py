import sys
import tempfile
import time
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
import requests
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

sys.path.insert(0, str(Path(__file__).parent))

# Monitoring System Imports
from monitoring_system.detection.yolo import YOLODetector
from monitoring_system.detection.mediapipe_utils import FaceMonitor
from monitoring_system.logic.rules import BehaviorEngine
from monitoring_system.utils.video import VideoStream, annotate_frame

import agent_3_validator
import agent_2_evaluator
import agent_1_interviewer
import agent_4_assessor
from main import _extract_jd_requirements, _extract_resume_skill_items
from extractor import extract_resume, extract_skills
from vector_store import (
    store_resume_chunks,
    store_jd_requirements_tagged,
    store_skills,
    clear_collections,
)

# ── Cache heavy models ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading NLP models…")
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource(show_spinner=False)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_monitors():
    return YOLODetector(), FaceMonitor(), BehaviorEngine()

_load_cross_encoder()
_load_sentence_transformer()

# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Proctored Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INIT ---
if "step" not in st.session_state:
    st.session_state.step = "screening"
if "interview_transcript" not in st.session_state:
    st.session_state.interview_transcript = []
if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0
if "mcq_answers" not in st.session_state:
    st.session_state.mcq_answers = {}
if "dsa_response" not in st.session_state:
    st.session_state.dsa_response = ""
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""
if "proctoring_logs" not in st.session_state:
    st.session_state.proctoring_logs = []

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
            
            # Extract and store structured skills
            # ── KEY FIX: reuse jd_items (already cleaned by Ollama pipeline)
            # extract_skills(jd_text) is the regex path → produces garbage fragments
            resume_skills = extract_skills(" ".join([c.content for c in chunks]))
            store_skills(resume_skills, "resume")
            store_skills(jd_items, "jd")   # jd_items = Ollama-extracted, validated, deduped

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

with st.sidebar:
    st.title("🛡️ Secured Interview Suite")
    st.markdown("---")
    if st.button("🗑️ Reset Application", use_container_width=True):
        reset_all()
    st.divider()
    st.info("Continuous monitoring is active during Interview & Test phases.")

# --- SHARED MONITORING LOOP HELPER ---
def run_monitoring_frame(frame_placeholder, alert_placeholder, yolo, face_monitor, rules, stream):
    frame = stream.get_frame()
    if frame is not None:
        yolo_dets = yolo.detect(frame)
        face_results = face_monitor.process(frame)
        stats = rules.analyze(yolo_dets, face_results)
        
        # Draw & Display
        display_frame = annotate_frame(frame, yolo_dets)
        display_frame = face_monitor.draw_landmarks(display_frame, face_results)
        frame_placeholder.image(stream.to_st_format(display_frame), use_container_width=True)
        
        # Alerts
        if stats["status"] != "Normal":
            alert_placeholder.warning(f"⚠️ {stats['status']}: {stats['alerts'][-1] if stats['alerts'] else 'Unusual behavior detected'}")
        else:
            alert_placeholder.empty()
            
        st.session_state.proctoring_logs = rules.get_logs()
        return stats
    return None

# --- STEP 1: SCREENING ---
if st.session_state.step == "screening":
    st.title("🎯 AI Resume Screening")
    uploaded_pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    jd_title     = st.text_input("Job title")
    jd_text      = st.text_area("Paste job description", height=200)

    if st.button("🚀 Run Analysis", type="primary") and uploaded_pdf and jd_text.strip():
        with st.spinner("🔄 Processing pipeline..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            try:
                clear_collections()
                chunks = extract_resume(tmp_path)
                store_resume_chunks(chunks)
                jd_items = _extract_jd_requirements(jd_text)
                resume_items = _extract_resume_skill_items(chunks)
                store_jd_requirements_tagged(jd_items, resume_items, jd_title or "Target Role")

                st.session_state.ats_result = agent_3_validator.run(jd_text)
                st.session_state.eval_result = agent_2_evaluator.run(st.session_state.ats_result)
                st.session_state.questions = agent_1_interviewer.run(st.session_state.ats_result, st.session_state.eval_result)
                st.session_state.assessment = agent_4_assessor.generate_assessment(st.session_state.ats_result)
                st.rerun()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    if "ats_result" in st.session_state:
        st.success(f"✅ Screening Complete! Score: {st.session_state.ats_result.ats_score}%")
        if st.button("🏁 Start Proctored Interview", type="primary"):
            st.session_state.step = "interview"
            st.rerun()

# --- STEP 2: PROCTORED INTERVIEW ---
elif st.session_state.step == "interview":
    st.title("🎤 Proctored Mock Interview")
    
    col_cam, col_interview = st.columns([1, 1])
    
    with col_cam:
        st.subheader("📸 Live Monitoring")
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        
    with col_interview:
        q = st.session_state.questions
        all_qs = q.technical + q.behavioral + q.scenario_based
        
        if st.session_state.current_q_idx < len(all_qs):
            current_q = all_qs[st.session_state.current_q_idx]
            st.chat_message("assistant").write(current_q)
            
            # Voice Input
            st.write("---")
            st.write("🎤 **Push to Talk**")
            audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop & Submit", key="voice_recorder")
            
            if audio:
                # Transcribe
                recognizer = sr.Recognizer()
                try:
                    # Save bytes to temp file to use with SpeechRecognition if needed, 
                    # but recognize_google often takes the audio object directly or we convert.
                    # streamlit-mic-recorder returns bytes in ['bytes']
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                        tmp_audio.write(audio['bytes'])
                        tmp_audio_path = tmp_audio.name
                    
                    with sr.AudioFile(tmp_audio_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                        
                        st.session_state.interview_transcript.append({"q": current_q, "a": text})
                        st.session_state.current_q_idx += 1
                        Path(tmp_audio_path).unlink(missing_ok=True)
                        st.rerun()
                except Exception as e:
                    st.error(f"Voice Transcription failed: {e}. Please type below.")
            
            # Fallback Text Input
            user_text = st.chat_input("Or type your answer here...")
            if user_text:
                st.session_state.interview_transcript.append({"q": current_q, "a": user_text})
                st.session_state.current_q_idx += 1
                st.rerun()
        else:
            st.success("Interview Finished!")
            if st.button("📝 Start Assessments", type="primary"):
                st.session_state.step = "test"
                st.rerun()

    # Shared Monitoring Logic for Interview
    yolo, face, rules = get_monitors()
    with VideoStream() as stream:
        # We run a short loop until a rerun is triggered by inputs
        while st.session_state.step == "interview":
            run_monitoring_frame(frame_placeholder, alert_placeholder, yolo, face, rules, stream)
            time.sleep(0.01)

# --- STEP 3: PROCTORED TESTING ---
elif st.session_state.step == "test":
    st.title("📝 Proctored Assessment Center")
    col_cam, col_test = st.columns([1, 2])
    
    with col_cam:
        st.subheader("📸 Monitoring Active")
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()

    with col_test:
        assessment = st.session_state.assessment
        st.subheader("MCQ Test")
        for mcq in assessment.mcqs:
            st.session_state.mcq_answers[mcq.id] = st.radio(f"**{mcq.id}. {mcq.question}**", mcq.options, index=None, key=f"v_mcq_{mcq.id}")
        
        st.divider()
        st.subheader("DSA Coding Logic")
        st.info(f"**Problem:** {assessment.dsa.title}\n\n{assessment.dsa.description}")
        st.code(assessment.dsa.base_code)
        st.session_state.dsa_response = st.text_area("Your solution logic:", height=200, key="dsa_text")
        
        if st.button("📤 Submit All", type="primary"):
            st.session_state.step = "report"
            st.rerun()

    yolo, face, rules = get_monitors()
    with VideoStream() as stream:
        while st.session_state.step == "test":
            run_monitoring_frame(frame_placeholder, alert_placeholder, yolo, face, rules, stream)
            time.sleep(0.01)

# --- STEP 4: FINAL INTEGRITY REPORT ---
elif st.session_state.step == "report":
    st.title("📊 Integrated Performance & Integrity Report")
    
    tab_skills, tab_integrity = st.tabs(["🎯 Skill Evaluation", "🛡️ Proctoring Analytics"])
    
    with tab_skills:
        # (Existing report logic...)
        st.subheader("Interview & Test Scores")
        # Logic to summarize MCQ/DSA...
        st.info("Summary of candidate skill set based on interview and test performance.")
        
    with tab_integrity:
        st.subheader("Behavioral Integrity Log")
        if st.session_state.proctoring_logs:
            st.table(st.session_state.proctoring_logs)
            score = 100 - sum([l['score_increment'] for l in st.session_state.proctoring_logs])
            st.metric("Integrity Score", f"{max(0, score)}/100")
        else:
            st.success("Perfect Integrity! No suspicious behavior detected.")
            
    if st.button("🏁 Finish Session"):
        reset_all()
