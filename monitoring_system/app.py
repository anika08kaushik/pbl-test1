import streamlit as st
import cv2
import time
import requests
import json
from detection.yolo import YOLODetector
from detection.mediapipe_utils import FaceMonitor
from logic.rules import BehaviorEngine
from utils.video import VideoStream, annotate_frame
from utils.state import init_state, add_event, reset_state

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Mock Interview Monitor", layout="wide")

# Custom CSS for Premium Dashboard Cards
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-danger { color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Mock Interview Monitoring System")

# Initialize Session State
init_state()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Session Controls")
    if not st.session_state.is_monitoring:
        if st.button("🚀 Start Session", use_container_width=True):
            st.session_state.is_monitoring = True
            st.session_state.session_start_time = time.time()
            st.rerun()
    else:
        if st.button("🛑 Stop Session", use_container_width=True, type="primary"):
            st.session_state.is_monitoring = False
            st.rerun()

    st.divider()
    st.subheader("Ollama Configuration")
    ollama_model = st.selectbox("LLaVA Model", ["llava", "llama3"], index=0)
    st.info("AI Analysis logic runs post-session.")

# --- DASHBOARD METRICS (Big Cards) ---
st.subheader("📊 Real-time Monitoring Dashboard")
m1, m2, m3, m4, m5 = st.columns(5)

# Metrics Holders
eye_card = m1.empty()
face_card = m2.empty()
phone_card = m3.empty()
people_card = m4.empty()
score_card = m5.empty()

def update_ui_cards(stats):
    # 1. Eye Contact
    pct = stats.get("eye_contact_pct", 0)
    e_class = "Good" if pct > 70 else "Moderate" if pct > 40 else "Poor"
    e_color = "status-good" if pct > 70 else "status-warning" if pct > 40 else "status-danger"
    eye_card.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👁 Eye Contact</div>
            <div class="metric-value {e_color}">{pct}%</div>
            <div style="font-weight: 500;">{e_class}</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Face Presence
    f_present = stats.get("face_present", False)
    f_txt = "👤 Present" if f_present else "❌ Missing"
    f_color = "status-good" if f_present else "status-danger"
    face_card.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Face Presence</div>
            <div class="metric-value {f_color}">{f_txt}</div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Phone Detection
    p_detected = stats.get("phone_detected", False)
    p_txt = "📱 Phone!" if p_detected else "✅ Safe"
    p_color = "status-danger" if p_detected else "status-good"
    phone_card.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Phone Detection</div>
            <div class="metric-value {p_color}">{p_txt}</div>
        </div>
    """, unsafe_allow_html=True)

    # 4. Multiple People
    m_detected = stats.get("multiple_people", False)
    m_txt = "👥 Multiple!" if m_detected else "👤 Single"
    m_color = "status-danger" if m_detected else "status-good"
    people_card.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Multiple People</div>
            <div class="metric-value {m_color}">{m_txt}</div>
        </div>
    """, unsafe_allow_html=True)

    # 5. Suspicion Score
    score = stats.get("suspicion_score", 0)
    s_color = "status-good" if score < 30 else "status-warning" if score < 60 else "status-danger"
    score_card.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Suspicion</div>
            <div class="metric-value {s_color}">{score}/100</div>
        </div>
    """, unsafe_allow_html=True)

# Initial UI State
if not st.session_state.is_monitoring:
    update_ui_cards({"eye_contact_pct": 0, "suspicion_score": 0})

# --- MAIN VIDEO FEED ---
col_video, col_timeline = st.columns([2, 1])

with col_video:
    frame_placeholder = st.empty()
    if not st.session_state.is_monitoring:
        frame_placeholder.info("Click 'Start Session' to begin.")

with col_timeline:
    st.subheader("🕒 Event Timeline")
    timeline_placeholder = st.empty()

# --- MONITORING LOOP ---
if st.session_state.is_monitoring:
    yolo = YOLODetector()
    face_monitor = FaceMonitor()
    rules = BehaviorEngine()
    stream = VideoStream()
    
    try:
        while st.session_state.is_monitoring:
            frame = stream.get_frame()
            if frame is None: break
            
            yolo_dets = yolo.detect(frame)
            face_results = face_monitor.process(frame)
            
            # Analyze
            stats = rules.analyze(yolo_dets, face_results)
            
            # Update Dashboard
            update_ui_cards(stats)
            
            # Sync logs
            st.session_state.event_logs = rules.get_logs()
            st.session_state.suspicion_score = stats["suspicion_score"]
            
            # Display timeline
            if st.session_state.event_logs:
                timeline_placeholder.table(st.session_state.event_logs[-8:])
            
            # Annotate & Display
            display_frame = annotate_frame(frame, yolo_dets)
            display_frame = face_monitor.draw_landmarks(display_frame, face_results)
            frame_placeholder.image(stream.to_st_format(display_frame), use_container_width=True)
            
            time.sleep(0.01) # Low sleep for max FPS (~15-20 target)
            
    finally:
        stream.release()

# --- POST SESSION LLaVA SUMMARY ---
if not st.session_state.is_monitoring and st.session_state.event_logs:
    st.divider()
    st.subheader("🤖 AI Session Summary (Post-Session Study)")
    
    if st.button("Analyze with AI"):
        with st.spinner("Ollama is processing session logs..."):
            try:
                hist = "\n".join([f"{l['timestamp']}: {l['message']}" for l in st.session_state.event_logs])
                p = f"Summarize these interview proctoring logs. Identify patterns of distraction or cheating. \nLogs:\n{hist}"
                r = requests.post("http://localhost:11434/api/generate", json={"model": ollama_model, "prompt": p, "stream": False}, timeout=60)
                if r.status_code == 200: st.markdown(f"### AI Analysis:\n{r.json()['response']}")
                else: st.warning("Ollama error. Ensure server is running.")
            except Exception as e: st.error(f"Ollama connection failed: {e}")
