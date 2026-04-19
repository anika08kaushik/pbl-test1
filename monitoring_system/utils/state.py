import streamlit as st
import time

def init_state():
    """Initializes the session state variables for monitoring."""
    if "is_monitoring" not in st.session_state:
        st.session_state.is_monitoring = False
    
    if "suspicion_score" not in st.session_state:
        st.session_state.suspicion_score = 0
    
    if "event_logs" not in st.session_state:
        st.session_state.event_logs = []
        
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = None
        
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = "Normal"

    if "frames_buffer" not in st.session_state:
        st.session_state.frames_buffer = [] # Store sampled frames for LLaVA

def add_event(message, severity="info"):
    """Adds an event to the log with a timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.event_logs.append({
        "timestamp": timestamp,
        "message": message,
        "severity": severity
    })
    st.session_state.last_alert = message

def update_score(increment):
    """Updates the suspicion score, capped at 100."""
    st.session_state.suspicion_score = min(100, st.session_state.suspicion_score + increment)

def reset_state():
    """Resets the state for a new session."""
    st.session_state.is_monitoring = False
    st.session_state.suspicion_score = 0
    st.session_state.event_logs = []
    st.session_state.session_start_time = None
    st.session_state.last_alert = "Normal"
    st.session_state.frames_buffer = []
