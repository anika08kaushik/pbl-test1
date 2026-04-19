# 🛡️ AI Screening & Interview Intelligence Suite

A dual-purpose platform designed to automate and monitor the hiring lifecycle using advanced LLMs and Computer Vision.

---

## 🏗️ Platform Overview

This repository contains two independent, high-performance systems that can coexist:

1.  **AI Resume Screener & Agent Interviewer**: A RAG-powered ATS that extracts skills, evaluates job fit, and generates tailored interview questions.
2.  **AI Mock Interview Monitor**: A real-time, vision-based monitoring system to detect cheating, distractions, and maintain interview integrity.

---

## 📂 Project Structure

```text
pbl-test1/
├── app.py                  # [System 1] Resume Screener Dashboard
├── agent_*.py              # [System 1] Multi-agent Reasoning Logic
├── monitoring_system/       # [System 2] Sub-project
│   ├── app.py              # [System 2] Monitoring Dashboard
│   ├── detection/          # [System 2] Vision Pipeline (YOLO/MediaPipe)
│   └── logic/              # [System 2] Behavioral Rules
└── README.md               # You are here
```

---

## 🚀 System 1: AI Resume Screener & Agent Interviewer

Analyze resumes against job descriptions with semantic precision.

### Key Features
- **ATS Scoring**: 0-100% match rating.
- **RAG-Powered Extraction**: Uses ChromaDB to provide evidence-based skill matching.
- **Agent Interrogation**: Automated Strengths/Gaps analysis.
- **Dynamic Questioning**: Generates Technical, Behavioral, and Scenario questions.

### How to Run
From the root directory:
```bash
python -m streamlit run app.py
```

---

## 🤖 System 2: AI Mock Interview Monitor (Real-time)

Monitor candidate behavior during online tests/interviews using AI.

### Key Features
- **👁️ Eye Contact Tracking**: Real-time percentage + classification (Good/Moderate/Poor).
- **📱 Phone Detection**: YOLOv8-powered smartphone detection.
- **👥 Multi-Person Alert**: Detects if more than one person is in the frame.
- **⚠️ Smart Cheating Logic**: Flags 5+ seconds of "Looking Down" combined with distraction signals.
- **🧠 Post-Session Analysis**: Generates an integrity summary via **Ollama (LLaVA)**.

### Prerequisites
- **Ollama**: Must be installed and running.
- **LLaVA Model**: Run `ollama pull llava` before starting.

### How to Run
From the root directory:
```bash
# Start the AI Summary backend first
ollama run llava

# In a new terminal, launch the dashboard
python -m streamlit run monitoring_system/app.py
```

---

## 🛠️ Setup & Installation

1. **Clone & Explore**:
   Ensure you are in the `pbl-test1` folder.

2. **Environment**:
   It is recommended to use the existing `venv`.
   ```bash
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r monitoring_system/requirements.txt
   ```

---

## 💡 Technical Stack
- **Dashboard**: Streamlit
- **CV**: OpenCV, YOLOv8 (Ultralytics), MediaPipe Tasks (v0.10.33+)
- **LLM/RAG**: Ollama (LLaVA/LLaMA3), LangChain (Optional), ChromaDB
- **Models**: `face_landmarker.task`, `yolov8n.pt`

---

> [!NOTE]
> **Performance**: The monitoring system is optimized to run at ~10-15 FPS on standard hardware by resizing input frames to 640x480.
