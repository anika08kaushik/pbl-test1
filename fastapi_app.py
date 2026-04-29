from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
from pathlib import Path
import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from vector_store import (
    store_resume_chunks,
    store_jd_requirements_tagged,
    clear_collections,
)
from extractor import extract_resume
from main import _extract_jd_requirements, _extract_resume_skill_items
import agent_3_validator, agent_2_evaluator, agent_1_interviewer, agent_4_assessor, agent_5_interview_evaluator
from api_utils import create_session, get_session, decode_base64_frame, log_integrity_event
from monitoring_system.detection.yolo import YOLODetector
from monitoring_system.detection.mediapipe_utils import FaceMonitor
from monitoring_system.detection.pose_analyzer import PoseAnalyzer
from monitoring_system.logic.rules import BehaviorEngine
from database import (
    init_db, create_user, authenticate_user, get_user_by_email,
    create_job, get_jobs, create_resume, get_resumes_by_jd, get_assessments_by_user,
    create_assessment, add_integrity_log, get_resume_by_id, update_resume_analysis, delete_job, get_or_create_google_user,
    User, JobDescription, Resume, Assessment, SessionLocal
)
from code_executor import execute_code

# Initialize DB and create demo users
init_db()
db = SessionLocal()
try:
    if not get_user_by_email("recruiter@demo.ai"):
        create_user("recruiter@demo.ai", "password123", "RECRUITER")
    if not get_user_by_email("candidate@demo.ai"):
        create_user("candidate@demo.ai", "password123", "INDIVIDUAL")
finally:
    db.close()

app = FastAPI(title="SmartHire API", version="2.1.0")

# Lazy load detectors to speed up startup for simple endpoints
yolo = None
face_monitor = None
pose_analyzer = None
rules = None

def get_detectors():
    global yolo, face_monitor, pose_analyzer, rules
    if yolo is None:
        yolo = YOLODetector()
        face_monitor = FaceMonitor()
        pose_analyzer = PoseAnalyzer()
        rules = BehaviorEngine()
    return yolo, face_monitor, pose_analyzer, rules

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_progress(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# --- AUTH ENDPOINTS ---

class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    try:
        user = create_user(req.email, req.password, req.role)
        return {"id": user.id, "email": user.email, "role": user.role}
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="Email already registered")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"id": user.id, "email": user.email, "role": user.role, "token": f"demo-token-{user.id}"}

class GoogleAuthRequest(BaseModel):
    email: str

@app.post("/api/auth/google")
async def google_auth(req: GoogleAuthRequest):
    user = get_or_create_google_user(req.email)
    return {"id": user.id, "email": user.email, "role": user.role, "token": f"google-token-{user.id}"}


# --- RECRUITER ENDPOINTS ---

class JobCreateRequest(BaseModel):
    title: str
    raw_text: str
    requirements: List[str]

@app.get("/api/jobs")
async def list_jobs(user_id: int):
    jobs = get_jobs(user_id)
    return [
        {
            "id": j.id,
            "title": j.title,
            "raw_text": j.raw_text,
            "requirements": j.requirements,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "resume_count": len(j.resumes) if j.resumes else 0
        }
        for j in jobs
    ]

@app.post("/api/jobs")
async def create_job_endpoint(req: JobCreateRequest, user_id: int):
    job = create_job(user_id, req.title, req.raw_text, req.requirements)
    return {"id": job.id, "title": job.title, "created_at": job.created_at.isoformat()}

@app.delete("/api/jobs/{job_id}")
async def delete_job_endpoint(job_id: int, user_id: int):
    success = delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "success", "message": "Job deleted"}

@app.post("/api/recruiter/upload-resumes")
async def upload_resumes(
    job_id: int = Form(...),
    files: List[UploadFile] = File(...),
):
    results = []
    # Fetch job text
    db_session = SessionLocal()
    job = db_session.query(JobDescription).filter(JobDescription.id == job_id).first()
    db_session.close()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        try:
            clear_collections()
            chunks = extract_resume(tmp_path)
            store_resume_chunks(chunks)
            jd_items = _extract_jd_requirements(job.raw_text)
            resume_items = _extract_resume_skill_items(chunks)
            store_jd_requirements_tagged(jd_items, resume_items, job.title)
            
            ats = agent_3_validator.run(job.raw_text)
            
            result_data = {
                "ats_score": ats.ats_score,
                "matching_skills": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in ats.matching_skills],
                "missing_skills": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in ats.missing_skills],
            }
            
            resume = create_resume(
                candidate_email=f"bulk_{int(time.time())}@example.com",
                file_path=tmp_path,
                file_name=file.filename,
                jd_id=job_id,
                analysis_data=result_data
            )
            
            results.append({"id": resume.id, "file": file.filename, "score": ats.ats_score})
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return {"status": "success", "processed": len(results), "results": results}

@app.get("/api/recruiter/candidates/{jd_id}")
async def get_candidates(jd_id: int):
    resumes = get_resumes_by_jd(jd_id)
    return [
        {
            "id": r.id,
            "candidate_email": r.candidate_email,
            "file_name": r.file_name,
            "ats_score": r.ats_score,
            "matching_skills": r.matching_skills,
            "missing_skills": r.missing_skills,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None
        }
        for r in resumes
    ]

# --- INDIVIDUAL ENDPOINTS ---

@app.get("/api/individual/profile")
async def get_profile(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    assessments = get_assessments_by_user(user_id)
    return {
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "total_assessments": len(assessments),
        "average_score": sum(a.overall_score for a in assessments) / len(assessments) if assessments else 0
    }

@app.get("/api/individual/history")
async def get_history(user_id: int):
    assessments = get_assessments_by_user(user_id)
    return [
        {
            "id": a.id,
            "mcq_score": a.mcq_score,
            "integrity_score": a.integrity_score,
            "overall_score": a.overall_score,
            "completed_at": a.completed_at.isoformat() if a.completed_at else None
        }
        for a in assessments
    ]

@app.get("/api/individual/resume-suggestions")
async def get_resume_suggestions(user_id: int = 2):
    # Fetch the last resume uploaded by this user
    db_session = SessionLocal()
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user or not user.resumes:
        db_session.close()
        return {"suggestions": "Upload your resume in the assessment suite to get suggestions."}
    
    last_resume = user.resumes[-1]
    db_session.close()
    
    return {
        "missing_skills": last_resume.missing_skills,
        "matching_skills": last_resume.matching_skills,
        "ats_score": last_resume.ats_score,
        "file_name": last_resume.file_name
    }

# --- ASSESSMENT & PROCTORING ---

@app.post("/api/screen")
async def screen(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    job_title: str = Form("Target Role"),
    candidate_email: str = Form("candidate@demo.ai")
):
    session_id = create_session()
    session = get_session(session_id)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(resume.filename).suffix) as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name
    
    try:
        clear_collections()
        chunks = extract_resume(tmp_path)
        store_resume_chunks(chunks)
        jd_items = _extract_jd_requirements(jd_text)
        resume_items = _extract_resume_skill_items(chunks)
        store_jd_requirements_tagged(jd_items, resume_items, job_title)
        
        ats = agent_3_validator.run(jd_text)
        evaluation = agent_2_evaluator.run(ats)
        questions = agent_1_interviewer.run(ats, evaluation)
        assessment = agent_4_assessor.generate_assessment(ats)
        
        data = {
            "ats_result": ats.model_dump() if hasattr(ats, "model_dump") else ats.dict(),
            "evaluation": evaluation.model_dump() if hasattr(evaluation, "model_dump") else evaluation.dict(),
            "interview_questions": questions.model_dump() if hasattr(questions, "model_dump") else questions.dict(),
            "assessment": assessment.model_dump() if hasattr(assessment, "model_dump") else assessment.dict(),
        }
        session["screener"] = data
        
        # Link to user if exists
        db_session = SessionLocal()
        user = db_session.query(User).filter(User.email == candidate_email).first()
        db_session.close()
        
        create_resume(
            candidate_email=candidate_email,
            file_path=tmp_path,
            file_name=resume.filename,
            jd_id=None,
            analysis_data=data
        )
        
        return {"session_id": session_id, "data": data}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

class ProctorRequest(BaseModel):
    session_id: str
    frame_b64: str

@app.post("/api/proctor")
async def proctor(req: ProctorRequest):
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    frame = decode_base64_frame(req.frame_b64)
    if frame is None:
        return {"status": "Error", "message": "Invalid frame"}
    
    yolo_obj, face_obj, pose_obj, rules_obj = get_detectors()
    
    yolo_dets = yolo_obj.detect(frame)
    face_results = face_obj.process(frame)
    pose_results = pose_obj.analyze_pose(frame)
    stats = rules_obj.analyze(yolo_dets, face_results, pose_results)
    
    if stats["status"] != "Normal":
        last_log = session["integrity"][-1] if session["integrity"] else None
        if not last_log or last_log["status"] != stats["status"] or (time.time() - session.get("last_log_time", 0) > 5):
            log_integrity_event(
                req.session_id,
                stats["status"],
                stats["alerts"][-1] if stats["alerts"] else "Suspicious behavior",
                score_increment=5
            )
            session["last_log_time"] = time.time()
    
    # Store behavior data for report
    if "behavior_data" not in session:
        session["behavior_data"] = []
    session["behavior_data"].append({
        "timestamp": time.time(),
        "confidence": stats["behavior"]["confidence_level"],
        "posture": stats["behavior"]["posture_score"],
        "fidgeting": stats["behavior"]["fidgeting_rate"]
    })
    
    return {
        "status": stats["status"],
        "suspicion_score": session["suspicion_score"],
        "alerts": stats["alerts"],
        "behavior": stats["behavior"]
    }

@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get behavior summary from rules engine
    _, _, _, rules_obj = get_detectors()
    behavior_summary = rules_obj.get_behavior_summary()

    return {
        "screener": session["screener"],
        "integrity": session["integrity"],
        "suspicion_score": session["suspicion_score"],
        "behavior_summary": behavior_summary,
        "behavior_data": session.get("behavior_data", []),
        "mock_interview_feedback": session.get("mock_interview_feedback", {}),
        "dsa_feedback": session.get("dsa_feedback", {})
    }

class CodeExecuteRequest(BaseModel):
    code: str
    language: str
    test_cases: List[dict] = []

@app.post("/api/code/execute")
async def execute_code_endpoint(req: CodeExecuteRequest):
    result = execute_code(req.code, req.language, req.test_cases)
    return result

class AssessmentSubmitRequest(BaseModel):
    session_id: str
    interview_answers: List[dict] = []
    resume_id: Optional[int] = 1
    user_id: Optional[int] = 2
    mcq_score: float
    dsa_code: str
    dsa_feedback: dict
    integrity_score: float

@app.post("/api/individual/submit-assessment")
async def submit_assessment(req: AssessmentSubmitRequest):
    # Evaluate interview answers using agent_5
    mock_interview_feedback = {}
    if req.interview_answers:
        try:
            mock_interview_feedback = agent_5_interview_evaluator.run(req.interview_answers)
        except Exception as e:
            print(f"[submit-assessment] Interview evaluation failed: {e}")
            mock_interview_feedback = {
                "overall_impression": "Evaluation unavailable.",
                "strengths": [],
                "areas_for_improvement": ["Unable to evaluate interview answers."],
                "technical_accuracy": "N/A"
            }

    # Save feedback to session
    session = get_session(req.session_id)
    if session:
        session["mock_interview_feedback"] = mock_interview_feedback
        session["dsa_feedback"] = req.dsa_feedback

    assessment = create_assessment(
        resume_id=req.resume_id or 1,
        individual_id=req.user_id or 2,
        mcq_score=req.mcq_score,
        dsa_code=req.dsa_code,
        dsa_feedback=req.dsa_feedback,
        integrity_score=req.integrity_score
    )
    return {"id": assessment.id, "overall_score": assessment.overall_score}

@app.websocket("/ws/progress/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)