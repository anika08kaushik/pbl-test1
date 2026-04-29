import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine

from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, joinedload
from passlib.context import CryptContext

Base = declarative_base()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "app.db"

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # RECRUITER or INDIVIDUAL
    created_at = Column(DateTime, default=datetime.utcnow)

    jobs = relationship("JobDescription", back_populates="recruiter")
    resumes = relationship("Resume", back_populates="candidate")
    assessments = relationship("Assessment", back_populates="individual")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    recruiter_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, nullable=False)
    raw_text = Column(Text, nullable=False)
    requirements = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    recruiter = relationship("User", back_populates="jobs")
    resumes = relationship("Resume", back_populates="job", cascade="all, delete-orphan")


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    candidate_email = Column(String, index=True)
    file_path = Column(String)
    file_name = Column(String)
    jd_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=True)
    ats_score = Column(Float, default=0.0)
    matching_skills = Column(JSON, default=list)
    missing_skills = Column(JSON, default=list)
    analysis_data = Column(JSON, default=dict)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("User", back_populates="resumes")
    job = relationship("JobDescription", back_populates="resumes")
    assessments = relationship("Assessment", back_populates="resume", cascade="all, delete-orphan")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    individual_id = Column(Integer, ForeignKey("users.id"))
    mcq_score = Column(Float, default=0.0)
    dsa_code = Column(Text)
    dsa_feedback = Column(JSON, default=dict)
    integrity_score = Column(Float, default=100.0)
    overall_score = Column(Float, default=0.0)
    completed_at = Column(DateTime, default=datetime.utcnow)

    resume = relationship("Resume", back_populates="assessments")
    individual = relationship("User", back_populates="assessments")
    integrity_logs = relationship("IntegrityLog", back_populates="assessment", cascade="all, delete-orphan")


class IntegrityLog(Base):
    __tablename__ = "integrity_logs"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"))
    event_type = Column(String)
    message = Column(String)
    score_increment = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)

    assessment = relationship("Assessment", back_populates="integrity_logs")


class UserCreate(BaseModel):
    email: str
    password: str
    role: str  # RECRUITER or INDIVIDUAL


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_user(email: str, password: str, role: str) -> User:
    password_hash = pwd_context.hash(password)
    user = User(email=email, password_hash=password_hash, role=role)
    db = SessionLocal()
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def get_or_create_google_user(email: str) -> User:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Specific mappings requested by user
            role = "INDIVIDUAL" # Default
            if email == "moksh8600.beaift24@chitkara.edu.in":
                role = "RECRUITER"
            elif email == "mokshkulshrestha@gmail.com":
                role = "INDIVIDUAL"
            
            # For Google users, we don't need a real password hash
            user = User(email=email, password_hash="GOOGLE_OAUTH_ACCOUNT", role=role)
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()



def authenticate_user(email: str, password: str) -> Optional[User]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and pwd_context.verify(password, user.password_hash):
            return user
        return None
    finally:
        db.close()


def get_user_by_email(email: str) -> Optional[User]:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()


def create_job(recruiter_id: int, title: str, raw_text: str, requirements: list) -> JobDescription:
    db = SessionLocal()
    try:
        job = JobDescription(
            recruiter_id=recruiter_id,
            title=title,
            raw_text=raw_text,
            requirements=requirements
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    finally:
        db.close()


def get_jobs(recruiter_id: int) -> list:
    db = SessionLocal()
    try:
        return db.query(JobDescription).options(joinedload(JobDescription.resumes)).filter(JobDescription.recruiter_id == recruiter_id).all()
    finally:
        db.close()


def delete_job(job_id: int) -> bool:
    db = SessionLocal()
    try:
        job = db.query(JobDescription).filter(JobDescription.id == job_id).first()
        if job:
            db.delete(job)
            db.commit()
            return True
        return False
    finally:
        db.close()



def create_resume(candidate_email: str, file_path: str, file_name: str, jd_id: Optional[int], analysis_data: dict) -> Resume:
    db = SessionLocal()
    try:
        resume = Resume(
            candidate_email=candidate_email,
            file_path=file_path,
            file_name=file_name,
            jd_id=jd_id,
            ats_score=analysis_data.get("ats_score", 0.0),
            matching_skills=analysis_data.get("matching_skills", []),
            missing_skills=analysis_data.get("missing_skills", []),
            analysis_data=analysis_data
        )
        db.add(resume)
        db.commit()
        db.refresh(resume)
        return resume
    finally:
        db.close()


def get_resumes_by_jd(jd_id: int) -> list:
    db = SessionLocal()
    try:
        return db.query(Resume).filter(Resume.jd_id == jd_id).order_by(Resume.ats_score.desc()).all()
    finally:
        db.close()


def create_assessment(resume_id: int, individual_id: int, mcq_score: float, dsa_code: str, dsa_feedback: dict, integrity_score: float) -> Assessment:
    db = SessionLocal()
    try:
        overall = (mcq_score * 0.5) + (integrity_score * 0.5)
        assessment = Assessment(
            resume_id=resume_id,
            individual_id=individual_id,
            mcq_score=mcq_score,
            dsa_code=dsa_code,
            dsa_feedback=dsa_feedback,
            integrity_score=integrity_score,
            overall_score=overall
        )
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        return assessment
    finally:
        db.close()


def get_assessments_by_user(user_id: int) -> list:
    db = SessionLocal()
    try:
        return db.query(Assessment).filter(Assessment.individual_id == user_id).order_by(Assessment.completed_at.desc()).all()
    finally:
        db.close()


def add_integrity_log(assessment_id: int, event_type: str, message: str, score_increment: int) -> IntegrityLog:
    db = SessionLocal()
    try:
        log = IntegrityLog(
            assessment_id=assessment_id,
            event_type=event_type,
            message=message,
            score_increment=score_increment
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log
    finally:
        db.close()


def get_resume_by_id(resume_id: int) -> Optional[Resume]:
    db = SessionLocal()
    try:
        return db.query(Resume).filter(Resume.id == resume_id).first()
    finally:
        db.close()


def update_resume_analysis(resume_id: int, ats_score: float, matching_skills: list, missing_skills: list, analysis_data: dict):
    db = SessionLocal()
    try:
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if resume:
            resume.ats_score = ats_score
            resume.matching_skills = matching_skills
            resume.missing_skills = missing_skills
            resume.analysis_data = analysis_data
            db.commit()
    finally:
        db.close()