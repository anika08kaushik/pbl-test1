from pydantic import BaseModel
from typing import List, Optional


class ResumeChunk(BaseModel):
    chunk_id: str
    section: str          # "skills", "experience", "education", "projects", "summary"
    content: str
    metadata: dict = {}


class JobDescription(BaseModel):
    title: str
    raw_text: str
    requirements: List[str] = []   # individual requirement sentences after parsing


class MatchedSkill(BaseModel):
    requirement: str
    resume_excerpt: str
    similarity_score: float


class MissingSkill(BaseModel):
    requirement: str
    similarity_score: float           # best score found (below threshold)


class ATSResult(BaseModel):
    ats_score: float                  # 0–100
    matching_skills: List[MatchedSkill]
    missing_skills: List[MissingSkill]
    partial_matches: List[MatchedSkill]   # 0.55–0.75 range
    summary: str


class EvaluationResult(BaseModel):
    qualitative_feedback: str
    strengths: List[str]
    gaps: List[str]
    overall_fit: str                  # "strong" | "moderate" | "weak"


class InterviewQuestions(BaseModel):
    technical: List[str]
    behavioral: List[str]
    scenario_based: List[str]


class ScreenerOutput(BaseModel):
    ats_result: ATSResult
    evaluation: EvaluationResult
    interview_questions: InterviewQuestions
