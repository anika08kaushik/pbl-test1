from pydantic import BaseModel
from typing import List, Optional

class ResumeChunk(BaseModel):
    chunk_id: str
    section: str
    content: str
    metadata: dict = {}

class JobDescription(BaseModel):
    title: str
    raw_text: str
    requirements: List[str] = []

class MatchedSkill(BaseModel):
    requirement: str
    resume_excerpt: str
    similarity_score: float

class MissingSkill(BaseModel):
    requirement: str
    similarity_score: float

class ATSResult(BaseModel):
    ats_score: float
    matching_skills: List[MatchedSkill]
    missing_skills: List[MissingSkill]
    partial_matches: List[MatchedSkill]
    summary: str

class EvaluationResult(BaseModel):
    qualitative_feedback: str
    strengths: List[str]
    gaps: List[str]
    overall_fit: str

class InterviewQuestions(BaseModel):
    technical: List[str]
    behavioral: List[str]
    scenario_based: List[str]

# --- NEW ASSESSMENT MODELS ---

class MCQQuestion(BaseModel):
    id: int
    question: str
    options: List[str]
    correct_idx: int
    explanation: str

class DSAQuestion(BaseModel):
    title: str
    description: str
    constraints: List[str]
    base_code: str
    solution_logic: str # For AI comparison

class AssessmentData(BaseModel):
    mcqs: List[MCQQuestion]
    dsa: DSAQuestion

class ScreenerOutput(BaseModel):
    ats_result: ATSResult
    evaluation: EvaluationResult
    interview_questions: InterviewQuestions
    assessment: Optional[AssessmentData] = None
