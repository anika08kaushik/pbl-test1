"""
Agent 1 — Interview Question Generator
Generates technical, behavioral, and scenario-based questions
grounded in the candidate's actual resume and JD gaps.
Uses MCP tools to pull relevant resume context dynamically.
"""

import json
import requests

from models import ATSResult, EvaluationResult, InterviewQuestions
from mcp_tools import call_tool

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"


def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,  # 9 questions don't need more than this
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if resp.status_code == 404:
            raise RuntimeError(f"[agent_1] Model '{MODEL}' not found. Run: ollama pull {MODEL}")
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("[agent_1] Ollama is not running. Start it with: ollama serve")


def _build_prompt(
    ats: ATSResult,
    eval_result: EvaluationResult,
    skills_context: str,
    exp_context: str,
) -> str:
    # Trimmed to top items only — smaller prompt = faster inference
    matched = [m.requirement for m in ats.matching_skills[:4]]
    missing = [m.requirement for m in ats.missing_skills[:3]]
    partial = [m.requirement for m in ats.partial_matches[:2]]

    return f"""You are a senior technical interviewer preparing questions for a specific candidate.

ATS Score: {ats.ats_score}% | Fit: {eval_result.overall_fit}
Strengths: {', '.join(eval_result.strengths[:3]) or 'None'}
Gaps: {', '.join(eval_result.gaps[:3]) or 'None'}
Matched: {', '.join(matched) or 'None'}
Partial: {', '.join(partial) or 'None'}
Missing: {', '.join(missing) or 'None'}

Skills: {skills_context}
Experience: {exp_context}

Return ONLY valid JSON, no extra text, directly parseable by json.loads().
Focus questions on missing skills and gaps. No generic questions.

{{
  "technical": ["question 1", "question 2", "question 3"],
  "behavioral": ["question 1", "question 2", "question 3"],
  "scenario_based": ["question 1", "question 2", "question 3"]
}}"""


def run(ats_result: ATSResult, eval_result: EvaluationResult) -> InterviewQuestions:
    # Section-filtered retrieval
    skills_chunks = call_tool(
        "query_resume",
        query="technical skills programming languages frameworks tools",
        section="skills",
        n_results=3,
    )
    exp_chunks = call_tool(
        "query_resume",
        query="work experience projects responsibilities achievements built developed",
        section="experience",
        n_results=3,
    )

    # Fallback if section filter returns nothing
    if not skills_chunks:
        skills_chunks = call_tool("query_resume", query="technical skills programming", n_results=3)
    if not exp_chunks:
        exp_chunks = call_tool("query_resume", query="work experience projects", n_results=3)

    skills_context = "\n".join([c["content"] for c in skills_chunks]) or "Not found"
    exp_context = "\n".join([c["content"] for c in exp_chunks]) or "Not found"

    prompt = _build_prompt(ats_result, eval_result, skills_context, exp_context)

    print("[agent_1] Generating interview questions via Gemma 4B...")
    raw = _ollama_generate(prompt)

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

        data = json.loads(clean)

        technical = data.get("technical", [])
        behavioral = data.get("behavioral", [])
        scenario_based = data.get("scenario_based", [])

        if not technical and not behavioral and not scenario_based:
            raise ValueError("All question lists empty")

        print(f"[agent_1] Generated {len(technical)} technical, "
              f"{len(behavioral)} behavioral, {len(scenario_based)} scenario questions")

        return InterviewQuestions(
            technical=technical,
            behavioral=behavioral,
            scenario_based=scenario_based,
        )

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[agent_1] Parse failed ({e}), using fallback")
        matched = [m.requirement for m in ats_result.matching_skills[:3]]
        missing = [m.requirement for m in ats_result.missing_skills[:3]]
        return InterviewQuestions(
            technical=[
                f"Walk me through your experience with {matched[0]}." if matched else "Describe your core technical skills.",
                f"How would you approach learning {missing[0]} for this role?" if missing else "What are you currently learning?",
                "Describe the most technically complex project you have worked on.",
            ],
            behavioral=[
                "Tell me about a time you delivered under a tight deadline.",
                "Describe a situation where you had to learn a new technology quickly.",
            ],
            scenario_based=[
                f"If this role required {missing[0]}, how would you get up to speed?" if missing else "How do you handle unfamiliar technical requirements?",
            ],
        )
