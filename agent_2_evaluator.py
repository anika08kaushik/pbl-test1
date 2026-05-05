"""
Agent 2 — Qualitative Evaluator
Uses Gemma 7B via Ollama for local, private, narrative evaluation.
Takes the ATSResult + resume context and produces human-readable feedback.
"""

import json
import requests
from typing import List

from models import ATSResult, EvaluationResult
from mcp_tools import call_tool

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma:7b"


def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 400,  # smaller budget for Gemma 7B to reduce timeout risk
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if resp.status_code == 404:
            raise RuntimeError(f"[agent_2] Model '{MODEL}' not found. Run: ollama pull {MODEL}")
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("[agent_2] Ollama is not running. Start it with: ollama serve")


def _build_prompt(ats: ATSResult, resume_context: List[dict]) -> str:
    matched = [m.requirement for m in ats.matching_skills[:4]]
    missing = [m.requirement for m in ats.missing_skills[:3]]
    partial = [m.requirement for m in ats.partial_matches[:2]]
    context_text = "\n---\n".join([r["content"] for r in resume_context[:4]])

    return f"""You are an expert technical recruiter evaluating a candidate.

ATS Score: {ats.ats_score}%
Matched: {', '.join(matched) or 'None'}
Partial: {', '.join(partial) or 'None'}
Missing: {', '.join(missing) or 'None'}

Resume context:
{context_text or 'Not available'}

Return ONLY valid JSON, no extra text, directly parseable by json.loads().

{{
  "qualitative_feedback": "2-3 sentence assessment referencing specific skills",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["gap 1", "gap 2", "gap 3"],
  "overall_fit": "strong"
}}

overall_fit must be exactly: "strong" (score>=65%), "moderate" (35-64%), or "weak" (<35%)"""


def run(ats_result: ATSResult) -> EvaluationResult:
    # Build query from both missing and matched skills for context
    missing_query = " ".join([m.requirement for m in ats_result.missing_skills[:3]])
    matched_query = " ".join([m.requirement for m in ats_result.matching_skills[:2]])
    query = f"{missing_query} {matched_query}".strip() or "experience skills projects"

    # Section-filtered retrieval
    skills_context = call_tool("query_resume", query=query, section="skills", n_results=2)
    exp_context = call_tool("query_resume", query=query, section="experience", n_results=2)
    resume_context = skills_context + exp_context

    if not resume_context:
        resume_context = call_tool("query_resume", query=query, n_results=4)

    prompt = _build_prompt(ats_result, resume_context)

    print("[agent_2] Calling Gemma 7B via Ollama...")
    raw_response = _ollama_generate(prompt)

    try:
        clean = raw_response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

        data = json.loads(clean)

        fit = data.get("overall_fit", "moderate").lower().strip()
        if fit not in ("strong", "moderate", "weak"):
            fit = "strong" if ats_result.ats_score >= 65 else (
                "moderate" if ats_result.ats_score >= 35 else "weak"
            )

        strengths = data.get("strengths", []) or [m.requirement for m in ats_result.matching_skills[:3]]
        gaps = data.get("gaps", []) or [m.requirement for m in ats_result.missing_skills[:3]]

        print(f"[agent_2] Done — fit: {fit}, strengths: {len(strengths)}, gaps: {len(gaps)}")

        return EvaluationResult(
            qualitative_feedback=data.get("qualitative_feedback", ""),
            strengths=strengths,
            gaps=gaps,
            overall_fit=fit,
        )

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[agent_2] Parse failed ({e}), using ATS-based fallback")
        fit = "strong" if ats_result.ats_score >= 65 else (
            "moderate" if ats_result.ats_score >= 35 else "weak"
        )
        return EvaluationResult(
            qualitative_feedback=f"ATS score: {ats_result.ats_score}%. "
                                  f"Matched {len(ats_result.matching_skills)} requirements, "
                                  f"missing {len(ats_result.missing_skills)}.",
            strengths=[m.requirement for m in ats_result.matching_skills[:3]],
            gaps=[m.requirement for m in ats_result.missing_skills[:3]],
            overall_fit=fit,
        )
