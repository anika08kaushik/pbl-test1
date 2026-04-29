"""
Agent 5 — Interview Evaluator
Generates qualitative feedback based on candidate's answers to the interview questions.
"""

import json
import requests
from typing import List

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava:7b"

def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if resp.status_code == 404:
            raise RuntimeError(f"[agent_5] Model '{MODEL}' not found. Run: ollama pull {MODEL}")
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("[agent_5] Ollama is not running. Start it with: ollama serve")

def _build_prompt(answers: List[dict]) -> str:
    context = ""
    for idx, item in enumerate(answers):
        context += f"Q{idx+1}: {item.get('question')}\nA{idx+1}: {item.get('answer')}\n\n"

    return f"""You are a technical interviewer evaluating a candidate's responses.

Candidate's Answers:
{context}

Based on these answers, provide constructive feedback on the candidate's performance. Return ONLY valid JSON, no extra text, directly parseable by json.loads().

{{
  "overall_impression": "1-2 sentence overall impression.",
  "strengths": ["strength 1", "strength 2"],
  "areas_for_improvement": ["area 1", "area 2"],
  "technical_accuracy": "A brief comment on technical accuracy if applicable."
}}"""

def run(answers: List[dict]) -> dict:
    if not answers:
        return {
            "overall_impression": "No answers provided.",
            "strengths": [],
            "areas_for_improvement": ["Please provide answers during the interview."],
            "technical_accuracy": "N/A"
        }

    prompt = _build_prompt(answers)
    print("[agent_5] Evaluating interview answers via Gemma 4B...")
    
    try:
        raw = _ollama_generate(prompt)
        clean = raw.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

        data = json.loads(clean)
        return data

    except Exception as e:
        print(f"[agent_5] Parse failed ({e}), using fallback")
        return {
            "overall_impression": "The candidate provided answers, but the detailed evaluation could not be completed.",
            "strengths": ["Communicated during the interview"],
            "areas_for_improvement": ["Need more detailed technical responses"],
            "technical_accuracy": "Evaluation unavailable."
        }
