import json
import requests
import random
from models import ATSResult, MCQQuestion, DSAQuestion, AssessmentData
from mcp_tools import call_tool

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 1024},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"[agent_4] Ollama call failed: {e}")

def _build_assessment_prompt(ats: ATSResult, context: str) -> str:
    # Preferred languages based on user request
    languages = ["Python", "C++"]
    target_lang = random.choice(languages)
    
    return f"""You are a technical examiner. Create a specialized assessment for a candidate.
Ground the questions in the candidate's resume and job requirements.

Context:
{context}

Matched Skills: {', '.join([m.requirement for m in ats.matching_skills[:3]])}
Gaps: {', '.join([m.requirement for m in ats.missing_skills[:3]])}

Strictly return ONLY valid JSON in this format:
{{
  "mcqs": [
    {{
      "id": 1,
      "question": "A technical/behavioral question about [Context Item]",
      "options": ["Opt A", "Opt B", "Opt C", "Opt D"],
      "correct_idx": 0,
      "explanation": "Why A is correct"
    }}
  ],
  "dsa": {{
    "title": "A coding challenge related to the role",
    "description": "Problem statement for {target_lang}",
    "constraints": ["Time complexity requirement"],
    "base_code": "The initial {target_lang} function signature",
    "language": "{target_lang.lower()}",
    "solution_logic": "Pseudocode explanation of the optimal approach"
  }}
}}

Generate exactly 3 MCQs (2 technical, 1 behavioral) and 1 DSA coding question.
"""

def generate_assessment(ats_result: ATSResult) -> AssessmentData:
    # Pull resume context
    chunks = call_tool("query_resume", query="technical skills projects algorithms", n_results=4)
    context = "\n".join([c["content"] for c in chunks])
    
    prompt = _build_assessment_prompt(ats_result, context)
    print("[agent_4] Generating MCQ & DSA Assessment...")
    raw = _ollama_generate(prompt)
    
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        data = json.loads(clean[start:end])
        
        mcqs = [MCQQuestion(**q) for q in data.get("mcqs", [])]
        dsa = DSAQuestion(**data.get("dsa", {}))
        
        return AssessmentData(mcqs=mcqs, dsa=dsa)
    except Exception as e:
        print(f"[agent_4] Parsing failed: {e}. Using fallback assessment.")
        return AssessmentData(
            mcqs=[
                MCQQuestion(id=1, question="What is the primary benefit of using a Vector Database in RAG?", 
                            options=["Encryption", "Semantic Search", "Compression", "Formatting"], 
                            correct_idx=1, explanation="Vector DBs allow high-speed similarity search based on embeddings."),
                MCQQuestion(id=2, question="Which data structure is most efficient for a LIFO (Last-In-First-Out) implementation?", 
                            options=["Queue", "Linked List", "Stack", "Heap"], 
                            correct_idx=2, explanation="A Stack is a LIFO structure."),
                MCQQuestion(id=3, question="How do you handle a conflict in a team setting?", 
                            options=["Avoid it", "Listen and find common ground", "Report to HR immediately", "Force your opinion"], 
                            correct_idx=1, explanation="Collaboration requires empathy and listening.")
            ],
            dsa=DSAQuestion(
                title="Reverse a String (In-place)",
                description="Write a function that reverses a string in-place without using extra memory.",
                constraints=["O(1) extra space", "O(N) time complexity"],
                base_code="def reverse_string(s: list[str]) -> None:",
                language="python",
                solution_logic="Use two-pointer technique swapping elements from start and end."
            )
        )
