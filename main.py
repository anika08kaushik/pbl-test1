"""
main.py — AI Resume Screener
"""

import re
import sys
import json
from pathlib import Path
from typing import List

from models import ScreenerOutput
from extractor import extract_resume
from vector_store import (
    store_resume_chunks,
    store_jd_requirements_tagged,
    clear_collections,
    parse_jd_requirements_from_text,
)
import agent_3_validator
import agent_2_evaluator
import agent_1_interviewer


# ── Noise: only hard structural signals, nothing aggressive ───────────────────

def _is_hard_noise(text: str) -> bool:
    """
    Only filter things that are DEFINITELY not requirements.
    Keep the filter loose — better to include borderline items than miss real ones.
    """
    t = text.strip()
    if not t or len(t) < 4:
        return True
    if "@" in t and "." in t:          # email
        return True
    if re.search(r"https?://|www\.", t, re.I):  # URL
        return True
    if sum(c.isdigit() for c in t) / len(t) > 0.5:  # mostly digits
        return True
    if sum(c.isalpha() for c in t) / len(t) < 0.35:  # mostly symbols
        return True
    return False


# # Lines that are pure JD metadata — skip entirely
# _SKIP_LINE_RE = re.compile(
#     r"^(good to have skills\s*:|must have skills\s*:|educational qualification\s*:"
#     r"|minimum \d+ year|additional information|this position is based"
#     r"|the candidate should have minimum"
#     r"|link\s*/\s*url|websites,?\s*portfolios|profiles"
#     r"|project role\s*:|project role description\s*:)",
#     re.I
# )


def _extract_jd_requirements(jd_text: str):
    return parse_jd_requirements_from_text(jd_text)

def _extract_resume_skill_items(chunks) -> List[str]:
    """
    Extract evidence items from resume skills/projects/experience sections.
    Skills section: 1–6 word items.
    Other sections: 2–8 word phrases.
    """
    items: List[str] = []
    seen:  set = set()

    for priority in ("skills", "skill_item", "projects", "experience", "certifications"):
        for chunk in chunks:
            if chunk.section != priority:
                continue

            raw = re.sub(r"[•·▪▸◦\*]\s*", "\n", chunk.content)
            raw = re.sub(r"[|;]\s*", "\n", raw)

            for line in raw.splitlines():
                line = re.sub(r"^[\-–—\d\.]+\s*", "", line).strip()
                line = re.sub(r"[\.;,]+$", "", line).strip()
                if not line:
                    continue

                wc = len(line.split())
                if priority in ("skills", "skill_item"):
                    if wc < 1 or wc > 6:
                        continue
                    if len(line) < 2 or "@" in line or re.search(r"https?://", line):
                        continue
                else:
                    if wc < 2 or wc > 8:
                        continue
                    if _is_hard_noise(line):
                        continue

                key = re.sub(r"\s+", " ", line.lower())
                if key not in seen:
                    seen.add(key)
                    items.append(line)

    return items


# ── Main pipeline ─────────────────────────────────────────────────────────────

def screen(resume_pdf_path: str, jd_text: str, jd_title: str = "Target Role") -> ScreenerOutput:
    print("\n" + "="*60)
    print("AI RESUME SCREENER")
    print("="*60)

    clear_collections()

    print("\n[1/6] Extracting resume...")
    chunks = extract_resume(resume_pdf_path)
    print(f"      Sections: {list(dict.fromkeys(c.section for c in chunks))}")

    print("\n[2/6] Storing resume chunks...")
    store_resume_chunks(chunks)

    print("\n[3/6] Extracting requirements + evidence...")
    jd_items     = _extract_jd_requirements(jd_text)
    resume_items = _extract_resume_skill_items(chunks)

    print(f"\n      JD requirements ({len(jd_items)}):")
    for it in jd_items:
        print(f"        · {it}")
    print(f"\n      Resume evidence ({len(resume_items)}):")
    for it in resume_items[:12]:
        print(f"        · {it}")

    store_jd_requirements_tagged(
        jd_items=jd_items,
        resume_items=resume_items,
        title=jd_title,
    )

    print("\n[4/6] Agent 3: CrossEncoder judgment per requirement...")
    ats_result = agent_3_validator.run(verbose=True)

    print("\n[5/6] Agent 2: Qualitative evaluation...")
    eval_result = agent_2_evaluator.run(ats_result)

    print("\n[6/6] Agent 1: Interview questions...")
    questions = agent_1_interviewer.run(ats_result, eval_result)

    output = ScreenerOutput(
        ats_result=ats_result,
        evaluation=eval_result,
        interview_questions=questions,
    )
    _print_results(output)
    return output


def _print_results(output: ScreenerOutput) -> None:
    ats = output.ats_result
    ev  = output.evaluation
    print("\n" + "="*60)
    print(f"  ATS SCORE: {ats.ats_score}%")
    print("="*60)
    print(f"\n✅ MATCHED ({len(ats.matching_skills)})")
    for m in ats.matching_skills:
        print(f"   [{m.similarity_score:.3f}] {m.requirement}")
    print(f"\n⚠  PARTIAL ({len(ats.partial_matches)})")
    for m in ats.partial_matches:
        print(f"   [{m.similarity_score:.3f}] {m.requirement}")
    print(f"\n❌ MISSING ({len(ats.missing_skills)})")
    for m in ats.missing_skills[:15]:
        print(f"   [{m.similarity_score:.3f}] {m.requirement}")
    print(f"\n📋 Fit: {ev.overall_fit.upper()} — {ev.qualitative_feedback}\n")


extract_jd_skills    = _extract_jd_requirements
extract_jd_items     = _extract_jd_requirements
extract_resume_items = _extract_resume_skill_items


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <resume.pdf> <jd.txt> [Job Title]")
        sys.exit(1)
    jd_text = Path(sys.argv[2]).read_text(encoding="utf-8")
    result  = screen(sys.argv[1], jd_text, sys.argv[3] if len(sys.argv) > 3 else "Target Role")
    with open("screener_output.json", "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    print("Saved → screener_output.json")
