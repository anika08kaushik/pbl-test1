"""
main.py — AI Resume Screener
"""

import re
import sys
import json
from pathlib import Path
from typing import List

from models import ScreenerOutput
from extractor import extract_resume, extract_skills
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


def _normalize_jd_text(jd_text: str) -> str:
    """
    Pre-process JD text before sending to Gemma.
    Expands contractions so apostrophe-truncation can't produce fragment tokens.
    e.g. "you'll" → "you will",  "bachelor's" → "bachelor",  "we're" → "we are"
    """
    # Order matters: longer patterns first
    contractions = [
        (r"\byou'll\b", "you will"),
        (r"\bwe'll\b", "we will"),
        (r"\bthey'll\b", "they will"),
        (r"\bit'll\b", "it will"),
        (r"\bwe're\b", "we are"),
        (r"\byou're\b", "you are"),
        (r"\bthey're\b", "they are"),
        (r"\bwe've\b", "we have"),
        (r"\byou've\b", "you have"),
        (r"\bwe'd\b", "we would"),
        (r"\byou'd\b", "you would"),
        (r"\bwon't\b", "will not"),
        (r"\bdon't\b", "do not"),
        (r"\bcan't\b", "cannot"),
        (r"\bisn't\b", "is not"),
        (r"\baren't\b", "are not"),
        (r"\bwasn't\b", "was not"),
        (r"\bweren't\b", "were not"),
        (r"\bhasn't\b", "has not"),
        (r"\bhaven't\b", "have not"),
        (r"\bdidn't\b", "did not"),
        (r"\bdoesn't\b", "does not"),
        # Possessives that create fragments: "bachelor's" → "bachelor"
        (r"\bbachelor's\b", "bachelor"),
        (r"\bmaster's\b", "master"),
        (r"\bcandidate's\b", "candidate"),
        (r"\bcompany's\b", "company"),
        # Generic possessive apostrophe-s — strip 's suffix
        (r"(\w{3,})'s\b", r"\1"),
    ]
    text = jd_text
    for pattern, replacement in contractions:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


# Contraction tail tokens that can NEVER be the start of a real skill
_CONTRACTION_TAILS = {
    "ll", "re", "ve", "d", "s",        # you'll → ll, we're → re, etc.
    "m",                                # I'm → m
}

# Words that indicate this is a duty/responsibility sentence, not a skill
_DUTY_STARTERS = {
    "write", "develop", "build", "create", "conduct", "collaborate", "participate",
    "deliver", "ensure", "maintain", "support", "provide", "manage",
    "tackle", "work", "join", "looking", "open", "passionate",
}


# ── Abbreviation normalisation: expand common shorthand before dedup ──────────
_SKILL_NORMALIZE = {
    "js":           "javascript",
    "ts":           "typescript",
    "nodejs":       "node.js",
    "node js":      "node.js",
    "vue.js":       "vue",
    "angular.js":   "angular",
    "reactjs":      "react",
    "react.js":     "react",
    "es6":          "javascript",
    "es2015":       "javascript",
    "expressjs":    "express",
    "express.js":   "express",
}

# Truncation suffixes Gemma produces when the token budget runs out mid-string
_TRUNCATED_SUFFIXES = (
    "w", "ks", "netw", "ork", "rk", "nd", "nce", "ment",
    "tion", "ing", "ed", "er", "ly", "al", "ful", "ous",
)

# Soft/non-technical words that can NEVER be a skill on their own or as a prefix
_SOFT_WORDS = {
    "good", "spoken", "written", "team", "collaborate", "experts",
    "interest", "passion", "eager", "willing", "ability", "understanding",
    "knowledge", "experience", "exposure", "familiarity", "competency",
    "professional", "contractual", "leading", "global", "worldwide",
    "innovative", "cutting-edge", "dynamic",
}


def _final_skill_filter(skills: List[str]) -> List[str]:
    """
    Hard gate applied BEFORE embedding validation.
    Drops:
    - Truncated tokens (cut off mid-word by Gemma's token budget)
    - Soft/non-technical phrases
    - Items containing only single-char words
    - Anything that looks like a sentence fragment
    Also normalises common abbreviations.
    """
    # These last-words of multi-word items signal a sentence fragment, not a skill
    PHRASE_NOISE_WORDS = {
        "structured", "architecture", "applications", "application",
        "development", "spoken", "written", "experts", "network",
        "worldwide", "code", "security", "scalable", "documented",
        "clean", "well-structured", "planning",
    }

    # First-word fragment signals (Gemma cuts mid-token → leftover token start)
    BAD_STARTS = {"ks", "rks", "nce", "ork", "ing"}

    cleaned = []
    for raw in skills:
        s = raw.lower().strip()

        # 1. Normalise abbreviations first
        s = _SKILL_NORMALIZE.get(s, s)

        words = s.split()
        if not words:
            continue

        # 2. Too short to be meaningful
        if len(s) < 2:
            continue

        first = words[0]

        # 3. First word is a known truncation-fragment token
        if first in BAD_STARTS:
            continue

        # 4. Any word in the skill is a soft/non-technical keyword
        if any(w in _SOFT_WORDS for w in words):
            continue

        # 5. Phrase-length guard (3 words max for skills)
        if len(words) > 3:
            continue

        # 6. Starts with a preposition / article — definitely a sentence fragment
        FRAGMENT_STARTERS = {"a", "an", "the", "in", "on", "at", "to", "for",
                              "of", "with", "by", "as", "like", "from"}
        if first in FRAGMENT_STARTERS:
            continue

        # 7. Multi-word items whose last word is a generic/noise word
        #    e.g. "clean architecture", "modular applications", "deliver structured"
        if len(words) >= 2 and words[-1] in PHRASE_NOISE_WORDS:
            continue

        # 8. Truncated suffix on last word — last word looks like a mid-token cut
        last = words[-1]
        GOOD_ENDINGS = {
            "js", "ts", "db", "ai", "ml", "go", "r", "c",
            "sql", "css", "git", "aws", "gcp", "api", "oop",
            "rest", "json", "html", "java", "ruby", "rust",
            "vue", "php", "elk", "eks", "rds", "ecs", "sns",
            "sqs", "iam", "ec2", "s3", "nlp", "cv", "py",
        }
        if 2 <= len(last) <= 7 and re.match(r"^[a-z]+$", last) and last not in GOOD_ENDINGS:
            if any(last.startswith(bad[:3]) for bad in (
                "framew", "netw", "archit", "applic", "platf",
            )):
                continue

        # 9. Contains a single-character word that isn't a known skill token
        single_char = [w for w in words if len(w) == 1 and w not in {"c", "r"}]
        if single_char:
            continue

        cleaned.append(s)

    return cleaned


def _ollama_extract_skills(jd_text: str) -> List[str]:
    """
    Stage 1: Ask Gemma (local Ollama) to extract candidate skills from JD.
    Returns a raw list — not yet validated.
    Normalizes JD text first to prevent apostrophe-fragment tokens.
    """
    import requests as _req
    OLLAMA_URL = "http://localhost:11434/api/generate"

    # ── CRITICAL: expand contractions BEFORE sending to Gemma ──────────────
    clean_jd = _normalize_jd_text(jd_text)

    prompt = (
        "Extract ONLY technical skills from this job description.\n\n"
        "Rules:\n"
        "- Only tools, technologies, frameworks, programming languages\n"
        "- No sentences, no soft skills, no education requirements\n"
        "- No numbers like '30 hours' or '20 hours'\n"
        "- No degree requirements (bachelor, master, etc.)\n"
        "- No verb phrases (write, develop, build, collaborate, etc.)\n"
        "- Max 3 words per skill\n"
        "- Return a JSON array of strings ONLY, no explanation\n\n"
        f"Job Description:\n{clean_jd[:2500]}\n\n"
        "Return ONLY valid JSON array like: [\"javascript\", \"node.js\", \"react\"]"
    )
    try:
        resp = _req.post(
            OLLAMA_URL,
            json={"model": "gemma:7b", "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 600}},  # raised: 300→600
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        raw = raw.replace("```json", "").replace("```", "")

        # ── Truncation-safe JSON extraction ──────────────────────────────
        # If the token budget cut the response mid-array, we still want the
        # items that completed properly. Strategy: find the last COMPLETE
        # quoted string before any truncation, then parse only that prefix.
        start = raw.find("[")
        if start == -1:
            return []

        # Try the full array first
        end = raw.rfind("]") + 1
        raw_array = raw[start:end] if end > start else ""

        candidates = []
        if raw_array:
            try:
                candidates = json.loads(raw_array)
            except json.JSONDecodeError:
                # Array was cut mid-way — extract all complete quoted strings
                candidates = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', raw[start:])

        return [str(s).strip().lower() for s in candidates if isinstance(s, str) and s.strip()]
    except Exception as e:
        print(f"[main] Ollama skill extraction failed: {e}")
        return []


def _embedding_validate_skills(candidates: List[str]) -> List[str]:
    """
    Stage 2: Balanced structural filter using word count + pattern heuristics.
    Accepts 1-3 word technical skills, rejects duty sentences and non-skills.
    """
    VALID_SHORT = {
        "js", "ts", "api", "sql", "html", "css", "aws", "gcp",
        "nlp", "llm", "llms", "db", "ci", "cd", "rest", "json",
    }
    REJECT_STARTERS = {
        "conduct", "build", "design", "develop", "work",
        "collaborate", "implement", "create", "write",
        "deliver", "ensure", "maintain", "support",
    }

    valid = []
    for skill in candidates:
        s = skill.lower().strip()
        if not s:
            continue

        words = s.split()
        if not words:
            continue

        first = words[0]

        # ── Hard structural rejects ──────────────────────────────────────
        if re.match(r"^\d+$", s):
            continue
        if first in _CONTRACTION_TAILS:
            continue
        if first in REJECT_STARTERS:
            continue
        if len(words) > 4:
            continue
        if re.search(r"\d+\s*(hours?|hrs?|days?|weeks?|months?|years?)", s):
            continue
        if re.search(r"\w\.\s+\w", s):
            continue

        # ── Allow valid skills ───────────────────────────────────────────
        # Allow known short skill tokens
        if s in VALID_SHORT:
            valid.append(skill)
            continue

        # Allow any 1-3 word skill that looks technical (alphabetic)
        if 1 <= len(words) <= 3:
            # Must have alphabetic chars and not be pure common words
            if any(ch.isalpha() for ch in s):
                valid.append(skill)
                continue

    return valid


def _deduplicate_skills(skills: List[str]) -> List[str]:
    """
    Stage 3: Remove near-duplicates using embedding cosine similarity.
    Keeps the first occurrence when two skills are > 0.85 similar.
    """
    from embedder import embed_single, cosine_similarity
    unique: List[str] = []
    unique_embs: list = []
    for skill in skills:
        try:
            emb = embed_single(skill)
        except Exception:
            unique.append(skill)
            continue
        is_dup = False
        for u_emb in unique_embs:
            try:
                if cosine_similarity(emb, u_emb) > 0.85:
                    is_dup = True
                    break
            except Exception:
                pass
        if not is_dup:
            unique.append(skill)
            unique_embs.append(emb)
    return unique


def _extract_jd_requirements(jd_text: str) -> List[str]:
    """
    Dynamic 3-stage JD skill extraction pipeline:
      1. Gemma 7B (Ollama) → candidate skill list
      2. Embedding validation → drop noise / non-skills semantically
      3. Embedding deduplication → remove near-duplicates

    Falls back to Anthropic API (if key set) then regex if Ollama is unavailable.
    """
    import os

    # ── Stage 1: LLM extraction ───────────────────────────────────────────────
    candidates = _ollama_extract_skills(jd_text)

    if not candidates:
        # Ollama unavailable — try Anthropic API
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import requests as _req
                payload = {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 600,
                    "system": (
                        "Extract ONLY concrete atomic technical skills from the job description. "
                        "Return a JSON array of strings. Rules: max 3 words per item, no verb phrases, "
                        "no soft skills, no education/degree words, no numbers or durations. "
                        "Split compounds: 'Python and Django' → ['python','django']. "
                        "Return ONLY the JSON array, no markdown."
                    ),
                    "messages": [{"role": "user", "content": f"Extract skills:\n\n{jd_text[:3000]}"}],
                }
                resp = _req.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json=payload, timeout=30,
                )
                resp.raise_for_status()
                raw = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","")
                candidates = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
                candidates = [str(s).strip().lower() for s in candidates
                              if isinstance(s, str) and s.strip()]
                print(f"[main] Anthropic API extracted {len(candidates)} candidate skills")
            except Exception as e:
                print(f"[main] Anthropic API extraction failed ({e}), using regex fallback")

    if not candidates:
        print("[main] Falling back to regex JD parser")
        return parse_jd_requirements_from_text(jd_text)

    print(f"[main] Stage 1 — raw candidates ({len(candidates)}): {candidates[:10]}{'...' if len(candidates)>10 else ''}")

    # ── Stage 1.5: Hard structural filter + normalisation ─────────────────────
    filtered = _final_skill_filter(candidates)
    print(f"[main] Stage 1.5 — after filter ({len(filtered)}): {filtered}")

    # ── Stage 2: Embedding validation ────────────────────────────────────────
    validated = _embedding_validate_skills(filtered)
    print(f"[main] Stage 2 — after validation ({len(validated)}): {validated[:10]}{'...' if len(validated)>10 else ''}")

    # ── Stage 3: Deduplication ────────────────────────────────────────────────
    final = _deduplicate_skills(validated)
    print(f"[main] Stage 3 — after dedup ({len(final)}): {final}")

    if not final:
        print("[main] Pipeline returned empty — falling back to regex")
        return parse_jd_requirements_from_text(jd_text)

    return final

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

    # Extract structured skills
    resume_skills = extract_skills(" ".join([c.content for c in chunks]))
    # ── KEY FIX: reuse jd_items (already cleaned by Ollama pipeline) ──────
    # Do NOT call extract_skills(jd_text) here — that's the regex path that
    # produces garbage like "ll do write clean", "30", "s degree in computer science".
    jd_skills = jd_items  # already validated + deduplicated
    # Store skills in ChromaDB

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
