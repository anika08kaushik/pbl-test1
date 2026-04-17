import re
import numpy as np
from sentence_transformers import CrossEncoder

from models import ATSResult, MatchedSkill, MissingSkill
from vector_store import get_jd_only_requirements, query_resume_top_k

_cross_encoder = None


def _get_model():
    global _cross_encoder
    if _cross_encoder is None:
        print("[agent_3] Loading CrossEncoder model...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ── Thresholds ────────────────────────────────────────────────────────────────
MATCH_THRESHOLD   = 0.60
PARTIAL_THRESHOLD = 0.38
MISSING_THRESHOLD = 0.20

SKILL_CONCEPT_MAP = {
    "relational databases": [
        "mysql", "postgresql", "sql server", "sqlite",
        "mariadb", "oracle", "sql"
    ],
    "nosql databases": [
        "mongodb", "firebase", "cassandra", "dynamodb",
        "redis", "couchbase"
    ],
    "frontend development": [
        "react", "vue", "angular", "html", "css",
        "javascript", "typescript", "svelte"
    ],
    "backend development": [
        "node.js", "express", "django", "spring", "flask",
        "fastapi", "java", "python", "ruby on rails", "php", "go"
    ],
    "cloud": ["aws", "azure", "gcp", "amazon web services", "cloud platforms"],
    "javascript": ["react", "reactjs", "angular", "vue", "typescript", "node.js"],
    "mongodb": ["nosql databases", "nosql", "document database"],
    "aws": ["cloud", "amazon web services"],
    "azure": ["cloud"],
    "gcp": ["cloud"],
    "ci/cd": ["cicd", "jenkins", "github actions", "gitlab ci", "circleci", "travis", "azure devops"],
    "node.js": ["nodejs", "node js", "node"],
    "react.js": ["react", "reactjs", "react js"],
    "vue.js": ["vue", "vuejs", "vue js"],
    "express.js": ["express", "expressjs", "express js"],
    "rest api": ["rest apis", "restful api", "restful apis"],
    "git": ["github", "gitlab", "bitbucket"],
}

REQ_LEAD_IN = re.compile(
    r"^(?:must\s+have\s+|should\s+have\s+|strong\s+experience\s+in\s+|experience\s+with\s+|"
    r"proficiency\s+in\s+|knowledge\s+of\s+|familiarity\s+with\s+|expertise\s+in\s+|"
    r"hands\.on\s+experience\s+(?:with|in)\s+|working\s+knowledge\s+of\s+|"
    r"understanding\s+of\s+|ability\s+to\s*)",
    re.I,
)


def _normalize_requirement(requirement: str) -> str:
    req = requirement.lower().strip()
    req = REQ_LEAD_IN.sub("", req)
    req = re.sub(r"[^a-z0-9\s.+#/-]", " ", req)
    req = re.sub(r"\s+", " ", req).strip()
    return req


def _concept_terms(requirement: str) -> list[str]:
    requirement = _normalize_requirement(requirement)
    terms = [requirement]
    if requirement in SKILL_CONCEPT_MAP:
        terms.extend(SKILL_CONCEPT_MAP[requirement])
    return terms


# ── Keyword boost (BOUNDARY SAFE) ─────────────────────────────────────────────
def _keyword_boost(requirement: str, chunk_content: str) -> float:
    req_lower = _normalize_requirement(requirement)
    chunk_lower = chunk_content.lower()

    # direct phrase match (boundary safe)
    for term in _concept_terms(req_lower):
        escaped_term = re.escape(term)
        if re.search(rf"(?<![a-z0-9]){escaped_term}(?![a-z0-9])", chunk_lower):
            return 0.72

    tokens = [t for t in re.split(r"[\s\-]+", req_lower) if t and len(t) >= 2]

    if not tokens:
        return 0.0

    def _tok_in_text(tok: str, text: str) -> bool:
        escaped = re.escape(tok)

        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text):
            return True

        # plural/singular handling
        if tok.endswith("s") and len(tok) > 3:
            stem = re.escape(tok[:-1])
            if re.search(rf"(?<![a-z0-9]){stem}(?![a-z0-9])", text):
                return True

        if not tok.endswith("s"):
            plural = re.escape(tok + "s")
            if re.search(rf"(?<![a-z0-9]){plural}(?![a-z0-9])", text):
                return True

        return False

    matches = sum(1 for tok in tokens if _tok_in_text(tok, chunk_lower))
    ratio = matches / len(tokens)

    # Concept synonyms can also improve partial coverage. If a concept term matches
    # via tokens, treat it as a valid boost source.
    if not matches:
        for alt in _concept_terms(req_lower):
            if alt == req_lower:
                continue
            alt_tokens = [t for t in re.split(r"[\s\-]+", alt) if t and len(t) >= 2]
            alt_matches = sum(1 for tok in alt_tokens if _tok_in_text(tok, chunk_lower))
            if alt_matches and alt_matches / len(alt_tokens) >= 0.67:
                return 0.65
            if alt_matches and len(alt_tokens) == 1:
                return 0.68

    if len(tokens) == 1:
        return 0.68 if matches else 0.0

    if ratio >= 0.67:
        return 0.65
    elif ratio >= 0.34:
        return 0.40

    return 0.0


# ── REAL SKILL DETECTOR (FIXED) ──────────────────────────────────────────────
def _is_likely_real_skill(req: str) -> bool:
    req = req.strip().lower()
    if not req:
        return False

    words = req.split()

    # 🔴 FINAL FIX: remove sentence-like phrases
    if len(words) > 4:
        return False

    # reject structure words
    BLOCK = {
        "job","role","type","position","location",
        "remote","apply","submit","form",
        "experience","working","responsibilities",
        "requirements","ability","comfort"
    }
    if any(w in BLOCK for w in words):
        return False

    # reject verb phrases
    VERBS = {
        "develop","build","work","collaborate",
        "participate","learn","identify","support"
    }
    if any(w in VERBS for w in words):
        return False

    # accept tech signals
    if any(char in req for char in ".#++"):
        return True

    if len(req) <= 8:
        return True

    if 1 <= len(words) <= 3:
        return True

    return False


# ── MAIN RUN FUNCTION ────────────────────────────────────────────────────────
def run(jd_text=None, verbose=True) -> ATSResult:
    try:
        model = _get_model()
        jd_requirements = get_jd_only_requirements(jd_text)

        matching = []
        partial = []
        missing = []
        weighted_points = 0.0

        # collect full resume text
        all_chunks = query_resume_top_k("skills experience projects", k=50)
        all_resume_text = " ".join(c["content"].lower() for c in all_chunks)

        for req in jd_requirements:

            # skip garbage
            if not _is_likely_real_skill(req):
                continue

            chunks = query_resume_top_k(req, k=10)

            best_score = 0
            best_chunk = ""

            for chunk in chunks:
                semantic = _sigmoid(model.predict([(req, chunk["content"])])[0])
                boost = _keyword_boost(req, chunk["content"])
                score = max(semantic, boost)

                if score > best_score:
                    best_score = score
                    best_chunk = chunk["content"]

            # classification, using the best available score from chunk-level
            # prediction or resume-wide concept matching.
            global_boost = _keyword_boost(req, all_resume_text)
            final_score = max(best_score, global_boost)
            # Step 2: HUMAN-LIKE BOOST
            # If mapping helped with a transferable skill, boost it slightly.
            if global_boost > 0:
                final_score += 0.10
            # If semantic match exists but is weak, give a small understanding boost.
            if 0.20 < best_score < 0.50:
                final_score += 0.05
            final_score = min(final_score, 1.0)

            if final_score >= MATCH_THRESHOLD:
                if best_score >= MATCH_THRESHOLD:
                    resume_excerpt = best_chunk[:200]
                    similarity_score = round(best_score, 3)
                else:
                    resume_excerpt = "(global match)"
                    similarity_score = round(global_boost, 3)

                matching.append(
                    MatchedSkill(
                        requirement=req,
                        resume_excerpt=resume_excerpt,
                        similarity_score=similarity_score
                    )
                )
                weighted_points += 1.0

            elif final_score >= PARTIAL_THRESHOLD:
                if best_score >= PARTIAL_THRESHOLD:
                    resume_excerpt = best_chunk[:200]
                    similarity_score = round(best_score, 3)
                else:
                    resume_excerpt = "(global partial)"
                    similarity_score = round(global_boost, 3)

                partial.append(
                    MatchedSkill(
                        requirement=req,
                        resume_excerpt=resume_excerpt,
                        similarity_score=similarity_score
                    )
                )
                weighted_points += 0.85

            else:
                missing.append(
                    MissingSkill(
                        requirement=req,
                        similarity_score=round(final_score, 3)
                    )
                )

        total = len(jd_requirements)
        score = (weighted_points / total) * 100 if total else 0

        return ATSResult(
            ats_score=round(score, 1),
            matching_skills=matching,
            partial_matches=partial,
            missing_skills=missing,
            summary=f"Score: {round(score,1)}%"
        )

    except Exception as e:
        print("[agent_3] ERROR:", e)
        return ATSResult(
            ats_score=0,
            matching_skills=[],
            partial_matches=[],
            missing_skills=[],
            summary="Error"
        )