import re
import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from extractor import normalize_text
from models import ATSResult, MatchedSkill, MissingSkill
from vector_store import get_jd_only_requirements, query_resume_top_k, get_skills
from embedder import embed_single, embed_texts

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
MATCH_THRESHOLD_STRICT   = 0.60   # Lower bar for better match acceptance (was 0.85)
MATCH_THRESHOLD_LOOSE    = 0.70   # Medium bar for general skills
MATCH_THRESHOLD          = 0.50   # Default alias used by app.py sidebar slider (lowered from 0.65)
PARTIAL_THRESHOLD        = 0.38   # Partial: semantic similarity acceptable (was 0.55)
MISSING_THRESHOLD        = 0.20   # Missing: below this

WEIGHTED_SKILLS = {
    "python": 2.0,
    "react": 2.0,
    "node.js": 2.0,
    "tensorflow": 2.0,
    "pytorch": 2.0,
    "scikit-learn": 1.8,
    "typescript": 1.8,
    "graphql": 1.8,
    "postgresql": 1.8,
    "redis": 1.7,
    "aws": 1.8,
    "docker": 1.5,
    "kubernetes": 1.5,
    "sql": 1.4,
    "rest api": 1.5,
    "terraform": 1.6,
    "ci/cd": 1.5,
    "express": 1.4,
    "mongodb": 1.4,
    "git": 1.2,
}
DEFAULT_SKILL_WEIGHT = 1.0

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

# SKILL NORMALIZATION (FIX 1)
NORMALIZE = {
    "llm": "large language models",
    "llms": "large language models",
    "nlp": "natural language processing",
    "js": "javascript",
    "nodejs": "node.js",
    "reactjs": "react",
    "vuejs": "vue",
    "expressjs": "express",
    "scikit learn": "scikit-learn",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "genai": "generative ai",
    "cicd": "ci/cd",
    "nosql": "nosql databases",
    "sql": "relational databases",
    "db": "database",
    "dbms": "database management system",
    "api": "rest api",
    "apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "bi": "business intelligence",
    "etl": "extract transform load",
    "devops": "devops tools",
}

def normalize_skill(skill: str) -> str:
    """Normalize skill abbreviations to full terms."""
    return NORMALIZE.get(skill.lower(), skill.lower())

REQ_LEAD_IN = re.compile(
    r"^(?:must\s+have\s+|should\s+have\s+|strong\s+experience\s+in\s+|experience\s+with\s+|"
    r"proficiency\s+in\s+|knowledge\s+of\s+|familiarity\s+with\s+|expertise\s+in\s+|"
    r"hands\.on\s+experience\s+(?:with|in)\s+|working\s+knowledge\s+of\s+|"
    r"understanding\s+of\s+|ability\s+to\s*)",
    re.I,
)


# ── JD SKILL CLEANING (FIX 1) ────────────────────────────────────────────────────────
def clean_jd_skill(skill: str) -> str:
    """
    Extract actual tech keywords from dirty JD skill phrases.
    Examples:
      "frameworks like node.js" -> "node.js"
      "interest in building scalable" -> None (not a real skill)
      "typescript" -> "typescript"
    Uses word boundary matching to avoid false positives like "scala" in "scalable".
    """
    skill = skill.lower().strip()
    
    # List of known tech keywords to extract (ordered: longer first to avoid partial matches)
    tech_keywords = [
        # Frontend
        "react", "vue", "angular", "svelte", "next.js", "gatsby", "html", "css",
        # Backend & Runtime
        "node.js", "nodejs", "python", "java", "rust", "php", "ruby",
        # Languages
        "javascript", "typescript", "c++", "c#", "kotlin", "scala", "swift",
        # Databases
        "mongodb", "postgresql", "mysql", "redis", "cassandra", "dynamodb", "elasticsearch",
        # Frameworks
        "express", "django", "flask", "spring", "fastapi", "nestjs", "nest.js",
        # Libraries
        "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "keras",
        # DevOps & Cloud
        "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "jenkins",
        # Testing
        "jest", "pytest", "mocha", "jasmine", "selenium",
        # AI/ML/Data
        "nlp", "llm", "llms", "machine learning", "deep learning",
        "generative ai", "prompt engineering", "vector data", "embeddings",
        # Other
        "git", "github", "gitlab", "sql", "graphql", "rest api", "ci/cd", "go",
    ]
    
    # If skill is already a known tech keyword, return it directly
    if skill in tech_keywords:
        return skill
    
    # Try to extract any tech keyword from the skill phrase with word boundaries
    for tech in sorted(tech_keywords, key=len, reverse=True):  # Longer keywords first
        # Use word boundary regex to avoid false positives
        pattern = rf"(?:^|\s|[\-.,]){re.escape(tech)}(?:$|\s|[\-.,])"
        if re.search(pattern, skill):
            return tech
    
    # If no tech keyword found, return None (skill is not a real tech skill)
    return None


# ── DIRECT MATCH BOOST (FIX 3) ──────────────────────────────────────────────────────
def is_direct_match(jd_skill: str, resume_skill: str) -> bool:
    """
    Check if JD skill directly matches or is substring of resume skill.
    Handles variations like "node.js" vs "nodejs", "react" vs "react.js".
    Returns True if there's a direct keyword match.
    """
    jd_lower = jd_skill.lower().strip()
    resume_lower = resume_skill.lower().strip()
    
    # Exact match
    if jd_lower == resume_lower:
        return True
    
    # Substring match (either direction)
    if jd_lower in resume_lower or resume_lower in jd_lower:
        return True
    
    # Normalize variations: remove dots and hyphens for comparison
    jd_normalized = jd_lower.replace(".", "").replace("-", "")
    resume_normalized = resume_lower.replace(".", "").replace("-", "")
    
    if jd_normalized and resume_normalized:
        if jd_normalized == resume_normalized:
            return True
        if jd_normalized in resume_normalized or resume_normalized in jd_normalized:
            return True
    
    return False


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
    if not req or len(req) < 2:
        return False

    words = req.split()

    # pure numbers (e.g. "30", "20", "40")
    if re.match(r"^\d+$", req):
        return False

    # number + unit phrases: "30 hours", "20 hours per week", "averaging 20"
    if re.search(r"\d+\s*(hours?|hrs?|days?|weeks?|months?|years?)", req):
        return False
    if re.match(r"^(averaging|minimum|at\s+least|up\s+to)\s+\d+", req):
        return False

    # sentence-like — too long to be a skill name
    if len(words) > 4:
        return False

    # reject structure/meta words
    BLOCK = {
        "job", "role", "type", "position", "location",
        "remote", "apply", "submit", "form",
        "experience", "working", "responsibilities",
        "requirements", "ability", "comfort",
        "bachelor", "master", "degree", "engineering",
        "averaging", "duration", "contract", "start",
        "communication", "written", "spoken", "english",
    }
    if any(w in BLOCK for w in words):
        return False

    # reject verb-lead phrases (duty sentences)
    VERBS = {
        "develop", "build", "work", "collaborate",
        "participate", "learn", "identify", "support",
        "write", "deliver", "ensure", "maintain",
        "looking", "open", "re", "ll", "we",
    }
    if words and words[0] in VERBS:
        return False

    # accept tech signals
    if any(char in req for char in ".#++"):
        return True

    if len(req) <= 8:
        return True

    if 1 <= len(words) <= 3:
        return True

    return False


# ── SIMILARITY-BASED SKILL MATCHING ───────────────────────────────────────────
_skill_embedding_cache = {}  # Cache embeddings to avoid recomputation


def _get_cached_embedding(skill: str):
    """Get embedding for a skill, using cache to avoid recomputation."""
    if skill not in _skill_embedding_cache:
        try:
            _skill_embedding_cache[skill] = embed_single(skill)
        except Exception as e:
            print(f"[agent_3] Embedding error for '{skill}': {e}")
            _skill_embedding_cache[skill] = None
    return _skill_embedding_cache[skill]


def _get_skill_weight(skill: str) -> float:
    skill = skill.lower().strip()
    return WEIGHTED_SKILLS.get(skill, DEFAULT_SKILL_WEIGHT)


def _is_strict_skill(skill: str) -> bool:
    """
    Dynamically determine if a skill should use strict matching (not just semantic).
    Heuristics:
    - Contains special chars (-, _, ., ++): libraries/frameworks
    - Multi-word compound terms: specific tools/concepts
    - Known library patterns: frameworks, databases, tools
    - Short acronyms: usually specific tools
    Returns True if skill should require higher match threshold.
    """
    skill_lower = skill.lower().strip()
    
    # Pattern 1: Contains special characters (indicates library/framework)
    if any(c in skill_lower for c in ['-', '_', '.', '++']):
        return True
    
    # Pattern 2: Multi-word compound terms (2+ words, not generic)
    words = [w for w in skill_lower.split() if w]
    if len(words) >= 2:
        generic_words = {'and', 'or', 'with', 'using', 'for', 'of', 'in', 'on', 'at', 'is', 'the', 'a'}
        non_generic = [w for w in words if w not in generic_words and len(w) > 2]
        # Multi-word technical terms like "feature engineering", "model deployment"
        if len(non_generic) >= 2:
            return True
    
    # Pattern 3: Known library/framework/tool patterns
    library_indicators = [
        'tensorflow', 'torch', 'pytorch', 'keras', 'scikit', 'sklearn',
        'pandas', 'numpy', 'flask', 'django', 'spring', 'laravel',
        'react', 'angular', 'vue', 'svelte', 'next',
        'mongo', 'postgres', 'mysql', 'cassandra', 'elastic',
        'docker', 'kubernetes', 'jenkins', 'terraform',
        'aws', 'azure', 'gcp', 'firebase'
    ]
    if any(indicator in skill_lower for indicator in library_indicators):
        return True

    # Pattern 4: Short known acronyms (lowercase normalized skills)
    acronym_indicators = {'sql', 'api', 'oop', 'dsa', 'ml', 'aws', 'gcp', 'ui', 'ux'}
    if skill_lower in acronym_indicators:
        return True
    
    # Pattern 5: Short uppercase acronyms (SQL, OOP, DSA, API, etc.)
    if 2 <= len(skill_lower) <= 5 and skill_lower.isupper():
        return True
    
    return False


def _get_match_threshold(skill: str) -> float:
    """
    Get appropriate match threshold based on skill type.
    Respects the module-level MATCH_THRESHOLD override (set by the Streamlit slider).
    Strict skills (specific frameworks/libs) use MATCH_THRESHOLD_STRICT (now 0.60 instead of 0.85).
    """
    base = MATCH_THRESHOLD  # honours the slider override
    if _is_strict_skill(skill):
        # Use the lowered strict threshold (0.60 instead of 0.85)
        return max(MATCH_THRESHOLD_STRICT, base + 0.05)  # Reduced bonus from 0.15 to 0.05
    return base


def _resume_text_for_context() -> str:
    chunks = query_resume_top_k("skills experience projects", k=20)
    return " ".join(c["content"] for c in chunks).lower()


_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience\s+)?(?:in|with|using)?",
    re.I,
)
_SENIOR_VERBS = re.compile(
    r"\b(architected|designed|led|built|developed|deployed|scaled|"
    r"implemented|migrated|optimized|owned|delivered|launched|"
    r"maintained|refactored)\b",
    re.I,
)
_QUALIFIER_RE = re.compile(
    r"\b(basic|familiar|introductory|some\s+exposure|exposure to|"
    r"working knowledge|understanding|beginner|limited|learning)\b",
    re.I,
)


def _contextual_confidence(skill: str, resume_text: str) -> float:
    """
    Returns a confidence multiplier for a JD skill's match score.
    Levels: basic qualifier→0.60, not found→0.75, once→0.85,
            2+ times→1.00, senior verb nearby→1.05, years found→1.10
    """
    if not skill or not resume_text:
        return 1.0

    normalized_resume = normalize_text(resume_text)
    normalized_skill = normalize_text(skill)

    occurrences = len(re.findall(
        rf"(?<![a-z0-9]){re.escape(normalized_skill)}(?![a-z0-9])",
        normalized_resume,
    ))

    if occurrences == 0:
        return 0.75

    # Check for explicit weakness qualifier near each mention
    window = 60
    for match in re.finditer(
        rf"(?<![a-z0-9]){re.escape(normalized_skill)}(?![a-z0-9])",
        normalized_resume,
    ):
        start = max(0, match.start() - window)
        end = min(len(normalized_resume), match.end() + window)
        if _QUALIFIER_RE.search(normalized_resume[start:end]):
            return 0.60

    # Check for years-of-experience near the skill mention
    for match in re.finditer(
        rf"(?<![a-z0-9]){re.escape(normalized_skill)}(?![a-z0-9])",
        normalized_resume,
    ):
        start = max(0, match.start() - 80)
        end = min(len(normalized_resume), match.end() + 80)
        year_match = _YEARS_RE.search(normalized_resume[start:end])
        if year_match and int(year_match.group(1)) >= 1:
            return 1.10

    # Check for senior action verbs near the skill mention
    for match in re.finditer(
        rf"(?<![a-z0-9]){re.escape(normalized_skill)}(?![a-z0-9])",
        normalized_resume,
    ):
        start = max(0, match.start() - 100)
        end = min(len(normalized_resume), match.end() + 100)
        if _SENIOR_VERBS.search(normalized_resume[start:end]):
            return 1.05

    return 1.00 if occurrences >= 2 else 0.85


def _compute_skill_similarity(skill1: str, skill2: str) -> float:
    """
    Compute cosine similarity between two skills using cached embeddings.
    Returns score from 0 to 1.
    """
    try:
        v1 = _get_cached_embedding(skill1)
        v2 = _get_cached_embedding(skill2)
        
        if v1 is None or v2 is None:
            return 0.0
        
        # Reshape for sklearn cosine_similarity
        v1 = np.array(v1).reshape(1, -1)
        v2 = np.array(v2).reshape(1, -1)
        similarity = cosine_similarity(v1, v2)[0][0]
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    except Exception as e:
        print(f"[agent_3] Similarity computation error: {e}")
        return 0.0


def _match_skills_similarity(resume_skills: list, jd_skills: list, debug=True) -> tuple:
    """
    Match JD skills against resume skills using strict + semantic matching.
    Priority order:
    1. Exact match (jd_skill in resume_skills)
    2. Strict skills enforcement (STRICT_SKILLS must be explicit)
    3. Semantic similarity with higher thresholds
    
    Returns (matched, partial, missing) lists.
    
    Thresholds:
    - >= 0.85: matched (strict semantic)
    - 0.60-0.85: partial (loose semantic)
    - < 0.60: missing
    """
    if not resume_skills or not jd_skills:
        return [], [], []
    
    # Apply skill normalization (FIX 1)
    resume_skills = [normalize_skill(s) for s in resume_skills]
    jd_skills = [normalize_skill(s) for s in jd_skills]
    
    matched = []
    partial = []
    missing = []

    resume_text = _resume_text_for_context()

    # Deduplicate and cache embeddings for all skills upfront
    all_unique_skills = set(resume_skills + jd_skills)
    print(f"[agent_3] Pre-computing embeddings for {len(all_unique_skills)} unique skills...")
    
    for skill in all_unique_skills:
        _get_cached_embedding(skill)

    if debug:
        print(f"\n[agent_3] ===== DEBUG: SKILL MATCHING =====")
        print(f"[agent_3] Resume skills: {resume_skills}")
        print(f"[agent_3] JD skills: {jd_skills}\n")

    print(f"[agent_3] Matching {len(jd_skills)} JD skills against {len(resume_skills)} resume skills...")
    
    for jd_skill in jd_skills:
        jd_skill_lower = jd_skill.lower().strip()
        best_score = 0.0
        best_score_raw = 0.0
        best_resume_skill = None
        best_confidence = 1.0
        is_exact_match = False
        is_strict = _is_strict_skill(jd_skill)
        match_threshold = _get_match_threshold(jd_skill)

        # PRIORITY 1: Check for direct/exact match (must come first)
        if jd_skill_lower in [s.lower().strip() for s in resume_skills]:
            is_exact_match = True
            best_score = 1.0
            best_resume_skill = jd_skill_lower
            if debug:
                print(f"\n[agent_3] Checking JD skill: '{jd_skill}' -> EXACT MATCH in resume")
        else:
            # PRIORITY 2: Check for direct substring match (FIX 3)
            for resume_skill in resume_skills:
                if is_direct_match(jd_skill, resume_skill):
                    is_exact_match = True
                    best_score = 1.0
                    best_resume_skill = resume_skill
                    if debug:
                        print(f"\n[agent_3] Checking JD skill: '{jd_skill}' -> DIRECT SUBSTRING MATCH: '{resume_skill}'")
                    break
            
            if not is_exact_match:
                # PRIORITY 3: Use semantic similarity with dynamic threshold
                confidence = _contextual_confidence(jd_skill, resume_text)

                if debug:
                    skill_type = "STRICT" if is_strict else "general"
                    print(f"\n[agent_3] Checking JD skill: '{jd_skill}' ({skill_type}, threshold={match_threshold:.2f})")

                for resume_skill in resume_skills:
                    raw_score = _compute_skill_similarity(jd_skill, resume_skill)
                    adjusted_score = max(0.0, min(1.0, raw_score * confidence))
                    
                    if debug:
                        print(f"  vs '{resume_skill}': raw={raw_score:.4f}, adj={adjusted_score:.4f}")
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_score_raw = raw_score
                        best_resume_skill = resume_skill
                        best_confidence = confidence

        if debug and not is_exact_match:
            print(f"  -> Best match: '{best_resume_skill}' [{best_score:.4f}]")

        # Classification using dynamic thresholds
        if best_score >= match_threshold:
            matched.append(MatchedSkill(
                requirement=jd_skill,
                resume_excerpt=best_resume_skill or "",
                similarity_score=round(best_score, 3),
            ))
        elif best_score >= PARTIAL_THRESHOLD:
            partial.append(MatchedSkill(
                requirement=jd_skill,
                resume_excerpt=best_resume_skill or "",
                similarity_score=round(best_score, 3),
            ))
        else:
            missing.append(MissingSkill(
                requirement=jd_skill,
                similarity_score=round(best_score, 3),
            ))

    if debug:
        print(f"\n[agent_3] ===== EMBEDDING CACHE DEBUG =====")
        for skill, embedding in _skill_embedding_cache.items():
            if embedding is not None:
                # Show first 5 values of embedding to verify they're different
                emb_sample = embedding[:5] if hasattr(embedding, '__getitem__') else "N/A"
                print(f"  '{skill}': {emb_sample}...")
            else:
                print(f"  '{skill}': None")

    return matched, partial, missing


# ── MAIN RUN FUNCTION ────────────────────────────────────────────────────────
def run(jd_text=None, verbose=True) -> ATSResult:
    """
    NEW: Skill-based similarity matching using structured skills from ChromaDB.
    More deterministic and consistent across domains.
    """
    global _skill_embedding_cache
    _skill_embedding_cache = {}  # Clear cache for fresh run
    
    try:
        # Get structured skills from ChromaDB
        resume_skills = get_skills("resume")
        jd_skills_raw = get_skills("jd")
        
        # FIX 1: Clean JD skills to extract actual tech keywords
        jd_skills = [clean_jd_skill(s) for s in jd_skills_raw]
        jd_skills = [s for s in jd_skills if s]  # Remove None values
        
        if verbose:
            print(f"\n[agent_3] ========== SKILL-BASED MATCHING ==========")
            print(f"[agent_3] Resume skills extracted: {len(resume_skills)}")
            if resume_skills:
                print(f"[agent_3]   Sample: {resume_skills[:5]}")
            print(f"[agent_3] JD skills extracted (raw): {len(jd_skills_raw)}")
            if jd_skills_raw:
                print(f"[agent_3]   Sample raw: {jd_skills_raw[:5]}")
            print(f"[agent_3] JD skills after cleaning: {len(jd_skills)}")
            if jd_skills:
                print(f"[agent_3]   Sample cleaned: {jd_skills[:5]}")

        if not jd_skills:
            print("[agent_3] ⚠️  No JD skills extracted. Falling back to keyword matching...")
            return _run_legacy_matching(jd_text, verbose)
        
        if not resume_skills:
            print("[agent_3] ⚠️  No resume skills extracted. All requirements will be marked as missing.")

        # Compute skill-based matching with debug output enabled
        matching, partial, missing = _match_skills_similarity(resume_skills, jd_skills, debug=verbose)

        if verbose:
            print(f"\n[agent_3] RESULTS (Strict Matching):")
            print(f"[agent_3]   ✅ Matched (exact or high similarity): {len(matching)}")
            print(f"[agent_3]   ⚠️  Partial (semantic similarity): {len(partial)}")
            print(f"[agent_3]   ❌ Missing (not found or low similarity): {len(missing)}")
            if missing:
                print(f"[agent_3]   Missing details: {[m.requirement for m in missing]}")

        total_weight = sum(_get_skill_weight(skill) for skill in jd_skills)
        matched_weight = sum(_get_skill_weight(m.requirement) for m in matching)
        partial_weight = sum(0.5 * _get_skill_weight(p.requirement) for p in partial)
        score = (matched_weight + partial_weight) / total_weight * 100 if total_weight > 0 else 0

        if verbose:
            print(f"[agent_3]   ---")
            print(f"[agent_3]   weighted total: {total_weight:.1f}")
            print(f"[agent_3]   matched weight: {matched_weight:.1f}")
            print(f"[agent_3]   partial weight: {partial_weight:.1f}")
            print(f"[agent_3]   ATS Score: {score:.1f}%")

        return ATSResult(
            ats_score=round(score, 1),
            matching_skills=matching,
            partial_matches=partial,
            missing_skills=missing,
            summary=f"Score: {round(score, 1)}% — {len(matching)} matched, {len(partial)} partial, {len(missing)} missing"
        )

    except Exception as e:
        print(f"[agent_3] Skill similarity matching failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to legacy matching
        return _run_legacy_matching(jd_text, verbose)


def _run_legacy_matching(jd_text=None, verbose=True) -> ATSResult:
    """Fallback to original CrossEncoder-based matching if skill extraction fails."""
    try:
        model = _get_model()
        jd_requirements = get_jd_only_requirements(jd_text)

        matching = []
        partial = []
        missing = []
        weighted_points = 0.0

        # collect full resume text
        all_chunks = query_resume_top_k("skills experience projects", k=20)
        all_resume_text = " ".join(c["content"].lower() for c in all_chunks)

        for req in jd_requirements:

            # skip garbage
            if not _is_likely_real_skill(req):
                continue

            chunks = query_resume_top_k(req, k=5)

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
        print("[agent_3] Fallback ERROR:", e)
        return ATSResult(
            ats_score=0,
            matching_skills=[],
            partial_matches=[],
            missing_skills=[],
            summary="Error"
        )