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
# Now that extraction is atomic, these thresholds are reliable.
# Short skill names (1-2 words) like "python", "spark" match much better
# than long sentences did before.

MATCH_THRESHOLD   = 0.60   # score >= 0.60 → matched ✅
PARTIAL_THRESHOLD = 0.38   # score >= 0.38 → partial ⚠️
MISSING_THRESHOLD = 0.20   # score >= 0.20 → missing ❌  (real gap, low but detectable)
                           # score  < 0.20 → IGNORED  (noise, not a real skill)

# ── Filler words — strip to get meaningful tech keywords ─────────────────────
_FILLER = re.compile(
    r"\b(and|or|the|with|to|of|in|for|a|an|such|as|its|that|is|are|on|"
    r"using|via|including|related|best|practices|concepts|services|platform|"
    r"tools|techniques|languages|processing|storage|efficiency|translate|"
    r"gather|optimize|enhance|collaborate|develop|stakeholders|requirements|"
    r"specifications|unified|analytics|manipulation|integration|modeling|"
    r"ability|strong|knowledge|understanding|experience|proficiency|"
    r"familiarity|working)\b",
    re.I,
)


def _extract_keywords(requirement: str) -> str:
    """Strip filler to get tight technical keywords for search queries."""
    cleaned = _FILLER.sub(" ", requirement)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    keywords = [w for w in cleaned.split() if len(w) >= 3]
    result = " ".join(keywords[:8])
    return result if result.strip() else requirement


def _keyword_boost(requirement: str, chunk_content: str) -> float:
    """
    Hybrid scoring: check if the requirement's tokens appear in the resume chunk.

    Since requirements are now ATOMIC (short skill names like "html", "node.js",
    "git version control"), we do token-level matching against the chunk.

    Works for:
      req="html"         chunk="HTML, CSS, JavaScript" → boost 0.72
      req="node.js"      chunk="Node.js Express React"  → boost 0.72
      req="rest apis"    chunk="built REST API backend" → boost 0.72
    """
    req_lower = requirement.lower().strip()
    chunk_lower = chunk_content.lower()

    # --- Direct substring match first (most reliable for short atomic skills) ---
    if req_lower in chunk_lower:
        return 0.72   # capped: strong signal but not overwhelming

    # --- Token-level match ---
    # Split requirement into meaningful tokens (>=2 chars)
    # Don't strip tech names — only strip pure English stopwords
    _stopwords = {
        "and", "or", "the", "with", "to", "of", "in", "for", "a", "an",
        "on", "at", "by", "is", "are", "be", "as", "it", "its",
    }
    tokens = [
        t for t in re.split(r"[\s\-]+", req_lower)
        if t and len(t) >= 2 and t not in _stopwords
    ]

    if not tokens:
        return 0.0

    def _tok_in_text(tok: str, text: str) -> bool:
        """Match token, also trying simple singular/plural variants."""
        if tok in text:
            return True
        # "apis" → also try "api"
        if tok.endswith("s") and len(tok) > 3 and tok[:-1] in text:
            return True
        # "api" → also try "apis"
        if not tok.endswith("s") and (tok + "s") in text:
            return True
        return False

    matches = sum(1 for tok in tokens if _tok_in_text(tok, chunk_lower))
    match_ratio = matches / len(tokens)

    # For single-token requirements (e.g. "python", "html"):
    # one match = definite hit
    if len(tokens) == 1:
        return 0.68 if matches == 1 else 0.0   # capped from 0.75

    # For multi-token requirements (e.g. "git version control"):
    if match_ratio >= 0.67:   # 2/3 or more tokens found
        return 0.65            # capped from 0.72
    elif match_ratio >= 0.34:  # at least 1 token found
        return 0.40            # capped from 0.45
    return 0.0


def _is_likely_real_skill(requirement: str) -> bool:
    """
    Determine if a requirement string is a real skill/technology.

    Uses ordered rejection → ordered acceptance. NO static skill lists needed.

    Rejects:
      - too long (>7 words)
      - starts with a duty verb
      - contains sentence-structure words
      - single plain English word (not a tech acronym/name)
      - known meta/noise phrases

    Accepts:
      - contains tech-signal chars (digit, dot, #, +)
      - dot-between-chars pattern (node.js, vue.js, express.js)
      - starts with a dot (.net)
      - short alphabetic word ≤5 chars (php, dart, swift, sql, html, css…)
      - 2–4 word compound with no rejection markers
    """
    req = requirement.strip().lower()
    if not req or len(req) < 2:
        return False

    words = req.split()

    # ── REJECTION RULES (ordered) ─────────────────────────────────────────────

    # 1. Too long → duty sentence
    if len(words) > 7:
        return False

    # 2. Starts with a duty/action verb
    _duty_start = re.compile(
        r"^(collaborate|communicate|coordinate|support|assist|gather|"
        r"translate|participate|perform|drive|deliver|ensure|establish|"
        r"develop|build|create|design|implement|manage|maintain|monitor|"
        r"write|fix|learn|study|work|handle|debug|test|identify|solve|"
        r"adhere|follow|produce|document|help|troubleshoot|optimize|"
        r"willingness|eager|quick\s+learner|fast\s+learner|"
        r"you\s+will|expected\s+to|responsible\s+for|ability\s+to)",
        re.I,
    )
    if _duty_start.search(req):
        return False

    # 3. Contains sentence-structure words
    _sentence_words = {
        "the", "this", "that", "which", "who", "where", "when",
        "how", "why", "our", "their", "your", "we", "they",
        "will", "would", "should", "could", "must", "shall",
    }
    if any(w in _sentence_words for w in words):
        return False

    # 4. Single plain English word that is clearly not a tech name
    _plain_singles = {
        "study", "learn", "efficient", "clean", "good", "strong",
        "ability", "knowledge", "exposure", "understanding", "work",
        "code", "pay", "location", "month", "sessions", "workshops",
        "responsibilities", "soft", "issues", "practices", "standards",
        "modules", "quality", "motivated", "oriented", "player", "detail",
    }
    if len(words) == 1 and req in _plain_singles:
        return False

    # 5. Known meta/noise multi-word phrases
    _meta_re = re.compile(
        r"(technical\s+issues|coding\s+best|best\s+coding|clean\s+code"
        r"|well.documented|code\s+quality|willingness\s+to|problem.solving"
        r"|back.end\s+module|communication\s+skill|interpersonal|analytical\s+skill"
        r"|time\s+management|attention\s+to\s+detail|team\s+player|self.motivated"
        r"|write\s+clean|write\s+efficient|write\s+well|produce\s+clean"
        r"|deliver\s+clean|maintain\s+clean|ensure\s+code"
        # Pure metadata/header phrases that are NOT skills
        r"|^skills\s+required$|^required\s+skills$|^key\s+skills$"
        r"|^technical\s+skills$|^soft\s+skills$|^core\s+skills$"
        r"|^must\s+have$|^good\s+to\s+have$|^nice\s+to\s+have$"
        r"|workshops\s+on|sessions\s+on|training\s+on|exposure\s+to"
        r"|gaining\s+hands.on|hands.on\s+experience\s+in"
        r"|role\s+focuses|position\s+focuses|job\s+focuses)",
        re.I,
    )
    if _meta_re.search(req):
        return False

    # 5b. Broad concept phrases — sound like skills but are really categories/duties
    # "mobile application development", "web application development", etc.
    _broad_concept_re = re.compile(
        r"^(mobile|web|software|application|full.?stack|front.?end|back.?end|"
        r"cloud|data|enterprise|digital)\s+"
        r"(application|app|development|engineering|architecture|solutions?|"
        r"services?|infrastructure|operations?|management)\s*"
        r"(development|engineering|architecture|solutions?)?$",
        re.I,
    )
    if _broad_concept_re.search(req):
        return False

    # ── ACCEPTANCE RULES (ordered) ────────────────────────────────────────────

    for w in words:
        # 6. Tech-signal chars: digits, #, + (e.g. "c++", "python3", ".net")
        if re.search(r"[0-9#+]", w):
            return True
        # 7. Dot-between-alphanumeric chars (e.g. "node.js", "vue.js", "express.js")
        if re.search(r"[a-zA-Z0-9]\.[a-zA-Z]", w):
            return True
        # 8. Starts with dot (.net)
        if w.startswith(".") and len(w) >= 3:
            return True
        # 9. Short-to-medium alphabetic word ≤8 chars → likely a tech name
        #    (covers: php=3, html=4, react=5, django=6, flutter=7, angular=7, pytorch=7)
        if len(w) <= 8 and w.isalpha():
            return True

    # 10. 2–4 word compound phrase — only accept if at least one word looks tech-like
    #     (prevents "skills required", "mobile application" etc. from slipping through)
    if 2 <= len(words) <= 4:
        _plain_connectors = {"and", "or", "of", "for", "in", "on", "to", "with",
                             "at", "by", "a", "an", "the", "is", "are"}
        meaningful = [w for w in words if w not in _plain_connectors]
        # At least one meaningful word must be short (≤8) alphabetic or contain tech chars
        tech_word_found = any(
            (len(w) <= 8 and w.isalpha()) or re.search(r"[0-9.#+]", w)
            for w in meaningful
        )
        if tech_word_found:
            return True

    return False


def _is_likely_noise(requirement: str) -> bool:
    """
    Pre-filter: detect requirements that are clearly job duties or noise
    BEFORE running the expensive CrossEncoder.

    Since our new extraction already filters most of these out upstream,
    this is a safety net for anything that slipped through.
    """
    req = requirement.lower().strip()
    words = req.split()

    # Very short (1 char) or very long (>10 words) after atomization = suspicious
    if len(req) < 2:
        return True
    if len(words) > 10:
        # Long phrase remaining after atomization — check if it's a duty sentence
        duty_verbs = re.compile(
            r"\b(collaborate|communicate|coordinate|support|assist|gather|"
            r"translate|participate|perform|drive|deliver|ensure|establish)\b",
            re.I,
        )
        if duty_verbs.search(req):
            return True

    return False


def run(jd_text=None, verbose=True) -> ATSResult:
    try:
        model = _get_model()
        jd_requirements = get_jd_only_requirements(jd_text)

        if not jd_requirements:
            print("[agent_3] WARNING: No JD requirements found in DB")
            return ATSResult(
                ats_score=0,
                matching_skills=[],
                partial_matches=[],
                missing_skills=[],
                summary="No JD requirements found",
            )

        # Pre-filter obvious noise before any embedding work
        filtered_requirements = []
        for req in jd_requirements:
            if _is_likely_noise(req):
                if verbose:
                    print(f"  [PRE-FILTER] Skipping noise: '{req}'")
            else:
                filtered_requirements.append(req)

        print(f"[agent_3] Found {len(jd_requirements)} JD requirements "
              f"({len(jd_requirements) - len(filtered_requirements)} pre-filtered as noise):")
        for r in filtered_requirements:
            print(f"  · {r}")

        jd_requirements = filtered_requirements

        # Initialize result lists early — needed by global keyword match fallback in Step 1
        matching: list = []
        partial: list = []
        missing: list = []
        weighted_points: float = 0.0

        # ── Pre-step: collect ALL resume text once ────────────────────────────
        all_resume_chunks = query_resume_top_k("skills experience projects", k=50)
        all_resume_text = " ".join(c["content"].lower() for c in all_resume_chunks)

        # ── Pre-step: Direct text lookup BEFORE CrossEncoder ─────────────────
        # For 1-3 word requirements, do a word-boundary regex search across all
        # resume text. This fixes CSS/HTML/Git/Flutter getting score=0 from bad
        # chunk retrieval — if the word is literally in the resume, it's a MATCH.
        direct_matched: set = set()

        for req in jd_requirements:
            req_words = req.split()
            if len(req_words) > 3:
                continue
            # Build a word-boundary pattern (handles "css", "node.js", ".net" etc.)
            escaped = re.escape(req.lower())
            pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
            if re.search(pattern, all_resume_text):
                if verbose:
                    print(f"  [DIRECT ✅] '{req}' found literally in resume text")
                direct_matched.add(req)
                matching.append(MatchedSkill(
                    requirement=req,
                    resume_excerpt="(direct text match in resume)",
                    similarity_score=0.85,
                ))
                weighted_points += 1.0

        # ── Step 1: Retrieve resume chunks per requirement ────────────────────
        req_chunks = {}
        empty_reqs = []

        for req in jd_requirements:
            # Skip requirements already resolved by direct text lookup
            if req in direct_matched:
                continue

            # Section-priority retrieval for short skills (1-2 words):
            # search skills/experience/projects FIRST to avoid education-chunk noise
            # (e.g. node.js appearing in a course title shouldn't count as a match)
            if len(req.split()) <= 2:
                chunks = query_resume_top_k(
                    req, k=15,
                    section_filter="skills,experience,projects",
                )
                if not chunks:
                    chunks = query_resume_top_k(req, k=15)
            elif len(req.split()) <= 3:
                # 3-word skill: search directly across all sections
                chunks = query_resume_top_k(req, k=15)
            else:
                keyword_query = _extract_keywords(req)
                search_query = keyword_query if keyword_query.strip() else req
                chunks = query_resume_top_k(
                    search_query,
                    k=15,
                    section_filter="skills,experience,projects",
                )
                if not chunks:
                    chunks = query_resume_top_k(search_query, k=15)
                if not chunks:
                    chunks = query_resume_top_k(req, k=15)

            if verbose:
                print(f"\n[agent_3] REQ: '{req}'")
                print(f"          CHUNKS FOUND: {len(chunks)}")

            if chunks:
                req_chunks[req] = chunks
            else:
                # No chunks at all (and not direct-matched) — check keyword boost
                boost = _keyword_boost(req, all_resume_text)
                if boost >= MATCH_THRESHOLD:
                    if verbose:
                        print(f"          → GLOBAL MATCH via keyword boost={boost:.3f}")
                    matching.append(MatchedSkill(
                        requirement=req,
                        resume_excerpt="(matched via keyword search across resume)",
                        similarity_score=round(boost, 3),
                    ))
                    weighted_points += 1.0
                elif boost >= PARTIAL_THRESHOLD:
                    if verbose:
                        print(f"          → GLOBAL PARTIAL via keyword boost={boost:.3f}")
                    partial.append(MatchedSkill(
                        requirement=req,
                        resume_excerpt="(partial match across resume)",
                        similarity_score=round(boost, 3),
                    ))
                    weighted_points += 0.5
                elif boost >= MISSING_THRESHOLD:
                    empty_reqs.append(req)
                else:
                    # No chunks AND low boost — if it looks like a real skill, it's missing
                    if _is_likely_real_skill(req):
                        empty_reqs.append(req)
                    elif verbose:
                        print(f"          → IGNORED (no chunks, boost={boost:.3f}, not a skill)")

        # ── Step 2: Batch CrossEncoder pairs ─────────────────────────────────
        all_pairs = []
        pair_map = []

        for req, chunks in req_chunks.items():
            keyword_query = _extract_keywords(req)
            # For short atomic skills, use the skill name directly as the question
            question_text = req if len(req.split()) <= 3 else keyword_query

            for i, chunk in enumerate(chunks):
                all_pairs.append((
                    f"Does the candidate have experience with {question_text}?",
                    f"Resume:\n{chunk['content']}",
                ))
                pair_map.append((req, i))

        # ── Step 3: Batch predict ─────────────────────────────────────────────
        if all_pairs:
            print(f"\n[agent_3] Running batch CrossEncoder on {len(all_pairs)} pairs...")
            raw_scores = model.predict(all_pairs)
            semantic_scores = [_sigmoid(float(s)) for s in raw_scores]
        else:
            semantic_scores = []

        # ── Step 4: Hybrid scoring ────────────────────────────────────────────
        final_scores = []
        for idx, score in enumerate(semantic_scores):
            req, chunk_idx = pair_map[idx]
            chunk_content = req_chunks[req][chunk_idx]["content"]
            boost = _keyword_boost(req, chunk_content)
            hybrid_score = max(score, boost)
            final_scores.append(hybrid_score)

        # ── Step 5: Best chunk per requirement ───────────────────────────────
        req_best = {}
        for idx, (req, chunk_idx) in enumerate(pair_map):
            score = final_scores[idx]
            chunk = req_chunks[req][chunk_idx]
            semantic = semantic_scores[idx]
            boost = _keyword_boost(req, chunk["content"])

            if req not in req_best or score > req_best[req][0]:
                req_best[req] = (score, chunk, semantic, boost)

        # ── Step 6: Classify ──────────────────────────────────────────────────
        #
        #  score >= MATCH_THRESHOLD   → MATCHED  (1.0 points)
        #  score >= PARTIAL_THRESHOLD → PARTIAL  (0.5 points) — real skill with weak evidence
        #  score >= MISSING_THRESHOLD → MISSING  (0.0 points)
        #  score <  MISSING_THRESHOLD → rescue via full-text boost, then IGNORE if noise
        # ─────────────────────────────────────────────────────────

        # Requirements with NO chunks at all → only add to missing if not noise
        for req in empty_reqs:
            if not _is_likely_noise(req):
                missing.append(MissingSkill(requirement=req, similarity_score=0.0))

        for req, (best_score, best_chunk, semantic, boost) in req_best.items():
            # Skip requirements already classified by direct text lookup
            if req in direct_matched:
                continue

            if best_score >= MATCH_THRESHOLD:
                tag = "MATCH ✅"
            elif best_score >= PARTIAL_THRESHOLD:
                tag = "PARTIAL ⚠️"
            elif best_score >= MISSING_THRESHOLD:
                tag = "MISSING ❌"
            else:
                tag = "IGNORED 🚫"

            if verbose:
                print(f"\n  [{tag}] {req}")
                print(f"    score={best_score:.3f} | semantic={semantic:.3f} | keyword_boost={boost:.3f}")
                print(f"    section={best_chunk.get('section','?')} | chunk: {best_chunk['content'][:100]}...")

            if best_score >= MATCH_THRESHOLD:
                weighted_points += 1.0
                matching.append(MatchedSkill(
                    requirement=req,
                    resume_excerpt=best_chunk["content"][:200],
                    similarity_score=round(best_score, 3),
                ))
            elif best_score >= PARTIAL_THRESHOLD:
                weighted_points += 0.5
                partial.append(MatchedSkill(
                    requirement=req,
                    resume_excerpt=best_chunk["content"][:200],
                    similarity_score=round(best_score, 3),
                ))
            elif best_score >= MISSING_THRESHOLD:
                missing.append(MissingSkill(
                    requirement=req,
                    similarity_score=round(best_score, 3),
                ))
            else:
                # Score is very low — try full-resume keyword boost as rescue.
                # Fixed floors used here so the slider can't block a keyword hit.
                global_boost = _keyword_boost(req, all_resume_text)

                if global_boost >= MATCH_THRESHOLD:
                    if verbose:
                        print(f"    → MATCH (rescued via full-text boost={global_boost:.3f})")
                    weighted_points += 1.0
                    matching.append(MatchedSkill(
                        requirement=req,
                        resume_excerpt="(keyword match across resume)",
                        similarity_score=round(global_boost, 3),
                    ))
                elif global_boost >= 0.35 and _is_likely_real_skill(req):
                    if verbose:
                        print(f"    → PARTIAL (rescued via full-text boost={global_boost:.3f})")
                    weighted_points += 0.5
                    partial.append(MatchedSkill(
                        requirement=req,
                        resume_excerpt="(partial keyword match across resume)",
                        similarity_score=round(global_boost, 3),
                    ))
                elif _is_likely_real_skill(req):
                    if verbose:
                        print(f"    → MISSING (real skill, score={best_score:.3f}, global_boost={global_boost:.3f})")
                    missing.append(MissingSkill(
                        requirement=req,
                        similarity_score=round(best_score, 3),
                    ))
                else:
                    if verbose:
                        print(f"    → IGNORED: score {best_score:.3f} < {MISSING_THRESHOLD} (noise/duty/not a skill)")

        # Sort missing by score descending (most relevant gaps first)
        missing = sorted(missing, key=lambda m: m.similarity_score, reverse=True)[:10]

        # Weighted score: matched=1.0pt, partial=0.5pt
        total = max(len(matching) + len(partial) + len(missing), len(jd_requirements))
        ats_score = round((weighted_points / total) * 100, 1) if total > 0 else 0

        print(f"\n[agent_3] ══════════════════════════════════════")
        print(f"[agent_3] ATS Score : {ats_score}%")
        print(f"[agent_3] Matched   : {len(matching)} (incl. {len(direct_matched)} direct)")
        print(f"[agent_3] Partial   : {len(partial)}")
        print(f"[agent_3] Missing   : {len(missing)}")
        print(f"[agent_3] ══════════════════════════════════════")

        return ATSResult(
            ats_score=ats_score,
            matching_skills=matching,
            partial_matches=partial,
            missing_skills=missing,
            summary=f"Score: {ats_score}%",
        )

    except Exception as e:
        print(f"[agent_3] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return ATSResult(
            ats_score=0,
            matching_skills=[],
            partial_matches=[],
            missing_skills=[],
            summary=f"Error: {e}",
        )
