import re
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# INIT
client = chromadb.Client()

collection = client.get_or_create_collection("resume")
jd_collection = client.get_or_create_collection("jd_requirements")

model = SentenceTransformer("all-MiniLM-L6-v2")


# SAFE TEXT CONVERSION
def _to_text(x):
    try:
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("content") or x.get("text") or str(x)
        return str(getattr(x, "content", x))
    except:
        return str(x)


# ── STORE RESUME ──────────────────────────────────────────────────────────────
def store_resume_chunks(chunks):
    if not chunks:
        return

    documents = []
    metadatas = []
    ids = []

    for c in chunks:
        if hasattr(c, "content"):
            text = c.content
            section = getattr(c, "section", "other")
        elif isinstance(c, dict):
            text = c.get("content") or c.get("text") or str(c)
            section = c.get("section", "other")
        else:
            text = str(c)
            section = "other"

        text = text.strip()
        if not text:
            continue

        documents.append(text)
        metadatas.append({"section": section})
        ids.append(str(uuid.uuid4()))

    if not documents:
        return

    embeddings = model.encode(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )
    print(f"[vector_store] Stored {len(documents)} resume chunks with section metadata")


# ── QUERY RESUME with optional section filter ─────────────────────────────────
def query_resume_top_k(query, k=25, section_filter=None):
    if not query:
        return []

    query_embedding = model.encode([str(query)])[0]

    where = None
    if section_filter:
        sections = [s.strip() for s in section_filter.split(",") if s.strip()]
        if len(sections) == 1:
            where = {"section": {"$eq": sections[0]}}
        elif len(sections) > 1:
            where = {"section": {"$in": sections}}

    try:
        if where:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
    except Exception:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    if not docs or not docs[0]:
        return []

    out = []
    for i, d in enumerate(docs[0]):
        meta = metas[0][i] if metas and metas[0] else {}
        out.append({
            "content": d,
            "section": meta.get("section", "unknown"),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  JD EXTRACTION — THE FIXED PART
#  Goal: produce ATOMIC skill units, not long sentences.
#  "develop and optimize data pipelines using Spark and Python for ETL workloads"
#  → ["data pipelines", "spark", "python", "etl"]
# ═══════════════════════════════════════════════════════════════════════════════

_BULLET_START = re.compile(r"^\s*(?:[-•*▪▸◦]+|\d+[\.)])\s+(.+)$", re.I)
_SKILL_HEADER = re.compile(
    r"^(must\s+have\s+skills?|good\s+to\s+have\s+skills?|required\s+skills?|"
    r"nice\s+to\s+have\s+skills?|preferred\s+skills?)\s*:\s*(.+)$",
    re.I,
)
_STRIP_LEAD_IN = re.compile(
    r"^(?:must\s+to\s+have\s+skills?\s*:?\s*|proficiency\s+in\s*|experience\s+with\s*|"
    r"strong\s+understanding\s+of\s*|familiarity\s+with\s*|knowledge\s+of\s*|"
    r"expertise\s+in\s*|hands.on\s+experience\s+(?:with|in)\s*|"
    r"working\s+knowledge\s+of\s*|understanding\s+of\s*|ability\s+to\s*)",
    re.I,
)
_SKIP_FULL_LINE = re.compile(
    r"(educational\s+qualification|minimum\s+\d+\s+year|years?\s+full\s+time|"
    r"full\s+time\s+education|this\s+position\s+is\s+based|"
    r"candidate\s+should\s+have\s+minimum|good\s+to\s+have\s+skills?\s*:\s*na\b|"
    r"^summary\s*:|"
    r"per\s+month|per\s+annum|ctc\b|lpa\b|inr\b|\bpay\b.*\d|salary\b|stipend\b|"
    r"\d[\d,]+\s*(?:/-|rs\.?|inr|usd|eur)|"
    r"work\s+(?:location|mode|from\s+home|from\s+office)|on.?site|wfh\b|wfo\b|"
    r"hybrid\s+model|remote\s+work|office\s+location|in\s+person\b|"
    r"^job\s+description\b|^roles?\s+(?:&|and)\s+responsibilities\b|"
    r"^soft\s+skills?\b|^key\s+responsibilities\b|^responsibilities\b|"
    r"^skills\s+required\b|^required\s+skills\b|^technical\s+skills\b|"
    r"kick\s+start\s+your\s+career|about\s+the\s+(?:role|company|team)|"
    r"the\s+role\s+(?:focuses|involves|requires|is)|"
    r"who\s+we\s+are|what\s+you.ll\s+do|what\s+we\s+offer|perks?\s+(?:&|and)|"
    r"benefits\b|equal\s+opportunity|diversity\s+and\s+inclusion|"
    r"interview\s+process|application\s+process|how\s+to\s+apply|"
    r"technical\s+issues\b|coding\s+(?:standards|best\s+practices)\b|"
    r"back.end\s+modules\b|problem.solving\s+ability\b|"
    r"best\s+coding\s+practices\b|clean\s+(?:and\s+)?(?:efficient|well.documented)\s+code\b|"
    r"write\s+(?:clean|efficient|maintainable)\b|well.documented\s+code\b|"
    r"willingness\s+to\s+learn\b|eager\s+to\s+learn\b|quick\s+learner\b|"
    # Duty phrases that survived atomization
    r"workshops\s+on\s+|programming\s+frameworks\b|"
    r"hands.on\s+experience\s+in|gaining\s+hands.on|"
    r"mobile\s+application\s+development\b|web\s+(?:and\s+)?mobile\s+development\b|"
    r"software\s+development\s+lifecycle\b|sdlc\b)",
    re.I,
)
_GARBAGE_SUBSTR = re.compile(
    r"(you\s+will|your\s+typical\s+day|as\s+a\s+data\s+engineer,\s*you|"
    r"ensuring\s+that\s+data\s+is\s+accessible)",
    re.I,
)

# Job duties/responsibilities — NOT skills; should never appear as "missing"
_RESPONSIBILITY_RE = re.compile(
    r"^(collaborate\s+with|work\s+with|partner\s+with|communicate\s+with|"
    r"coordinate\s+with|interact\s+with|engage\s+with|"
    r"contribute\s+(to|in)|participate\s+in|support\s+the|assist\s+(in|with)|"
    r"gather\s+requirements|translate\s+(requirements|into)|"
    r"perform\s+independently|become\s+an?\s+sme|"
    r"expected\s+to|required\s+to|responsible\s+for|"
    r"stakeholder|cross.functional|team\s+discussion|"
    r"you\s+will|you\s+are\s+expected|reporting\s+to|"
    r"help\s+the\s+team|drive\s+alignment|ensure\s+delivery)",
    re.I,
)

# Verb phrases that signal a job duty sentence, not a skill
_DUTY_SENTENCE_RE = re.compile(
    r"^(design\s+and\s+implement|build\s+and\s+maintain|develop\s+and\s+deploy|"
    r"create\s+and\s+manage|lead\s+and\s+coordinate|define\s+and\s+drive|"
    r"monitor\s+and\s+optimize|analyze\s+and\s+report|plan\s+and\s+execute)",
    re.I,
)

# Tech separators within a phrase — split on these to get atomic skills
_TECH_SPLIT_RE = re.compile(
    r"\s+(?:and|or|,|/|;|\|)\s+|\s*,\s*|\s*/\s*|\s*\|\s*|\s*;\s*",
    re.I,
)

# Noise words that aren't skills on their own
_SOLO_NOISE = {
    "data", "system", "process", "tools", "experience", "techniques",
    "collection", "management", "development", "design", "analysis",
    "solution", "platform", "service", "model", "concept", "framework",
    "architecture", "infrastructure", "performance", "quality", "best",
    "practices", "technologies", "systems", "applications",
}

# These words inside a phrase signal it's a duty, not a skill name
_DUTY_VERBS_INLINE = re.compile(
    r"\b(develop|build|create|design|implement|deploy|manage|maintain|"
    r"ensure|provide|support|collaborate|coordinate|optimize|enhance|"
    r"perform|deliver|drive|lead|establish|define|analyze|monitor|"
    r"prepare|produce|review|evaluate|assess|document|communicate|"
    r"translate|gather|facilitate|execute|oversee|own)\b",
    re.I,
)


def _looks_like_duty_sentence(text: str) -> bool:
    """
    Returns True if this phrase is a job duty/responsibility sentence
    rather than a skill name.
    Heuristic: has a duty verb AND is long (>6 words).
    """
    words = text.split()
    if len(words) <= 4:
        return False  # short = likely a skill name
    verb_matches = len(_DUTY_VERBS_INLINE.findall(text))
    # If 2+ duty verbs in a longer sentence → probably a duty
    return verb_matches >= 2


def _normalize_skill_phrase(raw: str) -> str | None:
    """
    Clean a raw skill string into a canonical form.
    Returns None if it's noise/invalid.
    """
    if not raw or not str(raw).strip():
        return None
    p = str(raw).strip()

    # Strip leading bullets/symbols
    p = re.sub(r"^[-–—•·▪▸◦*+\s]+", "", p)
    # Remove parenthetical explanations
    p = re.sub(r"\([^)]{0,120}\)", "", p)
    # Normalize chars
    p = re.sub(r"[^a-zA-Z0-9\s/+.#\-]", " ", p)
    p = re.sub(r"[/]", " ", p)
    p = re.sub(r"\s+", " ", p).strip().lower()
    p = p.rstrip(".,;:")

    # Strip common lead-ins iteratively
    while True:
        nxt = _STRIP_LEAD_IN.sub("", p).strip()
        if nxt == p:
            break
        p = nxt

    if not p or len(p) < 2:
        return None

    words = p.split()
    if len(words) > 24:
        return None
    if len(words) == 1 and p in _SOLO_NOISE:
        return None
    if _GARBAGE_SUBSTR.search(p):
        return None
    if _RESPONSIBILITY_RE.search(p):
        return None
    if re.search(r"minimum\s+\d+\s+year|years?\s+of\s+experience\s+is\s+required", p):
        return None
    if re.search(r"full\s+time\s+education|^\d+\s+years?\s+full", p):
        return None

    # ── EXTRA REJECTIONS ──────────────────────────────────────────────────────

    # Salary / number-heavy fragments: "7000", "7,000.00", "pay 7000"
    digit_chars = sum(c.isdigit() for c in p)
    if digit_chars > 0 and digit_chars / max(len(p), 1) > 0.30:
        return None
    if re.search(r"\d[\d,]+\.?\d*\s*(?:/-|rs\.?|inr|usd|eur|per|month|annum)?", p):
        return None

    # Single plain English action words / adjectives that are not tech names
    _plain_action_singles = {
        "study", "learn", "efficient", "identify", "location", "pay", "month",
        "sessions", "workshops", "responsibilities", "soft", "clean", "good",
        "strong", "ability", "knowledge", "exposure", "understanding",
        "work", "write", "fix", "debug", "test", "code", "handle",
    }
    if len(words) == 1 and p in _plain_action_singles:
        return None

    # Plain-verb-start phrases that are job duties not skills
    _verb_start_re = re.compile(
        r"^(write|fix|learn|work\s+on|willingness|eager|quick|study|"
        r"handle|debug|test\s+and|help\s+with|support\s+the|troubleshoot"
        r"|identify\s+and|solve\s+technical|adhere\s+to|follow\s+best"
        r"|maintain\s+clean|write\s+clean|produce\s+clean|deliver\s+clean"
        r"|ensure\s+code|well.documented|document\s+code)",
        re.I,
    )
    if _verb_start_re.search(p):
        return None

    # Generic JD meta-phrases (catch remaining)
    _meta_phrase_re = re.compile(
        r"(technical\s+issues|coding\s+best\s+practices|best\s+coding\s+practices"
        r"|back.end\s+modules|problem.solving\s+abilit|clean\s+code"
        r"|well.documented\s+code|code\s+quality|coding\s+standards"
        r"|willingness\s+to\s+learn|eagerness\s+to\s+learn|quick\s+learner"
        r"|fast\s+learner|self.motivated|team\s+player|attention\s+to\s+detail"
        r"|communication\s+skills|interpersonal\s+skills|analytical\s+skills"
        r"|time\s+management|result.oriented|detail.oriented"
        # Remaining leakers from log
        r"|skills\s+required|required\s+skills"
        r"|the\s+role\s+focuses|gaining\s+hands.on|hands.on\s+experience\s+in"
        r"|workshops\s+on|programming\s+frameworks"
        r"|mobile\s+application\s+development|web\s+(?:and\s+)?mobile"
        r"|software\s+development\s+lifecycle|development\s+lifecycle)",
        re.I,
    )
    if _meta_phrase_re.search(p):
        return None

    return p


def _resolve_alias(phrase: str) -> str:
    """
    Dynamically normalize known multi-token skill aliases to canonical single forms.
    No static lookup table — uses pattern matching so it handles variants automatically.
    e.g. "rest apis" → "rest api", "node js" → "node.js", "c sharp" → "c#", etc.
    """
    p = phrase.strip().lower()

    # REST API variants: "rest api", "rest apis", "restful api", "restful apis"
    if re.match(r"^restful?\s+apis?$", p):
        return "rest api"

    # Node.js variants: "node js", "node.js", "nodejs"
    if re.match(r"^node\.?j?s?$|^node\s+js$", p):
        return "node.js"

    # .NET variants: ".net", "dot net", "dotnet", "net framework"
    if re.match(r"^\.net$|^dot\s*net$|^dotnet$|^net\s+framework$", p):
        return ".net"

    # C# variants: "c sharp", "c#"
    if re.match(r"^c\s*sharp$|^c#$", p):
        return "c#"

    # C++ variants: "c plus plus", "c++"
    if re.match(r"^c\s*\+\+$|^c\s+plus\s+plus$", p):
        return "c++"

    # Vue.js variants
    if re.match(r"^vue\.?js?$|^vue\s+js$", p):
        return "vue.js"

    # React.js variants
    if re.match(r"^react\.?js?$|^react\s+js$", p):
        return "react.js"

    # Express.js variants
    if re.match(r"^express\.?js?$|^express\s+js$", p):
        return "express.js"

    # Next.js variants
    if re.match(r"^next\.?js?$|^next\s+js$", p):
        return "next.js"

    # TypeScript variants
    if re.match(r"^type\s*script$|^typescript$", p):
        return "typescript"

    # PostgreSQL variants
    if re.match(r"^postgres(?:ql)?$|^postgre\s+sql$", p):
        return "postgresql"

    # MongoDB variants: keep as-is but normalize spacing
    if re.match(r"^mongo\s*db$", p):
        return "mongodb"

    # Machine learning variants
    if re.match(r"^machine\s+learning$|^ml$", p):
        return "machine learning"

    # Deep learning variants
    if re.match(r"^deep\s+learning$|^dl$", p):
        return "deep learning"

    # CI/CD variants
    if re.match(r"^ci\s*/\s*cd$|^ci\s+cd$|^cicd$", p):
        return "ci/cd"

    # Version control / Git variants
    if re.match(r"^git\s+version\s+control$|^version\s+control\s+(?:with\s+)?git$", p):
        return "git"

    # No alias match — return as-is
    return phrase.strip()


def _is_tech_token(word: str) -> bool:
    """
    Returns True if a single word looks like a technology/skill name,
    not a plain English word.
    Examples: "python" ✅ "html" ✅ "node.js" ✅ "the" ❌ "learn" ❌
    """
    w = word.lower().strip()
    if not w or len(w) < 2:
        return False
    # Plain English stopwords / verbs — not tech
    _plain_words = {
        "the", "and", "or", "for", "to", "in", "of", "on", "at", "by",
        "a", "an", "is", "are", "be", "as", "it", "its", "this", "that",
        "learn", "write", "fix", "study", "identify", "assist", "work",
        "participate", "develop", "build", "create", "implement", "manage",
        "skills", "required", "experience", "ability", "practices", "issues",
        "web", "code", "clean", "efficient", "person", "location", "pay",
        "month", "sessions", "workshops", "soft", "responsibilities",
    }
    if w in _plain_words:
        return False
    # Known tech tokens — always keep
    _known_tech = {
        "html", "css", "javascript", "js", "python", "php", "node", "nodejs",
        "flutter", "dart", "java", "kotlin", "swift", "react", "angular",
        "vue", "typescript", "sql", "mysql", "mongodb", "postgres", "git",
        "docker", "kubernetes", "aws", "azure", "gcp", "linux", "bash",
        "c", "c++", "c#", "go", "rust", "r", "scala", "spark", "kafka",
        "redis", "graphql", "rest", "api", "apis", "json", "xml", "http",
        "tcp", "ip", "ssh", "ssl", "tls", "jwt", "oauth", "ci", "cd",
        "cicd", "devops", "agile", "scrum", "jira", "figma", "postman",
        "tensorflow", "pytorch", "sklearn", "pandas", "numpy", "matplotlib",
        "airflow", "dbt", "tableau", "powerbi", "excel", "spring", "django",
        "flask", "fastapi", "express", "rails", "laravel", "net", ".net",
        "orm", "mvc", "oop", "oops", "regex", "xml", "yaml", "toml",
    }
    if w in _known_tech:
        return True
    # Has digits, dots, hash, or plus → likely a tech name (e.g. "node.js", "c++")
    if re.search(r"[0-9.#+]", w):
        return True
    # Short words that look like acronyms (2-4 uppercase-origin chars)
    if len(w) <= 4 and w.isalpha():
        return True
    return False


def _strip_leading_connectors(text: str) -> str:
    """Strip leading 'and'/'or' that appear as artifacts of splitting compound phrases."""
    return re.sub(r"^(?:and|or)\s+", "", text.strip(), flags=re.I).strip()


def _atomize_phrase(phrase: str) -> list[str]:
    """
    Break a compound phrase into atomic skill units.

    Case ordering (important — each case is tried in priority order):
      0. Single word             → normalize and return
      X. Drop duty sentences     → []
      A. Alias resolution        → if whole phrase maps to canonical form, return it
      B. ≤3 words, no connector  → keep as meaningful skill name (e.g. "git version control")
      C. Comma/pipe/semicolon    → split into parts, recurse each
      D. All tech tokens (space-separated) → split into individual tokens
      E. "X using/via/with Y"   → root phrase + tool list
      F. "X and/or Y"           → connector list split
      G. Fallback                → normalize as-is
    """
    phrase = _strip_leading_connectors(phrase.strip())
    if not phrase:
        return []

    words = phrase.split()

    # ── 0. Single word ────────────────────────────────────────────────────────
    if len(words) == 1:
        n = _normalize_skill_phrase(phrase)
        return [n] if n else []

    # ── X. Drop duty sentences early ─────────────────────────────────────────
    if _looks_like_duty_sentence(phrase):
        return []

    # ── A. Alias resolution (BEFORE any splitting) ────────────────────────────
    # Try alias on the whole normalized phrase first.
    # This catches "REST APIs" → "rest api", "git version control" → "git", etc.
    # before anything else splits them apart.
    normed_whole = _normalize_skill_phrase(phrase)
    if normed_whole:
        aliased = _resolve_alias(normed_whole)
        if aliased != normed_whole:
            # Successfully resolved to a known canonical form
            return [aliased]

    # ── B. ≤3 words with no comma/pipe/semicolon → keep as-is skill phrase ────
    # e.g. "git version control", "rest api", "database management"
    # This MUST come before the all-tech-token check (CASE D) to prevent
    # "REST APIs" from splitting into ["rest", "apis"]
    if len(words) <= 3 and not re.search(r"[,;|]", phrase):
        # But still split on "and"/"or" if both sides look like tech
        and_or_parts = re.split(r"\s+(?:and|or)\s+", phrase, flags=re.I)
        if len(and_or_parts) >= 2:
            # Only split if every part is a single tech token (e.g. "php and flutter")
            if all(len(p.split()) == 1 and _is_tech_token(p.strip()) for p in and_or_parts):
                results = []
                for p in and_or_parts:
                    p = _strip_leading_connectors(p)
                    n = _normalize_skill_phrase(p)
                    if n:
                        results.append(n)
                if results:
                    return results
        # Not a simple and/or tech pair → keep whole phrase
        n = _normalize_skill_phrase(phrase)
        return [n] if n else []

    # ── C. Comma / pipe / semicolon separated list ───────────────────────────
    if re.search(r"[,;|]", phrase):
        parts = re.split(r"\s*[,;|]\s*", phrase)
        results = []
        for part in parts:
            part = _strip_leading_connectors(part)
            if not part:
                continue
            sub = _atomize_phrase(part)
            results.extend(sub)
        if results:
            return results

    # ── D. Space-separated tech token list (ALL words are tech tokens) ────────
    # e.g. "html css javascript", "node.js php flutter"
    # NOTE: this comes AFTER ≤3 word check so "REST APIs" never reaches here
    if all(_is_tech_token(w) for w in words):
        results = []
        for w in words:
            n = _normalize_skill_phrase(w)
            if n:
                results.append(n)
        if results:
            return results

    # ── E. "X using/via/with/through Y" → root phrase + tool fragments ────────
    using_match = re.split(r"\s+(?:using|via|with|through)\s+", phrase, maxsplit=1, flags=re.I)
    if len(using_match) == 2:
        root_phrase = using_match[0].strip()
        tool_part = using_match[1].strip()
        results = []
        n_root = _normalize_skill_phrase(root_phrase)
        if n_root and n_root not in _SOLO_NOISE:
            results.append(n_root)
        for tool in re.split(r"\s*(?:and|or|,)\s*", tool_part, flags=re.I):
            t = _strip_leading_connectors(tool)
            if not t or len(t.split()) > 4:
                continue
            # Resolve alias on each fragment too
            t_normed = _normalize_skill_phrase(t)
            if t_normed:
                t_normed = _resolve_alias(t_normed)
            if t_normed and t_normed not in _SOLO_NOISE:
                results.append(t_normed)
        if results:
            return results

    # ── F. "X and/or Y" connector list ───────────────────────────────────────
    parts = re.split(r"\s+(?:and|or)\s+", phrase, flags=re.I)
    if len(parts) >= 2:
        results = []
        for part in parts:
            part = _strip_leading_connectors(part)
            if not part:
                continue
            if len(part.split()) <= 4:
                n = _normalize_skill_phrase(part)
                if n:
                    n = _resolve_alias(n)
                if n and n not in _SOLO_NOISE:
                    results.append(n)
            else:
                sub = _atomize_phrase(part)
                results.extend(sub)
        if results:
            return results

    # ── G. Fallback: normalize as-is ─────────────────────────────────────────
    n = _normalize_skill_phrase(phrase)
    return [n] if n else []


def _preprocess_jd_blob(text: str) -> str:
    t = text
    for pat, repl in (
        (r"([.!?])\s*(Roles\s*&\s*Responsibilities\s*:)", r"\1\n\2"),
        (r"([.!?])\s*(Professional\s*&\s*Technical\s*Skills\s*:)", r"\1\n\2"),
        (r"([.!?])\s*(Additional\s+Information\s*:)", r"\1\n\2"),
        (r"\s+(Roles\s*&\s*Responsibilities\s*:)", r"\n\1"),
        (r"\s+(Professional\s*&\s*Technical\s*Skills\s*:)", r"\n\1"),
        (r"\s+(Additional\s+Information\s*:)", r"\n\1"),
    ):
        t = re.sub(pat, repl, t, flags=re.I)
    return t


def _dash_bullet_pieces(val: str) -> list[str]:
    val = val.strip()
    if not val:
        return []
    pieces = re.split(r"\s+-\s+", val)
    return [p.strip() for p in pieces if p.strip()]


def parse_jd_requirements_from_text(jd_text: str) -> list[str]:
    """
    Extract ATOMIC skill units from a JD.

    Two-pass approach:
    Pass 1: Extract raw phrases (sentences, bullets, key:value pairs)
    Pass 2: Atomize each phrase into skill-level units

    This means:
      "Experience with Python, Spark and Airflow for building data pipelines"
      → ["python", "spark", "airflow", "data pipelines"]

    NOT:
      → ["Experience with Python, Spark and Airflow for building data pipelines"]
    """
    if not jd_text or not str(jd_text).strip():
        return []

    text = _preprocess_jd_blob(str(jd_text).replace("\r\n", "\n").replace("\r", "\n"))
    lines = text.split("\n")

    raw_phrases: list[str] = []  # collect raw phrases before atomization

    def collect(raw: str):
        raw = raw.strip()
        if raw:
            raw_phrases.append(raw)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()

        if _SKIP_FULL_LINE.search(line):
            continue
        if low.startswith("project role description") and len(line) > 100:
            continue
        if (low.startswith("summary:") or (low.startswith("summary ") and len(line) > 80)):
            continue
        if "additional information" in low and len(line) > 60:
            continue

        # "Must have skills: Python; Spark; SQL"
        m_header = _SKILL_HEADER.match(line)
        if m_header:
            rest = m_header.group(2).strip()
            if re.match(r"^na\b", rest, re.I):
                continue
            for piece in re.split(r"\s*;\s*|\s*\|\s*", rest):
                piece = piece.strip()
                if piece and not re.match(r"^na\b", piece, re.I):
                    collect(piece)
            continue

        # "- Python and Spark for data pipelines"
        m_bullet = _BULLET_START.match(line)
        if m_bullet:
            body = m_bullet.group(1).strip()
            collect(body)
            continue

        # "Technical Skills: Python, Spark, SQL"
        if ":" in line:
            key, val = line.split(":", 1)
            key_l = key.strip().lower()
            val = val.strip()
            if not val or re.match(r"^na\b", val, re.I):
                continue
            if "responsibilit" in key_l and not re.match(r"^project\s+role\b", key_l):
                for piece in _dash_bullet_pieces(val):
                    collect(piece)
                continue
            if "technical skills" in key_l or key_l.endswith("technical skills") or (
                "professional" in key_l and "skill" in key_l
            ):
                for piece in _dash_bullet_pieces(val):
                    collect(piece)
                continue
            if key_l in {"must have skills", "good to have skills", "required skills"}:
                for piece in re.split(r"\s*;\s*|\s*\|\s*", val):
                    if piece.strip():
                        collect(piece.strip())
                continue
            if key_l == "project role" and len(val) < 80:
                collect(val)
                continue
            if len(val) > 160:
                continue
            if any(x in key_l for x in (
                "skill", "technology", "tool", "proficiency", "experience",
                "qualification", "requirement"
            )):
                if re.search(r"\s-\s+[A-Za-z]", val) and len(val) > 40:
                    pieces = re.split(r"\s+-\s+", val)
                    pieces = [p.strip() for p in pieces if p.strip()]
                else:
                    pieces = [x.strip() for x in re.split(r"\s*;\s*|\s*\|\s*", val) if x.strip()]
                for piece in pieces:
                    if len(piece) < 220:
                        collect(piece)
                continue

        # Plain line (no bullet, no colon) — only take if short enough to be a skill
        if len(line) <= 120 and not _SKIP_FULL_LINE.search(line):
            collect(line)

    # ── Pass 2: Atomize every collected phrase ────────────────────────────────
    final_skills: list[str] = []
    seen: set[str] = set()

    for phrase in raw_phrases:
        atoms = _atomize_phrase(phrase)
        for atom in atoms:
            if atom and atom not in seen and len(atom) >= 2:
                seen.add(atom)
                final_skills.append(atom)

    # ── Fallback: if nothing extracted, try raw lines ─────────────────────────
    if not final_skills:
        for raw_line in lines:
            line = raw_line.strip()
            if not line or len(line) > 220:
                continue
            if _BULLET_START.match(line):
                continue
            if ":" in line:
                continue
            if _SKIP_FULL_LINE.search(line):
                continue
            n = _normalize_skill_phrase(line)
            if n and n not in seen:
                seen.add(n)
                final_skills.append(n)

    return final_skills


def get_jd_only_requirements(jd_text=None):
    try:
        data = jd_collection.get(include=["documents"])
        docs = data.get("documents") or []
        out = []
        for d in docs:
            if d is None:
                continue
            s = str(d).strip()
            if s:
                out.append(s)
        if out:
            return list(dict.fromkeys(out))
    except Exception:
        pass

    if jd_text:
        return parse_jd_requirements_from_text(str(jd_text))
    return []


# STORE JD REQUIREMENTS
def store_jd_requirements_tagged(jd_items=None, resume_items=None, title=None):
    if not jd_items:
        return

    processed = [_to_text(item) for item in jd_items]
    processed = [p for p in processed if p and p.strip()]

    if not processed:
        return

    embeddings = model.encode(processed)
    jd_collection.add(
        documents=processed,
        embeddings=embeddings.tolist(),
        ids=[str(uuid.uuid4()) for _ in processed],
    )
    print(f"[vector_store] Stored {len(processed)} JD requirements")


# CLEAR DB
def clear_collections():
    global collection, jd_collection

    try:
        client.delete_collection("resume")
    except:
        pass
    try:
        client.delete_collection("jd_requirements")
    except:
        pass

    collection = client.get_or_create_collection("resume")
    jd_collection = client.get_or_create_collection("jd_requirements")


# ── Unified query_resume wrapper ──────────────────────────────────────────────
def query_resume(query=None, k=25, query_text=None, n_results=None, section_filter=None):
    final_query = query_text or query
    if not final_query:
        return []
    final_k = n_results or k or 25
    return query_resume_top_k(final_query, k=final_k, section_filter=section_filter)


def query_jd(query, k=10):
    if not query:
        return []
    query_embedding = model.encode([str(query)])[0]
    try:
        results = jd_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents"],
        )
        docs = results.get("documents", [[]])
        if not docs or not docs[0]:
            return []
        return [{"requirement": d} for d in docs[0]]
    except Exception:
        return []


def get_all_jd_requirements(jd_text=None):
    return get_jd_only_requirements(jd_text)
