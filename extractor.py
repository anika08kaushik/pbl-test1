import re
from pathlib import Path
from typing import List

import PyPDF2

from models import ResumeChunk


SECTION_PATTERNS = {
    "summary":        re.compile(r"\b(summary|objective|profile|about me)\b", re.I),
    "skills":         re.compile(r"\b(skills|technologies|technical skills|competencies|tools|tech stack)\b", re.I),
    "experience":     re.compile(r"\b(experience|work experience|employment|work history|internship)\b", re.I),
    "education":      re.compile(r"\b(education|academics|qualifications|degrees|university|college)\b", re.I),
    "projects":       re.compile(r"\b(projects|portfolio|personal projects|open.?source|academic projects)\b", re.I),
    "certifications": re.compile(r"\b(certifications|certificates|courses|training|achievements|awards)\b", re.I),
}

# ── Header noise patterns (structural/format only, no hardcoded names) ────────
_EMAIL_RE    = re.compile(r"@.*\.", re.I)
_URL_RE      = re.compile(r"https?://|www\.|linkedin\.com|github\.com", re.I)
_PHONE_RE    = re.compile(r"[\+\(]?\d[\d\s\-\(\)]{6,}")
_LOCATION_RE = re.compile(r"\b(india|usa|uk|canada|remote|hybrid|on.?site)\b", re.I)


def _is_header_line(line: str) -> bool:
    """
    Detect resume header lines (name, email, phone, city, LinkedIn, GitHub).
    Purely structural — no hardcoded names.
    """
    t = line.strip()
    if not t:
        return False
    if _EMAIL_RE.search(t):
        return True
    if _URL_RE.search(t):
        return True
    if _PHONE_RE.search(t):
        return True
    if _LOCATION_RE.search(t):
        return True
    # 1–3 title-cased words, no digits, short → likely a name or city line
    words = t.split()
    if (
        1 <= len(words) <= 3
        and all(w[0].isupper() for w in words if w)
        and not any(c.isdigit() for c in t)
        and len(t) < 40
        and sum(c.isalpha() for c in t) / max(len(t), 1) > 0.80
    ):
        return True
    return False


def _detect_section(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    if re.match(r"^[\-•·▪▸◦\*]\s+\w", stripped):
        return None
    for section, pattern in SECTION_PATTERNS.items():
        if pattern.search(stripped):
            return section
    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_resume(raw_text: str) -> List[ResumeChunk]:
    lines = raw_text.split("\n")
    chunks: List[ResumeChunk] = []
    chunk_index = 0

    current_section = "header"   # resume starts in header — gets skipped
    current_lines: List[str] = []

    def flush(section: str, lines_buf: List[str]) -> None:
        nonlocal chunk_index
        # FIX 1: skip the header section entirely — name/email/phone never stored
        if section == "header":
            return
        content = "\n".join(l for l in lines_buf if l.strip()).strip()
        if len(content) < 15:
            return
        chunks.append(ResumeChunk(
            chunk_id=f"{section}_{chunk_index}",
            section=section,
            content=content,
            metadata={"char_count": len(content)},
        ))
        chunk_index += 1

    for line in lines:
        detected = _detect_section(line)
        if detected and detected != current_section:
            flush(current_section, current_lines)
            current_section = detected
            current_lines = []
        else:
            # FIX 2: while in header, silently drop header-noise lines
            if current_section == "header" and _is_header_line(line):
                continue
            current_lines.append(line)

    flush(current_section, current_lines)

    # Extra pass: individual bullet items from skills section
    skills_chunks = [c for c in chunks if c.section == "skills"]
    for sc in skills_chunks:
        bullets = re.split(r"[\n,|/•·▪▸◦]+", sc.content)
        for b in bullets:
            b = b.strip()
            if 2 < len(b) < 60 and sum(c.isalpha() for c in b) / max(len(b), 1) > 0.5:
                chunks.append(ResumeChunk(
                    chunk_id=f"skill_item_{chunk_index}",
                    section="skills",
                    content=b,
                    metadata={"char_count": len(b), "type": "bullet"},
                ))
                chunk_index += 1

    return chunks


def extract_resume(pdf_path: str) -> List[ResumeChunk]:
    raw_text = extract_text_from_pdf(pdf_path)
    return chunk_resume(raw_text)


# ── SKILL EXTRACTION & NORMALIZATION ──────────────────────────────────────────
# ── CRITICAL: word-boundary safe normalization (FIXED) ──────────────────────
# These ONLY replace multi-word phrases and explicit variants, never single letters
NORMALIZATION_MAP = {
    "react.js": "react",
    "reactjs": "react",
    "react js": "react",
    "nodejs": "node.js",
    "node js": "node.js",
    "mongo db": "mongodb",
    "expressjs": "express",
    "express js": "express",
    "pythonprogramming": "python",
    "javaprogramming": "java",
    "sqldatabase": "sql",
    "nosqldatabase": "nosql",
    "awscloud": "aws",
    "azurecloud": "azure",
    "googlecloud": "gcp",
    "githubactions": "github actions",
    "gitlabci": "gitlab ci",
    "restful api": "rest api",
    "rest apis": "rest api",
    "api development": "rest api",
    "api design": "rest api",
    "typescriptlang": "typescript",
    "postgres sql": "postgresql",
    "postgre sql": "postgresql",
    "aws service": "aws",
}

CRITICAL_SKILLS = [
    "php", "flutter", ".net", "html", "css", "javascript",
    "rest api", "git", "database", "python", "node.js",
]

COMMON_SKILLS = [
    # Frontend
    "react", "angular", "vue", "svelte", "next.js",
    "html", "css", "javascript", "typescript",
    # Backend
    "node.js", "python", "java", "php", ".net", "go", "rust",
    "django", "flask", "fastapi", "spring", "express", "laravel",
    # Data & Visualization
    "numpy", "pandas", "tensorflow", "pytorch", "scikit-learn", "scikit learn",
    "matplotlib", "plotly", "seaborn", "bokeh",
    "streamlit", "power bi", "tableau", "looker", "qlik",
    "bi tools", "dashboarding", "visualization",
    # ML & Data Science
    "machine learning", "deep learning", "data science", "data analysis", "feature engineering",
    "data cleaning", "eda", "exploratory data analysis", "model deployment",
    # Databases
    "sql", "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "elasticsearch", "firebase", "database", "nosql",
    # Mobile
    "flutter", "react native", "swift", "kotlin",
    # DevOps & Infrastructure
    "docker", "kubernetes", "terraform", "jenkins",
    "aws", "azure", "gcp", "github actions", "gitlab ci", "ci/cd",
    # Tools & Methodologies
    "git", "github", "gitlab", "jira", "confluence",
    "agile", "scrum", "rest api", "graphql",
    # Soft Skills
    "report writing", "communication", "documentation",
    # Other
    "linux", "windows", "macos", "unix",
    "c++", "c#", "scala", "haskell",
]

DYNAMIC_SKILL_PATTERNS = [
    (r"\b(?:next|nuxt|vue|svelte|react)\.js\b", None),
    (r"\btypescript\b", None),
    (r"\bgraphql\b", None),
    (r"\bpostgres(?:ql)?\b", "postgresql"),
    (r"\bredis\b", None),
    (r"\bterraform\b", None),
    (r"\belasticsearch\b", None),
    (r"\bfirebase\b", None),
    (r"\bc\+\+\b", "c++"),
    (r"\bc#\b", "c#"),
    (r"\brust\b", None),
    (r"\bswift\b", None),
    (r"\bkotlin\b", None),
    (r"\b(api development|api design)\b", "rest api"),
]


def normalize_text(text: str) -> str:
    """Normalize text for skill extraction."""
    text = text.lower().strip()
    for k, v in NORMALIZATION_MAP.items():
        pattern = rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"
        text = re.sub(pattern, v, text)
    text = re.sub(r"[^a-z0-9\s\.\-#\+\/\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_dynamic_skills(text: str) -> List[str]:
    normalized = normalize_text(text)
    found = set()

    for pattern, replacement in DYNAMIC_SKILL_PATTERNS:
        matches = re.findall(pattern, normalized, re.I)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            match = match.lower().strip()
            if replacement:
                match = replacement
            match = normalize_text(match)
            if match:
                found.add(match)

    return sorted(found)


def clean_resume_skill(skill: str) -> str | None:
    """Clean resume skill by removing junk starters, long phrases, and non-technical terms."""
    skill = skill.lower().strip()
    # ❌ remove junk starters
    if skill.startswith(("and", "such", "with")):
        return None
    # ❌ too long → sentence
    if len(skill.split()) > 4:
        return None
    # ❌ no technical signal
    if not any(c.isalpha() for c in skill):
        return None
    return skill


def extract_skills(text: str) -> List[str]:
    """
    Extract structured skills from resume or JD text.
    Multi-stage extraction for reliability and completeness.
    
    Stages:
    1. Critical skills (must-haves)
    2. Common skills (exact phrase matching with word boundaries)
    3. Compound skill splitting (matplotlib/plotly, power bi or tableau)
    4. Dynamic patterns (edge cases like .net, c++, etc.)
    5. Requirements section focused extraction
    
    Returns sorted list of normalized skill names found in text.
    """
    if not text:
        return []

    text_lower = text.lower()
    text_normalized = normalize_text(text)
    found = set()

    # STAGE 1: Critical skills (exact + flexible matching)
    for skill in CRITICAL_SKILLS:
        skill_lower = skill.lower()
        if skill_lower in text_lower or skill_lower in text_normalized:
            found.add(skill_lower)

    # STAGE 2: Common skills (boundary-safe pattern matching)
    for skill in COMMON_SKILLS:
        skill_normalized = normalize_text(skill)
        pattern = r"\b" + re.escape(skill_normalized) + r"\b"
        if re.search(pattern, text_normalized):
            found.add(skill_normalized)

    # STAGE 3: Split compound skills (matplotlib/plotly, power bi or tableau)
    # Extract comma and slash-separated lists
    list_patterns = [
        r"([\w\s\+\-\.#]+)[,/]\s*([\w\s\+\-\.#]+)",  # Handles: Python, Java or Flask/Express
        r"([\w\s]+)\s+or\s+([\w\s]+)",  # Handles: Power BI or Tableau
    ]
    for pattern in list_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                for item in match:
                    skill = item.strip().lower()
                    if 2 <= len(skill) <= 50:  # Reasonable skill length
                        found.add(skill)
            else:
                skill = match.strip().lower()
                if 2 <= len(skill) <= 50:
                    found.add(skill)

    # STAGE 4: Dynamic patterns (for edge cases like .net, c++, etc.)
    for pattern, replacement in DYNAMIC_SKILL_PATTERNS:
        matches = re.findall(pattern, text_lower, re.I)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            match = match.lower().strip()
            if replacement:
                match = replacement
            if match and len(match) > 1:
                found.add(match)

    # STAGE 5: Extract from Requirements section for priority
    # Capture the most important part of the JD
    req_section = text_lower
    req_match = re.search(r"(requirement|skill)[s]?:(.+?)(?=\n\w+:|$)", req_section, re.IGNORECASE | re.DOTALL)
    if req_match:
        req_text = req_match.group(2)
        # Split by common delimiters: comma, semicolon, bullet points
        for delimiter_pattern in [r"[,;·]", r"\n\s*[-*]"]:
            chunks = re.split(delimiter_pattern, req_text)
            for chunk in chunks:
                chunk = chunk.strip().lower()
                # Short phrases (1-4 words) are likely skills
                words = chunk.split()
                if 1 <= len(words) <= 4 and 2 <= len(chunk) <= 50:
                    # Filter out common non-skills
                    if chunk not in {'and', 'or', 'the', 'a', 'an', 'is', 'are', 'strong', 'good', 'excellent', 'basic'}:
                        found.add(chunk)

    # Normalize all found skills
    found_normalized = set()
    for skill in found:
        normalized = normalize_text(skill)
        if normalized and len(normalized) > 1:
            found_normalized.add(normalized)

    # Apply resume skill cleaning (FIX 2)
    cleaned_skills = []
    for skill in found_normalized:
        cleaned = clean_resume_skill(skill)
        if cleaned:
            cleaned_skills.append(cleaned)

    return sorted(cleaned_skills)
