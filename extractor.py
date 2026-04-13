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
