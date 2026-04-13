"""
MCP tool definitions — these are the structured tools the LLM calls
to interact with ChromaDB. No raw text queries; everything is a typed tool call.
"""

from typing import Any, Callable, Dict, List
from vector_store import query_resume, query_jd, get_all_jd_requirements

# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS: Dict[str, Callable] = {}


def tool(name: str):
    """Decorator to register a function as an MCP tool."""
    def decorator(fn: Callable) -> Callable:
        TOOLS[name] = fn
        return fn
    return decorator


def call_tool(name: str, **kwargs) -> Any:
    """Dispatch a tool call by name."""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOLS.keys())}")
    return TOOLS[name](**kwargs)


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool("query_resume")
def tool_query_resume(query: str, n_results: int = 5, section: str = None) -> List[dict]:
    """
    Semantic search over the stored resume.
    Args:
        query:     Natural language query (e.g. "Python experience", "machine learning projects")
        n_results: How many chunks to return
        section:   Optional filter — "skills" | "experience" | "education" | "projects"
    Returns list of {content, section, similarity_score}
    """
    return query_resume(query_text=query, n_results=n_results, section_filter=section)


@tool("query_jd")
def tool_query_jd(query: str, n_results: int = 5) -> List[dict]:
    """
    Semantic search over the job description requirements.
    Args:
        query:     A skill or requirement to look for
        n_results: How many requirements to return
    Returns list of {requirement, similarity_score}
    """
    return query_jd(query_text=query, n_results=n_results)


@tool("get_all_requirements")
def tool_get_all_requirements() -> List[str]:
    """
    Returns every requirement from the job description.
    Used by Agent 3 to do a full ATS scan (checks every requirement against resume).
    """
    return get_all_jd_requirements()


@tool("compare_sections")
def tool_compare_sections(jd_requirement: str, resume_section: str = None) -> dict:
    """
    For a single JD requirement, find the best matching resume chunk.
    Args:
        jd_requirement:  The exact requirement string from the JD
        resume_section:  Optional section to narrow the search
    Returns {requirement, best_match_content, best_match_section, similarity_score}
    """
    results = query_resume(
        query_text=jd_requirement,
        n_results=1,
        section_filter=resume_section,
    )
    if not results:
        return {
            "requirement": jd_requirement,
            "best_match_content": None,
            "best_match_section": None,
            "similarity_score": 0.0,
        }
    top = results[0]
    return {
        "requirement": jd_requirement,
        "best_match_content": top["content"],
        "best_match_section": top["section"],
        "similarity_score": top["similarity_score"],
    }


# ── Tool manifest (for LLM system prompt) ────────────────────────────────────

def get_tool_manifest() -> str:
    """Returns a plain-text description of all tools for inclusion in LLM prompts."""
    return """
Available MCP tools:

1. query_resume(query, n_results=5, section=None)
   → Semantic search over resume. Returns matching chunks with similarity scores.

2. query_jd(query, n_results=5)
   → Semantic search over job description requirements.

3. get_all_requirements()
   → Returns every requirement from the job description. Use for full ATS scan.

4. compare_sections(jd_requirement, resume_section=None)
   → For one JD requirement, returns the single best-matching resume chunk + score.

All searches are purely semantic (context-based). No keyword or fuzzy matching.
""".strip()
