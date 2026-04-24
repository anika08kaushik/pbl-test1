from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

app = FastAPI(title="PBL Test1 API")

# Allow the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/clear")
async def clear():
    from vector_store import clear_collections
    clear_collections()
    return {"ok": True}


@app.post("/analyze")
async def analyze(resume: UploadFile = File(...), jd_text: str = Form(...), jd_title: str = Form("")):
    # Save uploaded resume to a temp file and run existing pipeline
    from vector_store import (
        store_resume_chunks,
        store_jd_requirements_tagged,
        clear_collections,
    )
    from extractor import extract_resume
    from main import _extract_jd_requirements, _extract_resume_skill_items
    import agent_3_validator, agent_2_evaluator, agent_1_interviewer, agent_4_assessor

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(resume.filename).suffix) as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    try:
        clear_collections()
        chunks = extract_resume(tmp_path)
        store_resume_chunks(chunks)
        jd_items = _extract_jd_requirements(jd_text)
        resume_items = _extract_resume_skill_items(chunks)
        store_jd_requirements_tagged(jd_items, resume_items, jd_title or "Target Role")

        ats = agent_3_validator.run(jd_text)
        evaluation = agent_2_evaluator.run(ats)
        questions = agent_1_interviewer.run(ats, evaluation)
        assessment = agent_4_assessor.generate_assessment(ats)

        # Convert pydantic models to dicts when possible
        result = {
            "ats_result": ats.dict() if hasattr(ats, "dict") else ats,
            "evaluation": evaluation.dict() if hasattr(evaluation, "dict") else evaluation,
            "questions": questions.dict() if hasattr(questions, "dict") else questions,
            "assessment": assessment.dict() if hasattr(assessment, "dict") else assessment,
        }
        return JSONResponse(content=result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/query_resume")
async def query_resume(query: str = Form(...), k: int = Form(5), section: str = Form(None)):
    from vector_store import query_resume
    res = query_resume(query_text=query, n_results=k, section_filter=section)
    return {"results": res}
