"""
FastAPI entry point — SkillBridge v2.0
Implements both HR Admin and Candidate flows.
"""

import io
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber

# --- Shared State Schemas ---
from app.state import SkillEntry, StoredJD, PipelineState, CurrentState, TargetState, Question

# --- Track 2: Extractor ---
from app.extractor.extractor import extract_skills, extract_skills_from_jd

# --- Track 3: Pathing ---
from app.pathing.dag_builder import build_dag
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph
from app.pathing.kahn import kahn_priority_sort
from app.pathing.tracer import generate_reasoning_trace
from app.pathing.pathing import add_skipped_nodes
from app.catalog.loader import load_catalog
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ===========================================================================
# Hackathon Persistence (In-Memory Databases)
# ===========================================================================
DB_JDS: Dict[str, StoredJD] = {}
DB_SESSIONS: Dict[str, CurrentState] = {}

# ===========================================================================
# Pydantic Request Schemas
# ===========================================================================
class JDUploadRequest(BaseModel):
    raw_text: str
    role_title: str
    company: str
    domain: str
    department: str = ""

class JDConfirmRequest(BaseModel):
    role_title: str
    company: str
    # FIX 6: Enforce correct Literal values — "technical" or "operational" only
    domain: Literal["technical", "operational"]
    raw_text: str
    required_skills: List[SkillEntry]
    department: str = ""

class ResumeConfirmRequest(BaseModel):
    raw_text: str
    confirmed_skills: List[SkillEntry]

class PathwayQuestionsRequest(BaseModel):
    current_state_id: str
    jd_id: str

class PathwayGenerateRequest(BaseModel):
    current_state_id: str
    jd_id: str
    preferences: Dict[str, Any]

# ===========================================================================
# PDF Text Extraction
# ===========================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract plain text from a resume PDF using pdfplumber.

    Strategy:
      - Iterates every page and collects text from page.extract_text().
      - Pages are separated by a double newline so section boundaries
        (e.g. end of Experience, start of Education) are preserved for
        the downstream SkillNER / JobBERT models.
      - If pdfplumber returns nothing for a page (scanned/image-only PDF),
        that page is silently skipped rather than crashing. The caller
        receives whatever text was extractable.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        Concatenated plain-text string. Raises HTTPException 400 if the
        PDF is completely empty (no extractable text on any page).
    """
    pages_text: List[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages_text.append(page_text.strip())
            else:
                logger.debug(
                    "[PDFExtract] Page %d returned no text (image-only or blank).",
                    page_num,
                )

    if not pages_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No extractable text found in the uploaded PDF. "
                "The file may be a scanned image. "
                "Please upload a text-based PDF or paste your resume as plain text."
            ),
        )

    full_text = "\n\n".join(pages_text)
    logger.info(
        "[PDFExtract] Extracted %d chars across %d pages.",
        len(full_text), len(pages_text),
    )
    return full_text


# ===========================================================================
# App Setup
# ===========================================================================
app = FastAPI(title="SkillBridge Engine v2.0", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/api/catalog")
async def get_catalog():
    """
    Returns the full course catalog with display metadata.
    Used by the frontend to resolve course_id → title, hours, difficulty, bloom_level.
    """
    catalog = load_catalog()
    return [
        {
            "course_id":       c["course_id"],
            "title":           c.get("title", c["course_id"]),
            "description":     c.get("description", ""),
            "estimated_hours": c.get("estimated_hours", 0),
            "difficulty":      c.get("difficulty", "Intermediate"),
            "bloom_level":     c.get("bloom_level", 3),
            "domain":          c.get("domain", "technical"),
            "prerequisites":   c.get("prerequisites", []),
            "skills_taught":   c.get("skills_taught", []),
            "skill_labels":    c.get("skill_labels", {}),
        }
        for c in catalog
    ]

# ===========================================================================
# 4.1 HR Management Flow
# ===========================================================================

@app.post("/api/jd/upload")
async def upload_jd(request: JDUploadRequest):
    """
    Runs extraction on JD text.
    Per Blueprint v2.0, mastery step is skipped for JDs (forced to 0.0).
    """
    draft_skills = extract_skills_from_jd(request.raw_text)
    return {"draft_skills": draft_skills}

@app.post("/api/jd/confirm")
async def confirm_jd(request: JDConfirmRequest):
    """Saves the HR-confirmed JD to the database."""
    jd_id = f"jd_{uuid.uuid4().hex[:8]}"

    new_jd: StoredJD = {
        "jd_id": jd_id,
        "role_title": request.role_title,
        "company": request.company,
        "department": request.department,
        "domain": request.domain,
        "raw_text": request.raw_text,
        "required_skills": request.required_skills,
        "created_at": datetime.now().isoformat()
    }

    DB_JDS[jd_id] = new_jd
    return {"jd_id": jd_id, "status": "confirmed"}

@app.get("/api/jd/list")
async def list_jds():
    """Returns all confirmed JDs for the Candidate dropdown."""
    return [
        {
            "jd_id": jd["jd_id"],
            "role_title": jd["role_title"],
            "company": jd["company"],
            "domain": jd["domain"]
        }
        for jd in DB_JDS.values()
    ]

# FIX 5: Add missing GET /api/jd/{jd_id} endpoint required by frontend Step 3
@app.get("/api/jd/{jd_id}")
async def get_jd(jd_id: str):
    """Returns full StoredJD including confirmed required_skills."""
    if jd_id not in DB_JDS:
        raise HTTPException(status_code=404, detail="JD not found")
    return DB_JDS[jd_id]

# ===========================================================================
# 4.2 Candidate Flow (Resume Processing)
# ===========================================================================

@app.post("/api/resume/upload")
async def upload_resume(
    file: Optional[UploadFile] = File(default=None),
    raw_text: Optional[str]    = Form(default=None),
):
    """
    Unified resume upload endpoint — accepts either:
      - A PDF file via multipart/form-data   (field name: "file")
      - Raw plain text via multipart/form-data (field name: "raw_text")

    Both paths run the full Track 2 pipeline:
    SkillNER → JobBERT → LLM filter → Groq mastery scoring.

    Returns {"extracted_skills": [...], "raw_text": "..."} in both cases.
    raw_text is always returned so the frontend can pass it to
    /api/resume/confirm without re-uploading.

    Frontend usage — PDF:
        const form = new FormData();
        form.append("file", pdfFile);
        fetch("/api/resume/upload", { method: "POST", body: form });

    Frontend usage — plain text:
        const form = new FormData();
        form.append("raw_text", pastedText);
        fetch("/api/resume/upload", { method: "POST", body: form });
    """
    resume_text: str = ""

    if file is not None and file.filename:
        # ── PDF path ─────────────────────────────────────────────────────────
        content_type = file.content_type or ""
        if content_type not in ("application/pdf", "application/octet-stream", ""):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{content_type}'. Please upload a PDF.",
            )

        pdf_bytes = await file.read()

        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file does not appear to be a valid PDF.",
            )

        logger.info(
            "[ResumeUpload] Received PDF '%s' (%d bytes).",
            file.filename, len(pdf_bytes),
        )
        resume_text = extract_text_from_pdf(pdf_bytes)

    elif raw_text and raw_text.strip():
        # ── Plain text path ───────────────────────────────────────────────────
        resume_text = raw_text.strip()
        logger.info(
            "[ResumeUpload] Received raw text (%d chars).", len(resume_text)
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a PDF file (field: 'file') or plain text (field: 'raw_text').",
        )

    catalog = load_catalog()
    extracted_skills = extract_skills(resume_text, catalog)
    return {"extracted_skills": extracted_skills, "raw_text": resume_text}

@app.post("/api/resume/confirm")
async def confirm_resume(request: ResumeConfirmRequest):
    """Saves the candidate's confirmed skills into a temporary session."""
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    DB_SESSIONS[session_id] = {
        "raw_resume_text": request.raw_text,
        "extracted_skills": request.confirmed_skills
    }

    return {"current_state_id": session_id}

# ===========================================================================
# 4.3 Pathway Generation (The Core Engine)
# ===========================================================================

@app.post("/api/pathway/questions")
async def generate_questions(request: PathwayQuestionsRequest):
    """
    Generates dynamic preference questions based on the actual content of the
    candidate's skill gap — not just domain and gap size.

    After computing gap_ids (a list of EMSI taxonomy ID strings), we resolve
    them back to labels using the confirmed JD skill entries, then inspect
    which courses in the catalog teach those gap skills to understand the
    character of the gap (infra-heavy, data-heavy, tool-specific, etc.).
    """
    if request.current_state_id not in DB_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    if request.jd_id not in DB_JDS:
        raise HTTPException(status_code=404, detail="JD not found")

    current_state = DB_SESSIONS[request.current_state_id]
    target_jd     = DB_JDS[request.jd_id]
    catalog       = load_catalog()
    domain        = target_jd["domain"]

    # Build a lookup from taxonomy_id → label using the confirmed JD skill entries
    jd_skill_label: Dict[str, str] = {
        s["taxonomy_id"]: s["label"]
        for s in target_jd["required_skills"]
    }

    required_ids = list(jd_skill_label.keys())
    gap_ids: List[str] = compute_skill_gap(
        current_state["extracted_skills"],
        required_ids,
        catalog,
        domain,
    )

    # Resolve gap IDs to lowercase labels for keyword matching
    gap_labels = {
        jd_skill_label.get(gid, "").lower()
        for gid in gap_ids
        if jd_skill_label.get(gid)
    }

    # Resolve gap IDs to the courses that teach them — gives us bloom levels,
    # difficulty, and domain sub-categories to ask smarter questions
    gap_course_bloom: List[int] = []
    gap_course_difficulties: List[str] = []
    for course in catalog:
        if set(course.get("skills_taught", [])).intersection(gap_ids):
            gap_course_bloom.append(int(course.get("bloom_level", 3)))
            gap_course_difficulties.append(
                course.get("difficulty", "Intermediate").lower()
            )

    has_gap         = len(gap_ids) > 0
    has_expert_gap  = any(b >= 4 for b in gap_course_bloom)   # bloom 4-6 in gap
    has_foundational_gap = any(b <= 2 for b in gap_course_bloom)  # bloom 1-2 in gap

    # Keyword sets for gap content detection
    infra_keywords  = {"docker", "kubernetes", "linux", "bash", "shell", "ci/cd",
                       "jenkins", "ansible", "terraform", "devops", "cloud"}
    data_keywords   = {"sql", "postgresql", "mysql", "mongodb", "data",
                       "analytics", "pipeline", "etl", "spark", "hadoop"}
    erp_keywords    = {"sap", "erp", "oracle", "supply chain", "inventory",
                       "procurement", "warehouse", "logistics"}
    ml_keywords     = {"machine learning", "deep learning", "pytorch", "tensorflow",
                       "nlp", "computer vision", "model", "neural"}

    gap_has_infra = bool(gap_labels & infra_keywords) or any(
        kw in lbl for lbl in gap_labels for kw in infra_keywords
    )
    gap_has_data  = bool(gap_labels & data_keywords) or any(
        kw in lbl for lbl in gap_labels for kw in data_keywords
    )
    gap_has_erp   = bool(gap_labels & erp_keywords) or any(
        kw in lbl for lbl in gap_labels for kw in erp_keywords
    )
    gap_has_ml    = bool(gap_labels & ml_keywords) or any(
        kw in lbl for lbl in gap_labels for kw in ml_keywords
    )

    # ── Baseline questions (always present) ──────────────────────────────────
    questions: List[Question] = [
        {
            "id":      "weekly_hours",
            "text":    "How much time can you dedicate to learning per week?",
            "options": ["1-3 hours", "4-6 hours", "7+ hours"],
        },
        {
            "id":      "learning_style",
            "text":    "What is your preferred learning style?",
            "options": ["Hands-on projects", "Structured reading", "Video lectures"],
        },
    ]

    # ── Dynamic questions based on gap content ────────────────────────────────

    # Depth preference — only meaningful when the gap spans both foundational
    # and advanced courses, so the candidate can choose where to start
    if has_gap and has_foundational_gap and has_expert_gap:
        questions.append({
            "id":      "depth_preference",
            "text":    "Your gap includes both foundational and advanced topics. Where would you like to start?",
            "options": ["Build foundations first", "Jump straight to advanced topics"],
        })

    # Infrastructure / DevOps environment question
    if domain == "technical" and has_gap and gap_has_infra:
        questions.append({
            "id":      "preferred_os",
            "text":    "Several gap topics involve hands-on infrastructure work. Which environment will you use?",
            "options": ["Linux/Unix", "Windows (WSL)", "Cloud-based IDE"],
        })

    # Data / SQL tooling question
    if domain == "technical" and has_gap and gap_has_data:
        questions.append({
            "id":      "data_tool_preference",
            "text":    "Your gap includes data or database skills. Which environment are you most comfortable working in?",
            "options": ["PostgreSQL / MySQL", "Cloud data warehouse (BigQuery, Redshift)", "No preference"],
        })

    # ML framework preference
    if domain == "technical" and has_gap and gap_has_ml:
        questions.append({
            "id":      "ml_framework",
            "text":    "Your gap includes machine learning topics. Which framework do you prefer?",
            "options": ["PyTorch", "TensorFlow / Keras", "No preference — teach me either"],
        })

    # ERP / operations tool question
    if domain == "operational" and has_gap and gap_has_erp:
        questions.append({
            "id":      "erp_familiarity",
            "text":    "Your gap includes ERP or supply chain modules. Which system are you most familiar with?",
            "options": ["SAP", "Oracle", "No prior ERP experience"],
        })

    # Generic operational gap with no specific ERP signal
    if domain == "operational" and has_gap and not gap_has_erp:
        questions.append({
            "id":      "ops_focus",
            "text":    "Which operational area do you want to prioritise first?",
            "options": ["Process & workflow", "Compliance & safety", "Team & people management"],
        })

    logger.info(
        "[Questions] %d questions generated for gap of %d skills (domain=%s).",
        len(questions), len(gap_ids), domain,
    )
    return {"questions": questions}


@app.post("/api/pathway/generate")
async def generate_pathway(request: PathwayGenerateRequest):
    """
    Runs Kahn's Algorithm + Priority Queue and returns the final PipelineState.

    FIX 1: Uses pre-confirmed skills from session and stored JD — does NOT
    re-run NLP extraction. This ensures the pathway matches what the candidate
    and HR admin actually reviewed and confirmed.
    """
    if request.current_state_id not in DB_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    if request.jd_id not in DB_JDS:
        raise HTTPException(status_code=404, detail="JD not found")

    current_state = DB_SESSIONS[request.current_state_id]
    target_jd = DB_JDS[request.jd_id]
    preferences = request.preferences

    try:
        catalog = load_catalog()
        G = build_dag(catalog)

        # Use confirmed skills — no re-extraction
        extracted_skills = current_state["extracted_skills"]
        required_skills = [s["taxonomy_id"] for s in target_jd["required_skills"]
                           if s.get("taxonomy_source") == "emsi"]
        domain_filter = target_jd["domain"]

        # Gap analysis on confirmed data
        skill_gap = compute_skill_gap(
            extracted_skills, required_skills, catalog, domain_filter
        )

        # Active subgraph
        node_states = get_active_subgraph(G, skill_gap, extracted_skills, domain_filter)
        active_node_ids = list(node_states.keys())

        # Build course metadata for Kahn sort — untouched mastery from confirmed skills
        course_metadata = {}
        gap_set = set(skill_gap)
        for cid in active_node_ids:
            meta = G.nodes[cid]
            taught = meta.get("skills_taught", [])
            gap_count = len([s for s in taught if s in gap_set])

            masteries = []
            for tid in taught:
                m = 0.0
                for s in extracted_skills:
                    if s["taxonomy_id"] == tid:
                        m = float(s.get("mastery_score", 0))
                        break
                masteries.append(m)
            avg_mastery = sum(masteries) / len(masteries) if masteries else 0.0

            course_metadata[cid] = {
                "prerequisites": meta.get("prerequisites", []),
                "gap_count":     gap_count,
                "mastery":       avg_mastery,
                "hours":         float(meta.get("estimated_hours", 1.0)),
                "bloom_level":   int(meta.get("bloom_level", 3)),
            }

        sorted_ids = kahn_priority_sort(active_node_ids, course_metadata)

        # Build pathway + traces
        final_pathway = []
        reasoning_traces = []
        assigned_count = 0
        total_hours = 0.0
        saved_hours = 0.0

        for cid in sorted_ids:
            state = node_states[cid]
            meta = G.nodes[cid]
            taught = meta.get("skills_taught", [])
            course_hours = float(meta.get("estimated_hours", 0.0))

            course_masteries = []
            course_confidences = []
            for tid in taught:
                m, c = 0.0, 0.5
                for s in extracted_skills:
                    if s["taxonomy_id"] == tid:
                        m = float(s.get("mastery_score", 0))
                        c = float(s.get("confidence_score", 0.5))
                        break
                course_masteries.append(m)
                course_confidences.append(c)

            avg_mastery = sum(course_masteries) / len(course_masteries) if course_masteries else 0.0
            avg_confidence = sum(course_confidences) / len(course_confidences) if course_confidences else 0.7

            final_pathway.append({
                "course_id": cid,
                "node_state": state,
                "mastery_score": avg_mastery,
                "confidence_score": avg_confidence
            })

            if state != "skipped":
                assigned_count += 1
                total_hours += course_hours
                dep_title = ""
                if state == "prerequisite":
                    for succ in G.successors(cid):
                        if succ in sorted_ids and node_states[succ] != "skipped":
                            dep_title = G.nodes[succ].get("title", succ)
                            break
                trace = generate_reasoning_trace(
                    cid, state, meta, skill_gap, extracted_skills, dep_title
                )
                reasoning_traces.append(trace)
            else:
                saved_hours += course_hours

        # Second pass: add mastered JD-required skipped nodes not in active subgraph
        final_pathway = add_skipped_nodes(
            final_pathway, extracted_skills, required_skills, catalog
        )

        # FIX 2: Compute total_hours and saved_hours for MetricsBar
        baseline_courses = max(len(catalog), 30)
        reduction_pct = 0.0
        if baseline_courses > 0:
            reduction_pct = ((baseline_courses - assigned_count) / baseline_courses) * 100

        metrics = {
            "baseline_courses": baseline_courses,
            "assigned_courses": assigned_count,
            "reduction_pct": round(reduction_pct, 2),
            "total_hours": round(total_hours, 1),
            "saved_hours": round(saved_hours, 1),
        }

        # FIX 3: Include preference_questions in PipelineState
        # Re-derive the questions that were shown to this candidate
        questions: List[Question] = [
            {
                "id": "weekly_hours",
                "text": "How much time can you dedicate to learning per week?",
                "options": ["1-3 hours", "4-6 hours", "7+ hours"]
            },
            {
                "id": "learning_style",
                "text": "What is your preferred learning style?",
                "options": ["Hands-on projects", "Structured reading", "Video lectures"]
            }
        ]
        if domain_filter == "technical" and len(skill_gap) > 0:
            questions.append({
                "id": "preferred_os",
                "text": "Which development environment do you prefer for hands-on modules?",
                "options": ["Linux/Unix", "Windows (WSL)", "Cloud-based IDE"]
            })
        if domain_filter == "operational" and len(skill_gap) > 0:
            questions.append({
                "id": "tool_preference",
                "text": "For supply chain modules, which ERP interface are you most familiar with?",
                "options": ["SAP", "Oracle", "No preference"]
            })

        result: PipelineState = {
            "current": {
                "raw_resume_text": current_state["raw_resume_text"],
                "extracted_skills": extracted_skills
            },
            "target": {
                "raw_jd_text": target_jd["raw_text"],
                "required_skills": required_skills
            },
            "skill_gap": skill_gap,
            "final_pathway": final_pathway,
            "reasoning_trace": reasoning_traces,
            "metrics": metrics,
            "preference_questions": questions,
        }

        return result

    except Exception as e:
        logger.error(f"Pathway generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)