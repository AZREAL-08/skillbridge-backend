"""
FastAPI entry point — SkillBridge v2.0
Implements both HR Admin and Candidate flows.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Shared State Schemas ---
from app.state import SkillEntry, StoredJD, PipelineState, CurrentState, TargetState, Question

# --- Track 2: Extractor ---
from app.extractor.extractor import extract_skills, extract_skills_from_jd

# --- Track 3: Pathing ---
from app.pathing.dag_builder import build_dag
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph, MASTERY_THRESHOLD
from app.pathing.kahn import kahn_priority_sort, compute_priority
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

class ResumeUploadRequest(BaseModel):
    raw_text: str

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
async def upload_resume(request: ResumeUploadRequest):
    """Runs full Track 2 pipeline (including Groq mastery scoring)."""
    catalog = load_catalog()
    extracted_skills = extract_skills(request.raw_text, catalog)
    return {"extracted_skills": extracted_skills}

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
    Generates dynamic preference questions based on the candidate's gap.
    """
    if request.current_state_id not in DB_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    if request.jd_id not in DB_JDS:
        raise HTTPException(status_code=404, detail="JD not found")

    current_state = DB_SESSIONS[request.current_state_id]
    target_jd = DB_JDS[request.jd_id]
    catalog = load_catalog()

    # Use confirmed JD skill IDs directly — no re-extraction
    required_ids = [s["taxonomy_id"] for s in target_jd["required_skills"]]
    gap_ids = compute_skill_gap(
        current_state["extracted_skills"],
        required_ids,
        catalog,
        target_jd["domain"]
    )

    # Baseline questions
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

    # FIX 6 (downstream): domain is now always "technical" or "operational"
    if target_jd["domain"] == "technical" and len(gap_ids) > 0:
        questions.append({
            "id": "preferred_os",
            "text": "Which development environment do you prefer for hands-on modules?",
            "options": ["Linux/Unix", "Windows (WSL)", "Cloud-based IDE"]
        })

    if target_jd["domain"] == "operational" and len(gap_ids) > 0:
        questions.append({
            "id": "tool_preference",
            "text": "For supply chain modules, which ERP interface are you most familiar with?",
            "options": ["SAP", "Oracle", "No preference"]
        })

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

        # FIX 4: Apply preference multipliers to priority before Kahn's runs
        weekly_hours_str = preferences.get("weekly_hours", "4-6 hours")
        learning_style = preferences.get("learning_style", "")

        def preference_multiplier(cid: str) -> float:
            meta = G.nodes[cid]
            bloom = int(meta.get("bloom_level", 3))
            mult = 1.0
            # Weekly hours: deprioritise long courses when time is tight
            if weekly_hours_str == "1-3 hours":
                course_hours = float(meta.get("estimated_hours", 1.0))
                mult *= 1.0 / (course_hours / 5.0 + 1.0)
            # Learning style: boost applied/advanced for hands-on, foundational for reading
            if learning_style == "Hands-on projects" and bloom >= 3:
                mult *= 1.2
            elif learning_style == "Structured reading" and bloom <= 2:
                mult *= 1.1
            # Domain-specific context boost
            for pref_key, pref_val in preferences.items():
                if pref_key not in ("weekly_hours", "learning_style") and pref_val:
                    if meta.get("domain") == domain_filter:
                        mult *= 1.15
                        break  # Apply boost once per course
            return mult

        # Build metadata with preference-adjusted priorities
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
            hours = float(meta.get("estimated_hours", 1.0))

            # Base priority scaled by preference multiplier
            base_priority = compute_priority(gap_count, avg_mastery, hours)
            adjusted_priority = base_priority * preference_multiplier(cid)

            course_metadata[cid] = {
                "prerequisites": meta.get("prerequisites", []),
                "gap_count": gap_count,
                "mastery": avg_mastery,
                "hours": hours,
                "bloom_level": int(meta.get("bloom_level", 3)),
                "_adjusted_priority": adjusted_priority,
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