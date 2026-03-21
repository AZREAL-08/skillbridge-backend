"""
FastAPI entry point — SkillBridge v2.0
Implements both HR Admin and Candidate flows.
"""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Shared State Schemas ---
from app.state import SkillEntry, StoredJD, PipelineState, CurrentState, TargetState, Question, StoredState

# --- Track 2: Extractor ---
from app.extractor.extractor import extract_skills, extract_skills_from_jd

# --- Track 3: Pathing ---
from app.pathing.pathing import run_pipeline 
from app.pathing.gap_analyzer import compute_skill_gap
from app.catalog.loader import load_catalog

logger = logging.getLogger(__name__)

# ===========================================================================
# Hackathon Persistence (In-Memory Databases)
# ===========================================================================
DB_JDS: Dict[str, StoredJD] = {}
DB_SESSIONS: Dict[str, CurrentState] = {}
DB_STATES: Dict[str, StoredState] = {}

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
    domain: str
    raw_text: str
    required_skills: List[SkillEntry]
    department: str = ""

class ResumeConfirmRequest(BaseModel):
    confirmed_skills: List[SkillEntry]

class PathwayQuestionsRequest(BaseModel):
    state_id: str
    jd_id: str

class PathwayGenerateRequest(BaseModel):
    state_id: str
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
        "domain": request.domain, # type: ignore
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
async def confirm_resume(file: UploadFile = File(...), confirmed_skills_json: str = None):
    """Saves the candidate's confirmed skills and creates a persistent state with ID."""
    resume_content = await file.read()
    raw_text = resume_content.decode('utf-8')
    
    # Parse confirmed skills from JSON string if provided
    try:
        confirmed_skills = json.loads(confirmed_skills_json) if confirmed_skills_json else []
    except json.JSONDecodeError:
        confirmed_skills = []
    
    # Create state entry
    state_id = f"state_{uuid.uuid4().hex[:8]}"
    
    current_state: CurrentState = {
        "raw_resume_text": raw_text,
        "extracted_skills": confirmed_skills
    }
    
    stored_state: StoredState = {
        "state_id": state_id,
        "current": current_state,
        "created_at": datetime.now().isoformat()
    }
    
    DB_STATES[state_id] = stored_state
    DB_SESSIONS[state_id] = current_state
    
    return {"state_id": state_id, "current_state_id": state_id, "status": "confirmed"}

@app.post("/api/pathway/questions")
async def generate_questions(request: PathwayQuestionsRequest):
    """
    Generates dynamic preference questions based on the candidate's gap.
    """
    if request.state_id not in DB_STATES:
        raise HTTPException(status_code=404, detail="State not found")
    if request.jd_id not in DB_JDS:
        raise HTTPException(status_code=404, detail="JD not found")

    current_state = DB_SESSIONS[request.current_state_id]
    target_jd = DB_JDS[request.jd_id]
    catalog = load_catalog()

    # 1. Compute Gap to drive dynamic questions
    required_ids = [s["taxonomy_id"] for s in target_jd["required_skills"]]
    gap_ids = compute_skill_gap(
        current_state["extracted_skills"], 
        required_ids, 
        catalog, 
        target_jd["domain"]
    )

    # 2. Baseline Questions
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

    # 3. Dynamic Domain-Specific Questions
    # If the gap contains technical skills, ask about environment
    if target_jd["domain"] == "technical" and len(gap_ids) > 0:
        questions.append({
            "id": "preferred_os",
            "text": "Which development environment do you prefer for hands-on modules?",
            "options": ["Linux/Unix", "Windows (WSL)", "Cloud-based IDE"]
        })
    
    # If it's operations and has a gap, ask about tool preference
    if target_jd["domain"] == "operations" and len(gap_ids) > 0:
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
    """
    if request.state_id not in DB_STATES:
        raise HTTPException(status_code=404, detail="State not found")
    if request.jd_id not in DB_JDS:
        raise HTTPException(status_code=404, detail="JD not found")

    stored_state = DB_STATES[request.state_id]
    current_state = stored_state["current"]
    target_jd = DB_JDS[request.jd_id]

    try:
        # Run the full pipeline
        result = run_pipeline(
            resume_text=current_state["raw_resume_text"],
            jd_text=target_jd["raw_text"],
            preferences=request.preferences,
            domain_filter=target_jd["domain"]
        )
        return result
    except Exception as e:
        logger.error(f"Pathway generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
