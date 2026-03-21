"""
FastAPI entry point — SkillBridge v2.0
Implements both HR Admin and Candidate flows.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Shared State Schemas ---
from app.state import SkillEntry, StoredJD, PipelineState, CurrentState, TargetState, Question

# --- Track 2: Extractor ---
from app.extractor.extractor import extract_skills

# --- Track 3: Pathing (To be integrated by your friend) ---
# from app.pathing.pathing import run_pipeline 

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
    domain: str
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
    draft_skills = extract_skills(request.raw_text)
    
    # Force mastery to 0.0 because this is a target requirement, not a user profile
    for skill in draft_skills:
        skill["mastery_score"] = 0.0
        
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
    extracted_skills = extract_skills(request.raw_text)
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

    # Mock questions to unblock the frontend until Track 3 finishes gap logic
    mock_questions: List[Question] = [
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
    return {"questions": mock_questions}

@app.post("/api/pathway/generate")
async def generate_pathway(request: PathwayGenerateRequest):
    """
    Runs Kahn's Algorithm + Priority Queue and returns the final PipelineState.
    """
    if request.current_state_id not in DB_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    if request.jd_id not in DB_JDS:
        raise HTTPException(status_code=404, detail="JD not found")

    current_state = DB_SESSIONS[request.current_state_id]
    target_jd = DB_JDS[request.jd_id]

    target_state: TargetState = {
        "raw_jd_text": target_jd["raw_text"],
        "required_skills": [skill["taxonomy_id"] for skill in target_jd["required_skills"]]
    }

    # ---------------------------------------------------------
    # ⚠️ TRACK 3 HANDOFF POINT ⚠️
    # Once your friend finishes pathing.py, uncomment this:
    # 
    # result = run_pipeline(
    #     current_state=current_state,
    #     target_state=target_state,
    #     preferences=request.preferences,
    #     domain_filter=target_jd["domain"] # New P0 Fix
    # )
    # return result
    # ---------------------------------------------------------
    
    return {"status": "Waiting on Track 3 Algorithm Integration"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)