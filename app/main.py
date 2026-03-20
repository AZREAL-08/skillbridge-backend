"""
FastAPI entry point — /api/analyze
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.state import PipelineState

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="SkillBridge Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
def analyze(body: AnalyzeRequest) -> PipelineState:
    """
    Full pipeline: extract_skills → compute_gap → run_kahn → generate_traces
    Returns the complete PipelineState JSON.
    """
    # TODO: wire up extractor and pathing modules
    raise NotImplementedError("Pipeline not yet wired up")
