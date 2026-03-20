from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.pathing.pathing import run_pipeline
from app.state import PipelineState

app = FastAPI(title="SkillBridge AI Onboarding Engine")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str

@app.post("/api/analyze", response_model=PipelineState)
async def analyze(request: AnalyzeRequest):
    """
    Analyzes resume and JD to compute skill gap and generate a learning pathway.
    """
    result = run_pipeline(request.resume_text, request.jd_text)
    return result

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
