# SkillBridge AI Onboarding Engine Backend

This is the FastAPI backend for the SkillBridge AI-adaptive onboarding engine.

## Features
- Extracts skills from resume (mocked during development).
- Computes skill gap against target Job Description.
- Generates an optimized learning pathway from a course catalog.
- Uses priority-aware Kahn's algorithm for topological sorting.
- Provides human-readable reasoning traces.
- Exports a stable `PipelineState` JSON for frontend rendering.

## Tech Stack
- FastAPI
- NetworkX (for DAG operations)
- Pydantic
- Python 3.10+

## Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn networkx pydantic
```

2. Run the server:
```bash
python -m app.main
```
The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /api/analyze
Analyzes a resume and job description.

**Request Body:**
```json
{
  "resume_text": "...",
  "jd_text": "..."
}
```

**Response:**
Returns a `PipelineState` object matching the frontend contract.

## Testing

Run unit tests:
```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```

Run isolated DAG experiment:
```bash
python experiments/test_kahn.py
```

Run end-to-end pipeline verification:
```bash
export PYTHONPATH=$PYTHONPATH:.
python experiments/test_full_pipeline.py
```

## Implementation Details

### Pathing Algorithm
The engine uses a priority-aware version of Kahn's topological sort. Among nodes with zero in-degree (no remaining prerequisites), the node with the highest priority is chosen.

**Priority Function:**
`priority = (skill_gap_count * (1 - mastery_score)) / (estimated_hours + 1e-5)`

- `skill_gap_count`: Number of skills taught by the course that are in the user's gap.
- `mastery_score`: Average mastery score of the skills taught by the course (for the user).
- `estimated_hours`: Estimated time to complete the course.
