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

The project uses [uv](https://github.com/astral-sh/uv) for fast, deterministic dependency management.

### 1. Install uv
If you don't have `uv` installed, follow the [installation guide](https://github.com/astral-sh/uv#installation).

### 2. Create Environment and Install Dependencies
```bash
uv sync
```
This command automatically creates a virtual environment, syncs `uv.lock` with `pyproject.toml`, and installs all dependencies.

> **Note:** We use `pyproject.toml` and `uv.lock` as the primary source of truth for dependencies. Avoid using `requirements.txt` unless absolutely necessary (e.g., for legacy CI/CD environments).

### 3. Download Required Models
The skill extraction pipeline requires the large spaCy English model:
```bash
uv run python -m spacy download en_core_web_lg
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env` and add your Groq API key:
```bash
cp .env.example .env
```
Open `.env` and set:
`GROQ_API_KEY=your_actual_api_key_here`

### 5. Run the Server
```bash
uv run python -m app.main
```
The API will be available at `http://localhost:8000`. You can view the interactive documentation at `http://localhost:8000/docs`.

## Testing

Ensure you run tests through `uv run` to use the managed environment:

### Run All Unit Tests
```bash
uv run python -m pytest tests/
```

### Run NLP Pipeline Integration Test (Extractor)
```bash
uv run python -m experiments.test_nlp_pipeline
```

### Run Pathing Engine Experiment
```bash
uv run python -m experiments.test_kahn
```

### Run Full End-to-End Pipeline Verification
```bash
uv run python -m experiments.test_full_pipeline
```

## Maintenance: Keeping requirements.txt updated
For legacy compatibility, we keep an exported `requirements.txt`. To update it after changing dependencies in `pyproject.toml`:
```bash
uv lock && uv export --format requirements-txt > requirements.txt
```

## Implementation Details

### Pathing Algorithm
The engine uses a priority-aware version of Kahn's topological sort. Among nodes with zero in-degree (no remaining prerequisites), the node with the highest priority is chosen.

**Priority Function:**
`priority = (skill_gap_count * (1 - mastery_score)) / (estimated_hours + 1e-5)`

- `skill_gap_count`: Number of skills taught by the course that are in the user's gap.
- `mastery_score`: Average mastery score of the skills taught by the course (for the user).
- `estimated_hours`: Estimated time to complete the course.
