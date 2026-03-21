# SkillBridge AI Onboarding Engine Backend

This is the FastAPI backend for the SkillBridge AI-adaptive onboarding engine.

## Features
- **Real Skill Extraction**: Extracts skills from resume and JD using SkillNER and JobBERT.
- **LLM-Driven Mastery**: Uses Groq-powered Llama 3 for context-aware mastery scoring (0.0-1.0).
- **Domain-Aware Gap Analysis**: Computes skill gaps with domain filters (Technical vs. Operations).
- **Seniority & Negation Detection**: Intelligently handles "8 years experience" boosts and "not production-grade" caps.
- **Optimized Pathing**: Generates pathways using priority-aware Kahn's algorithm.
- **Skipped Node Visibility**: Explicitly identifies and shows courses bypassed due to existing mastery.
- **Stable JSON Interface**: Exports a `PipelineState` JSON for frontend rendering.

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
uv run python -m tests.test_kahn
uv run python -m tests.test_gap_analyzer
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

## Testing with Your Own Resume

To test the engine against your own resume and a specific job description:

1. **Prepare your files**: Save your resume and the target job description as plain `.txt` files.
2. **Configure API Key**: Ensure your `GROQ_API_KEY` is set in the `.env` file.
3. **Run the Real Pipeline Test**:
   ```bash
   uv run python test_real_pipeline.py --resume path/to/resume.txt --jd path/to/jd.txt --domain technical
   ```

### Command Line Options:
- `--resume`: (Required) Path to your resume text file.
- `--jd`: (Required) Path to the job description text file.
- `--domain`: (Optional) Set to `technical` or `operations` to filter which courses can be assigned.

The script will output a **Full Pathway Report**, including:
- **Skill Gap**: A list of missing skills identified from the JD.
- **Recommended Courses**: The optimized learning path to bridge the gap.
- **Skipped Courses**: Courses you already know that were bypassed (proof of efficiency).

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
