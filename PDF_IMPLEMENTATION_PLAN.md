# SkillBridge PDF Implementation Plan

## PDFs reviewed

- `69bc3fa4be78c_ARTPARK_CodeForge_Hackathon.pdf`
- `AI Onboarding Engine Research Plan.pdf`
- `Master Architecture Blueprint (v1.5).pdf`
- `Master Architecture Blueprint (v1.5)-1.pdf`
- `Task Division & Timeline.pdf`
- `task.pdf`

## Source-of-truth hierarchy

- `Master Architecture Blueprint (v1.5).pdf` is the main implementation contract.
- `Task Division & Timeline.pdf` defines ownership, mock-first development, and integration timing.
- `task.pdf` is a duplicate of `Task Division & Timeline.pdf`.
- `Master Architecture Blueprint (v1.5)-1.pdf` appears to duplicate `Master Architecture Blueprint (v1.5).pdf`.
- `AI Onboarding Engine Research Plan.pdf` provides technical rationale, algorithm choices, and expected behavior.
- `69bc3fa4be78c_ARTPARK_CodeForge_Hackathon.pdf` defines judging criteria and submission requirements.

## What the project needs to do

Build an AI-adaptive onboarding engine that:

- accepts a resume and a target job description,
- extracts and normalizes skills to ESCO URIs,
- computes the candidate's skill gap against the role,
- generates an optimized learning pathway from a fixed course catalog,
- exposes the result through a FastAPI backend,
- returns a structured `PipelineState` JSON the frontend can render,
- includes reasoning traces and reduction metrics,
- stays grounded to the provided catalog with no hallucinated courses.

## Required backend contract from the PDFs

### Shared schema

`app/state.py` must define at least:

- `SkillEntry`
  - `esco_uri: str`
  - `label: str`
  - `mastery_score: float`
  - `confidence_score: float`
- `PathwayCourse`
  - `course_id: str`
  - `node_state: Literal["skipped", "assigned", "prerequisite"]`
  - `mastery_score: float`
  - `confidence_score: float`
- `Metrics`
  - `baseline_courses: int`
  - `assigned_courses: int`
  - `reduction_pct: float`
- `CurrentState`
  - `raw_resume_text: str`
  - `extracted_skills: List[SkillEntry]`
- `TargetState`
  - `raw_jd_text: str`
  - `required_skills: List[str]`
- `PipelineState`
  - `current: CurrentState`
  - `target: TargetState`
  - `skill_gap: List[str]`
  - `final_pathway: List[PathwayCourse]`
  - `reasoning_trace: List[str]`
  - `metrics: Metrics`

### Backend API

`app/main.py` must expose:

- `POST /api/analyze`
- request body:
  - `resume_text: str`
  - `jd_text: str`
- CORS for `http://localhost:3000`
- return type shaped like `PipelineState`

### Pathing algorithm requirements

From the blueprint and research plan:

- skill gap is set subtraction:
  - `required_skills - mastered_skills`
- mastery threshold:
  - `mastery_score >= 0.85` means skill is treated as mastered and can bypass courses
- catalog is a DAG:
  - each course has `course_id`, `skills_taught`, `prerequisites`, `difficulty`, `estimated_hours`, `domain`
- candidate courses are those whose `skills_taught` intersects the active skill gap
- prerequisite courses must be pulled in recursively unless already effectively mastered
- final ordering must use Kahn's topological sort
- among zero-indegree nodes, choose by max priority using a heap:
  - `priority = (skill_gap_count * (1 - mastery_score)) / (estimated_hours + 1e-5)`
- reasoning trace must explain why each non-skipped course was included
- metrics must report baseline size, assigned course count, and reduction percentage

## Team/process constraints from the PDFs

- `experiments/` is sandbox only and must never be imported by `app/`
- Person 1 owns `data/catalog.json` and `app/catalog/`
- Person 2 owns `app/extractor/`
- Person 3 owns `app/pathing/` and backend API composition
- Person 4 builds frontend against mock backend output
- Pathing and frontend should use mocks until extractor integration is explicitly ready
- final integration sequence:
  - extractor becomes real first
  - pathing swaps from mock extraction to real extraction
  - backend acceptance test passes
  - frontend flips off its mock toggle

## Current repo status

Observed in this repo:

- `app/main.py` is a stub
- `app/state.py` is a stub
- `app/pathing/dag_builder.py` is a stub
- `app/pathing/gap_analyzer.py` is a stub
- `app/pathing/kahn.py` is a stub
- `app/pathing/tracer.py` is a stub
- `app/pathing/pathing.py` is a stub
- `data/catalog.json` is currently `[]`
- `README.md` is nearly empty

This means the implementation is still at scaffold level.

## Execution plan

### Phase 1: lock contracts and local test harness

1. Implement `app/state.py` exactly to the PDF contract.
2. Define internal helper expectations for catalog records:
   - valid `course_id`
   - `skills_taught` as ESCO URI strings
   - `prerequisites` referencing valid course IDs
3. Implement `experiments/test_kahn.py` as an isolated 5-node DAG verification script.
4. Keep all experimental code separate from `app/`.

### Phase 2: implement the pathing core

1. `app/pathing/dag_builder.py`
   - load catalog via `app.catalog.loader`
   - create a NetworkX DAG
   - validate missing prerequisites and cycles
   - attach course metadata to nodes
2. `app/pathing/gap_analyzer.py`
   - convert extracted skills into mastered vs unmastered sets
   - compute `skill_gap`
   - identify directly assigned courses
   - recursively collect prerequisite pull-ins
   - determine skipped courses when user mastery already covers them
3. `app/pathing/kahn.py`
   - compute indegrees for the active subgraph
   - run heap-based Kahn ordering
   - use the priority function from the blueprint
4. `app/pathing/tracer.py`
   - generate trace strings for `assigned` and `prerequisite` nodes using the blueprint templates
5. `app/pathing/pathing.py`
   - compose the full pipeline
   - use `mock_extract()` until extractor integration is ready
   - output a valid `PipelineState`

### Phase 3: implement API composition

1. `app/main.py`
   - create FastAPI app
   - add CORS middleware
   - define request model
   - call `run_pipeline(...)`
   - return `PipelineState`
2. Keep extraction integration swappable:
   - default to mock extraction during backend development
   - swap to real `extract_skills()` only after extractor acceptance passes

### Phase 4: verification

1. Unit-level verification
   - toy DAG ordering from `experiments/test_kahn.py`
   - cycle and missing prerequisite detection
   - skill gap correctness
   - prerequisite pull-in correctness
   - metrics calculation correctness
2. Acceptance verification
   - `run_pipeline(mock_current_state, mock_target_state)` returns valid `PipelineState`
   - `POST /api/analyze` returns JSON matching the frontend contract
3. Integration verification
   - frontend can render `final_pathway`, `metrics`, and `reasoning_trace`
   - extractor swap does not change response shape

## Person 3 scope from the PDFs

This file now focuses on everything owned by Person 3, `Algorithm & API`.

### Files you own

- `app/pathing/dag_builder.py`
- `app/pathing/gap_analyzer.py`
- `app/pathing/kahn.py`
- `app/pathing/tracer.py`
- `app/pathing/pathing.py`
- `app/main.py`
- `app/state.py`
- `experiments/test_kahn.py`

### Files you depend on but do not own

- `data/catalog.json`
- `app/catalog/loader.py`
- `app/catalog/esco_lookup.py`
- `app/extractor/extractor.py`
- `app/extractor/skillner_model.py`
- `app/extractor/jobbert_model.py`
- `app/extractor/skillsim_model.py`
- `app/extractor/groq_mastery.py`

### What is explicitly in your scope

- build the adaptive pathing engine
- build the priority-aware Kahn ordering
- compose the backend pipeline
- expose the FastAPI endpoint
- keep the response shape stable for frontend integration
- use mock extraction until Person 2 is confirmed done

### What is explicitly not in your scope

- building the extractor internals
- populating the real catalog content
- changing the shared schema arbitrarily
- importing sandbox scripts from `experiments/` into `app/`

## Required runtime behavior for your task

### Input flow

For the backend, the required logical flow is:

- receive `resume_text` and `jd_text`
- obtain `current.extracted_skills`
  - use `mock_extract()` during development
  - later swap to real `extract_skills()`
- obtain `target.required_skills`
  - this may initially be mocked or derived from controlled inputs until extractor/JD parsing is finalized
- compute `skill_gap`
- find affected courses from catalog
- recursively add prerequisite courses when needed
- topologically order the resulting subgraph
- generate reasoning traces
- calculate metrics
- return `PipelineState`

### Output behavior

Your response must always be JSON-compatible and shaped exactly like `PipelineState`.

The frontend expects:

- `current`
- `target`
- `skill_gap`
- `final_pathway`
- `reasoning_trace`
- `metrics`

No extra fields should be required by the frontend for first integration.

## Detailed implementation notes for each owned file

### `app/state.py`

Must contain the shared TypedDict contract used by backend and mirrored by frontend.

Implementation notes:

- use `TypedDict`
- use `Literal` for `node_state`
- keep names exactly aligned with the blueprint
- do not include `candidate_courses` in `PipelineState`
- `candidate_courses` is only a local variable inside gap analysis per the timeline PDF

### `app/pathing/dag_builder.py`

Responsibilities:

- load the course catalog
- validate graph structure assumptions
- build the NetworkX DAG
- attach course metadata needed later by pathing

Validation requirements:

- every `course_id` must be unique
- every prerequisite reference must resolve to an existing course
- graph must be acyclic
- every course should expose the metadata required by the blueprint

Suggested outputs:

- the full catalog graph
- fast access to course metadata by `course_id`

### `app/pathing/gap_analyzer.py`

Responsibilities:

- identify which extracted skills count as mastered
- compute the JD skill gap using set subtraction
- identify direct candidate courses whose `skills_taught` intersects the skill gap
- recursively include upstream prerequisites
- determine the node state for each included course

Rules from the PDFs:

- mastery threshold is `>= 0.85`
- courses covering already-mastered skills should be bypassed where appropriate
- prerequisite pull-ins happen only when a prerequisite course is needed for an assigned course and is not already effectively bypassed

Expected local concepts:

- `mastered_skill_uris`
- `unmastered_skill_uris`
- `skill_gap`
- `candidate_courses`
- `active_subgraph_course_ids`
- per-course node state

### `app/pathing/kahn.py`

Responsibilities:

- perform topological sorting on the active subgraph
- respect prerequisite constraints
- use a priority queue so the highest-value course is selected first among currently available nodes

Required priority function:

- `priority = (skill_gap_count * (1 - mastery_score)) / (estimated_hours + 1e-5)`

Implementation notes:

- use `heapq`
- use negative values to simulate a max-heap
- indegree logic must be scoped to the active subgraph, not the entire catalog
- if a cycle is detected in the active subgraph, fail loudly because the catalog is invalid

### `app/pathing/tracer.py`

Responsibilities:

- generate human-readable reasoning traces for included non-skipped courses

Trace templates required by blueprint:

- assigned:
  - `"{Course Title} ({Course ID}) added: User has {mastery_score:.2f} mastery of {Skill Label}. This skill is directly required by the target JD."`
- prerequisite:
  - `"{Course Title} ({Course ID}) added: User has {mastery_score:.2f} mastery of {Skill Label}. Required as a prerequisite for {Dependent Course Title}, which addresses a gap in the target JD."`

Implementation notes:

- traces should be deterministic
- traces should map to actual course IDs and labels from the active decision path
- skipped nodes do not need explanatory strings in the PDFs' examples

### `app/pathing/pathing.py`

Responsibilities:

- expose `run_pipeline()`
- compose all pathing steps into one backend-facing function
- hold the development `mock_extract()` until extractor integration

Development rule from the PDFs:

- do not wait for Person 2
- use hardcoded mock extracted skills during development
- delete or replace `mock_extract()` only when extractor is explicitly confirmed done

Expected internal flow:

1. assemble current state
2. assemble target state
3. compute gap and active subgraph
4. run priority-aware Kahn sort
5. build `final_pathway`
6. build `reasoning_trace`
7. compute `metrics`
8. return `PipelineState`

### `app/main.py`

Responsibilities:

- expose the FastAPI app
- define request model
- configure CORS
- invoke the pipeline
- return `PipelineState`

Implementation notes:

- allow origin `http://localhost:3000`
- allow method `POST`
- request model fields must be `resume_text` and `jd_text`
- the endpoint path must be `/api/analyze`

### `experiments/test_kahn.py`

Responsibilities:

- build a toy 5-node DAG in isolation
- verify Kahn ordering
- verify the heap priority optimization does not violate prerequisites

This file is specifically part of your assigned experimental work in Phase 1.

## Testing plan for Person 3

The PDFs mention sandbox testing, acceptance conditions, and integration conditions. This section expands those into an explicit test plan tied to your scope.

### A. Sandbox tests you should have

#### `experiments/test_kahn.py`

Purpose:

- prove that a priority-aware Kahn implementation still obeys dependency order

Test cases:

- simple linear chain
- branching DAG with two zero-indegree nodes
- case where higher-priority node is chosen first when both are valid
- case where a high-priority node is blocked by prerequisites and must not be emitted early
- invalid cyclic graph should fail clearly

Expected outputs:

- printed node order
- assertion that every prerequisite appears earlier than its dependent
- assertion that available-node priority is respected only within zero-indegree candidates

### B. Pathing unit tests you should add or be prepared to add

Recommended test matrix for your owned backend logic:

#### `tests/test_state_contract.py`

Checks:

- `PipelineState` shape is buildable with blueprint fields
- `node_state` only accepts `skipped`, `assigned`, `prerequisite`

#### `tests/test_dag_builder.py`

Checks:

- valid catalog loads into a DAG
- duplicate `course_id` is rejected
- missing prerequisite reference is rejected
- cycle in catalog is rejected
- node metadata is attached correctly

Suggested fixtures:

- `catalog_valid.json`
- `catalog_missing_prereq.json`
- `catalog_cycle.json`
- `catalog_duplicate_ids.json`

#### `tests/test_gap_analyzer.py`

Checks:

- `skill_gap` equals `required_skills - mastered_skills`
- skills at `0.85` and above are treated as mastered
- direct candidate courses are identified correctly
- recursive prerequisite pull-ins are identified correctly
- mastered foundational course can be treated as skipped or bypassed as intended by implementation
- no unrelated course is pulled into active pathway

Suggested fixtures:

- extracted skills with mixed mastery
- target skills with partial overlap
- tiny catalog where one assigned course has one or two prerequisites

#### `tests/test_kahn.py`

Checks:

- topological order is valid
- heap priority breaks ties among currently available nodes
- indegree is computed on active subgraph correctly
- returned order includes all active nodes exactly once

#### `tests/test_tracer.py`

Checks:

- assigned trace matches required template
- prerequisite trace matches required template
- course title, course ID, dependent course title, mastery score, and skill label are injected correctly
- skipped nodes do not generate trace lines unless intentionally designed otherwise

#### `tests/test_pathing_pipeline.py`

Checks:

- `run_pipeline()` returns all required top-level fields
- `final_pathway` contains valid `node_state` values
- `metrics.baseline_courses` equals catalog size
- `metrics.assigned_courses` matches count of non-skipped courses
- `metrics.reduction_pct` is correct
- pipeline works with mock extraction

Suggested acceptance-style fixture:

- mock resume text
- mock JD text
- mock extracted skills matching the timeline PDF example
- small deterministic catalog with known expected pathway

#### `tests/test_api.py`

Checks:

- `POST /api/analyze` returns HTTP 200 on valid payload
- invalid payload shape returns validation error
- response JSON matches `PipelineState` shape
- CORS configuration is compatible with frontend origin

### C. Acceptance criteria from the PDFs

You are done with your track only when all of the following are true:

- `experiments/test_kahn.py` proves the priority heap does not break prerequisite ordering
- `run_pipeline(mock_current_state, mock_target_state)` returns valid `PipelineState` JSON
- `final_pathway` is populated with pathway nodes
- `metrics` is populated
- FastAPI endpoint is live locally
- the response shape is stable enough for frontend consumption
- extractor swap is deferred until Person 2 is fully done

### D. Manual test scenarios

These are useful because the PDFs care about product behavior, not just code correctness.

#### Scenario 1: user already has a core skill

- input contains strong mastery for one target skill
- result should skip or bypass redundant training where applicable
- reduction percentage should improve compared to baseline

#### Scenario 2: user lacks a target skill but not its prerequisites

- result should assign the direct gap-covering course
- prerequisite chain should not over-expand

#### Scenario 3: user lacks a target skill and lacks prerequisites

- result should include prerequisite pull-ins in valid order
- reasoning trace should explain both direct assignment and prerequisite addition

#### Scenario 4: two eligible courses available at once

- ordering should prefer the higher computed priority among zero-indegree nodes

#### Scenario 5: malformed catalog

- backend should fail clearly rather than silently produce an invalid pathway

### E. Test data you need for your task

You need small deterministic fixtures for pathing, even before the real catalog arrives.

Recommended local fixture design:

- 5-course toy DAG for `experiments/test_kahn.py`
- 6 to 8 course mini catalog for unit tests
- at least one branch where one course teaches a direct gap skill
- at least one branch where a course is prerequisite-only
- extracted skill fixture with:
  - one mastered skill
  - one low-mastery skill
  - one irrelevant skill
- target fixture with:
  - one already-mastered requirement
  - one missing requirement
  - one requirement that triggers prerequisite pull-in

### F. Verification commands to support your task

These are the checks your track should eventually be able to run:

- run the sandbox DAG experiment
- run unit tests for pathing logic
- run API tests
- run the FastAPI app locally
- post a sample payload to `/api/analyze`

The exact command set can be finalized once the test framework and dependency setup are added.

## Integration notes specific to your task

### Handoff from Person 1

You need:

- non-empty `data/catalog.json`
- valid prerequisite references
- ESCO URIs matching the exact verified format

Until that arrives:

- build logic against small local deterministic fixtures
- keep interfaces stable

### Handoff from Person 2

You need:

- `extract_skills(text)` returning valid `SkillEntry` objects

Until that arrives:

- keep using `mock_extract()`
- do not block your pathing/API work
- do not prematurely wire unstable extractor internals into your main flow

### Handoff to Person 4

You must provide:

- `POST /api/analyze`
- a response that matches `PipelineState`
- populated `metrics`
- populated `reasoning_trace`
- stable `node_state` values:
  - `skipped`
  - `assigned`
  - `prerequisite`

## Definition of done for your task

Your `Algorithm & API` track is complete when:

1. the shared schema in `app/state.py` matches the blueprint
2. the pathing modules are implemented
3. the priority-based Kahn sort works on a toy DAG
4. the pipeline can run end to end with mock extraction
5. `POST /api/analyze` returns valid `PipelineState`
6. reasoning traces are generated for included courses
7. metrics are computed correctly
8. the backend contract is stable for frontend integration
9. the real extractor swap is held until Person 2 is truly ready

## Immediate priorities for Algorithm & API

If working as Person 3, the immediate sequence should be:

1. implement `app/state.py`
2. implement `app/pathing/kahn.py` and `experiments/test_kahn.py`
3. implement `dag_builder.py`
4. implement `gap_analyzer.py`
5. implement `tracer.py`
6. compose `run_pipeline()` in `pathing.py` with mock extraction
7. implement `app/main.py`
8. validate with mocked end-to-end API output
9. wait to swap in the real extractor until Person 2 is actually done

## Risks and dependencies

- `data/catalog.json` is still empty, so real pathing depends on Person 1 finishing catalog data.
- ESCO URI format must exactly match the extractor and catalog output.
- Reasoning traces and node states must remain stable because the frontend depends on them.
- If the schema drifts between backend and frontend, integration will fail.
- The system will be judged partly on grounding and explainability, so invented courses or vague trace strings are unacceptable.

## Non-code deliverables implied by the PDFs

- `README.md` needs setup instructions, dependencies, and high-level algorithm explanation.
- the repo should be reproducible, and Docker support is encouraged
- the demo must show adaptive differences for different personas
- the 5-slide deck must cover:
  - solution overview
  - architecture/workflow
  - tech stack/models
  - algorithms/training
  - datasets/metrics

## Recommended implementation order in this repo

1. backend contracts and pathing
2. mock end-to-end API response
3. catalog integration
4. extractor integration
5. README and submission polish
