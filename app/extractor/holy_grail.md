# 🧠 NLP Extraction Engine (Track 2)

Welcome to the extraction core of the **AI-Adaptive Onboarding Engine**. This module is responsible for parsing highly unstructured, cross-domain human capital data (resumes) and deterministically mapping those capabilities to the European (ESCO) taxonomic standard.

To secure strict **"Zero Hallucination"** compliance and rapid execution latency, we explicitly avoid using a generalized LLM as a zero-shot entity extractor. Instead, we architected a highly specialized, cascading 4-stage pipeline.

## 🏗️ The Multi-Model Architecture

Our pipeline cleanly separates mechanical entity extraction from probabilistic cognitive reasoning:

1. **Explicit Lexical Extraction (`skillner_model.py`)**
   - **Backbone:** `en_core_web_lg` (spaCy) + EMSI Database.
   - **Function:** Rule-based extraction of explicit technological and operational keywords (e.g., "PostgreSQL", "Forklift Operation"). High precision, zero variance.

2. **Implicit Contextual Extraction (`jobbert_model.py`)**
   - **Backbone:** `jjzha/jobbert_knowledge_extraction` (BERT).
   - **Function:** Sequence labeling via token classification (BIO tagging). It stitches together implicit skills and transversal competencies hidden in responsibility clauses (e.g., "mentoring junior development staff").

3. **Semantic Taxonomic Mapping (`skillsim_model.py`)**
   - **Backbone:** `alvperez/skill-sim-model` (768-D Sentence Transformer).
   - **Function:** Solves the vocabulary mismatch problem. Maps arbitrarily phrased resume strings into a dense metric hyperspace, calculating Cosine Similarity against the official ESCO catalog. Only matches exceeding the `≥ 0.65` mathematical threshold are retained.

4. **Experience-Weighted Mastery Estimation (`groq_mastery.py`)**
   - **Backbone:** `Llama-3-70B` (via Groq API).
   - **Function:** Operates strictly as a constrained cognitive reasoner. It reads the exact contextual strings surrounding the mapped ESCO URIs and outputs a `mastery_score` (0.0 to 1.0) grounded entirely in **Benjamin Bloom’s Mastery Learning framework**.

## 🚀 Cross-Domain Scalability
This engine has been rigorously benchmarked against both deep-tech personas (e.g., Senior Full-Stack Engineers) and operational/labor roles (e.g., Warehouse Supervisors and Retail Managers). It successfully parses both hard technical frameworks and transversal soft skills without degrading in accuracy.

## ⚙️ Setup & Developer Guide

**1. Heavy Weight Caching**
To guarantee our 10-second latency budget, all NLP transformer weights are loaded globally into RAM upon server initialization. You must download the spaCy backbone locally before execution:
```bash
python -m spacy download en_core_web_lg
```

**2. Environment Configuration**
This module requires a Groq API key for the mastery estimation phase. Create a `.env` file at the root of the project:
```env
GROQ_API_KEY=gsk_your_api_key_here
```
*(Note: Never commit the `.env` file to version control. It is explicitly ignored in `.gitignore`.)*

## 🔌 API Contract (For Downstream DAG Integration)

The `extractor.py` orchestrator strictly enforces the LangGraph shared state schema. Downstream consumers (Track 3) will receive a pristine `List[SkillEntry]` TypedDict array, completely free of serialization errors or NumPy artifacts:

```json
[
  {
    "esco_uri": "[http://data.europa.eu/esco/skill/ks-python](http://data.europa.eu/esco/skill/ks-python)",
    "label": "Python (programming language)",
    "mastery_score": 0.8
  },
  {
    "esco_uri": "[http://data.europa.eu/esco/skill/ks-scm](http://data.europa.eu/esco/skill/ks-scm)",
    "label": "supply chain management",
    "mastery_score": 0.3
  }
]
```
```
