#!/bin/bash
# SkillBridge Engine - Model Pre-Download Script
# Run this once to cache all models locally.

echo "Installing dependencies..."
# pip install skillner transformers sentence-transformers spacy python-dotenv networkx groq fastapi uvicorn

# Verify SkillNer installed cleanly. If it fails, use the GitHub fallback:
# pip install git+https://github.com/AnasAito/SkillNER.git
# python -c "from skillner import SkillExtractor; print('SkillNer OK')"

# python -m spacy download en_core_web_lg
# python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('alvperez/skill-sim-model')"

echo "Setup complete."
