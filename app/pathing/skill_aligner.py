import os
import json
import logging
from typing import List, Dict, Any
from groq import Groq

logger = logging.getLogger(__name__)

def _get_val(skill: Any, key: str, default: Any = None) -> Any:
    """Safely extracts a value whether the skill is a dict or a Pydantic model."""
    if isinstance(skill, dict):
        return skill.get(key, default)
    return getattr(skill, key, default)

def align_skills(resume_skills: List[Any], jd_skills: List[Any]) -> Dict[str, str]:
    """
    Uses Groq (LLM) to map synonymous skills from the resume to the JD.
    
    Returns a mapping of taxonomy IDs: 
    { "JD_taxonomy_id": "Resume_taxonomy_id" }
    """
    if not resume_skills or not jd_skills:
        return {}

    # Extract names and IDs as objects to feed the LLM prompt
    jd_list = []
    for s in jd_skills:
        name = _get_val(s, 'label') or _get_val(s, 'name')
        tid = _get_val(s, 'taxonomy_id')
        if name and tid:
            jd_list.append({"id": tid, "name": name})
            
    res_list = []
    for s in resume_skills:
        name = _get_val(s, 'label') or _get_val(s, 'name')
        tid = _get_val(s, 'taxonomy_id')
        if name and tid:
            res_list.append({"id": tid, "name": name})

    if not jd_list or not res_list:
        return {}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found. Skipping intelligent alignment.")
        return {}

    prompt = f"""You are an expert HR ontology and technical skill mapping system.
Your task is to map a candidate's extracted Resume Skills to the required Job Description (JD) Skills based on their relevance and synonymy.

JD Skills (Required by employer): 
{json.dumps(jd_list)}

Resume Skills (Candidate's experience): 
{json.dumps(res_list)}

CRITICAL INSTRUCTION - NOISE FILTERING:
Ignore and DO NOT map generic, non-technical "noise" skills, even if they appear in both lists. This includes:
- Generic verbs/adjectives (e.g., "write", "nice", "scalable", "manage", "collaborate").
- Broad business terms or titles (e.g., "software", "professional", "manager", "operations").
- Soft skills (e.g., "communication", "leadership", "problem solving", "teamwork").
Only map concrete hard skills, tools, specific methodologies, or domain-specific expertise.

Find pairs where a Resume Skill satisfies, or is a strong synonym/subset of, a JD Skill.
Examples:
- If JD requires "Relational Database" and Resume has "SQL", "MySQL", or "DBMS", they match.
- If JD requires "CI/CD" and Resume has "Jenkins" or "GitHub Actions", they match.
- If JD requires "Version Control" and Resume has "Git", they match.

Return ONLY a valid JSON object mapping the EXACT JD Skill ID (`id`) to the EXACT Resume Skill ID (`id`).
Do not include skills that don't strongly match or any skills identified as noise.
Format:
{{
  "JD_taxonomy_id": "Resume_taxonomy_id"
}}
"""
    try:
        # llama3-8b-8192 is fast and highly capable of strict JSON formatting
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Mapping is natively returned as ID -> ID now
        id_mapping = json.loads(response.choices[0].message.content)
                
        if id_mapping:
            logger.info(f"[SkillAligner] Successfully mapped {len(id_mapping)} skills via LLM using taxonomy IDs.")
            
        return id_mapping
        
    except Exception as e:
        logger.error(f"[SkillAligner] Error in LLM skill alignment fallback to literal intersection: {e}")
        return {}