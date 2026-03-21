"""
benchmark_extraction.py
=======================
Measures Precision, Recall, and F1 for SkillBridge's multi-model extraction
pipeline vs. a single LLM baseline (Groq / Llama-3), against ground-truth
skills from the resume_data.csv dataset.

Usage:
    cd /mnt/extra-maal/skillbridge/skillbridge-backend
    python benchmark_extraction.py --csv dataset/archive/resume_data.csv \
                                   --n 20 \
                                   --output benchmark_results.json

Requirements:
    pip install pandas groq tqdm
    GROQ_API_KEY must be set in environment.

Output:
    - Console table comparing SkillBridge vs LLM baseline
    - JSON file with per-resume breakdown for Slide 5 presentation
"""

import ast
import json
import logging
import os
import re
import time
import argparse
from typing import List, Dict, Tuple, Set

import pandas as pd
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)  # suppress pipeline debug noise
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

# ── Groq client ───────────────────────────────────────────────────────────────
_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Dataset helpers ───────────────────────────────────────────────────────────

def parse_skills_column(raw: str) -> List[str]:
    """
    Parse the dataset's skills column which is a Python-list-formatted string.
    e.g. "['Python', 'Java', 'Machine Learning']" → ['python', 'java', 'machine learning']
    Returns lowercase normalized list for fuzzy matching.
    """
    if not raw or pd.isna(raw):
        return []
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(s).lower().strip() for s in parsed if s]
    except (ValueError, SyntaxError):
        # Fallback: split by comma if not a valid Python list
        cleaned = re.sub(r"[\[\]']", "", str(raw))
        return [s.lower().strip() for s in cleaned.split(",") if s.strip()]
    return []


def build_resume_text(row: pd.Series) -> str:
    """
    Reconstruct resume text from CSV row fields.
    Combines career objective, skills, responsibilities, and positions
    into a coherent text block that mimics a real resume.
    """
    parts = []

    if pd.notna(row.get("career_objective", "")):
        parts.append(str(row["career_objective"]))

    # Skills section
    skills_raw = str(row.get("skills", ""))
    if skills_raw and skills_raw != "nan":
        try:
            skills_list = ast.literal_eval(skills_raw)
            if isinstance(skills_list, list):
                parts.append("Technical Skills: " + ", ".join(str(s) for s in skills_list))
        except (ValueError, SyntaxError):
            parts.append("Technical Skills: " + skills_raw)

    # Positions
    positions_raw = str(row.get("positions", ""))
    if positions_raw and positions_raw != "nan":
        try:
            positions = ast.literal_eval(positions_raw)
            if isinstance(positions, list):
                parts.append("Positions: " + ", ".join(str(p) for p in positions))
        except (ValueError, SyntaxError):
            parts.append("Positions: " + positions_raw)

    # Responsibilities (key context for mastery signals)
    resp_raw = str(row.get("responsibilities", ""))
    if resp_raw and resp_raw != "nan":
        parts.append("Responsibilities:\n" + resp_raw)

    # Related skills in job
    related_raw = str(row.get("related_skils_in_job", ""))
    if related_raw and related_raw != "nan":
        try:
            related = ast.literal_eval(related_raw)
            if isinstance(related, list):
                flat = []
                for item in related:
                    if isinstance(item, list):
                        flat.extend(str(x) for x in item)
                    else:
                        flat.append(str(item))
                parts.append("Related Job Skills: " + ", ".join(flat))
        except (ValueError, SyntaxError):
            pass

    return "\n\n".join(p for p in parts if p.strip())


def normalize_skill(skill: str) -> str:
    """Normalize a skill label for fuzzy matching."""
    s = skill.lower().strip()
    # Remove punctuation and common suffixes
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def skills_match(extracted: str, ground_truth: str, threshold: int = 3) -> bool:
    """
    Check if an extracted skill matches a ground-truth skill.
    Uses token overlap to handle variations like:
      'machine learning' ↔ 'ml'  (too different → no match)
      'python programming' ↔ 'python' (overlap → match)
      'node js' ↔ 'node.js' → match after normalization
    """
    e = normalize_skill(extracted)
    g = normalize_skill(ground_truth)

    # Exact match after normalization
    if e == g:
        return True

    # Substring match (one contains the other, min 3 chars)
    if len(e) >= threshold and len(g) >= threshold:
        if e in g or g in e:
            return True

    # Token overlap: all tokens of shorter string in longer string
    e_tokens = set(e.split())
    g_tokens = set(g.split())
    if e_tokens and g_tokens:
        shorter = e_tokens if len(e_tokens) <= len(g_tokens) else g_tokens
        longer  = g_tokens if len(e_tokens) <= len(g_tokens) else e_tokens
        if shorter.issubset(longer):
            return True

    return False


def compute_metrics(
    extracted_labels: List[str],
    ground_truth_labels: List[str],
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1 using fuzzy matching.

    Precision = TP / (TP + FP) = fraction of extracted skills that are correct
    Recall    = TP / (TP + FN) = fraction of ground-truth skills that were found
    F1        = harmonic mean of precision and recall
    """
    if not extracted_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth_labels)}

    if not ground_truth_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(extracted_labels), "fn": 0}

    # For each extracted skill, check if it matches ANY ground-truth skill
    tp_extracted = 0
    matched_gt_indices: Set[int] = set()

    for ext in extracted_labels:
        for i, gt in enumerate(ground_truth_labels):
            if i not in matched_gt_indices and skills_match(ext, gt):
                tp_extracted += 1
                matched_gt_indices.add(i)
                break

    tp = tp_extracted
    fp = len(extracted_labels) - tp
    fn = len(ground_truth_labels) - len(matched_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


# ── SkillBridge Pipeline ──────────────────────────────────────────────────────

def run_skillbridge_pipeline(resume_text: str) -> List[str]:
    """
    Run SkillBridge's multi-model extraction pipeline.
    Returns list of extracted skill labels.
    
    Imports from app.extractor — must be run from the skillbridge-backend directory.
    """
    try:
        from app.extractor.extractor import extract_skills_from_jd
        # Use extract_skills (resume mode) for mastery — but we only need labels for benchmark
        # Using extract_skills_from_jd avoids the Groq mastery call which saves API quota
        # For a fairer test, use extract_skills (full pipeline) if you have quota to spare
        from app.extractor.extractor import extract_skills
        skills = extract_skills(resume_text, catalog=[])
        return [s["label"] for s in skills if s.get("label")]
    except Exception as exc:
        logger.error("[SkillBridge] Pipeline failed: %s", exc)
        return []


# ── LLM Baseline ─────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """
You are a skill extraction engine. Extract all technical and professional skills 
from the resume text provided.

Rules:
1. Only extract explicit skills — do not infer or hallucinate
2. Include both hard skills (tools, languages, frameworks) and soft skills
3. Return each skill as a clean, short phrase (1-3 words)
4. Return ONLY a JSON array of strings. No explanation. No markdown.

Example output: ["Python", "Machine Learning", "SQL", "Team Leadership", "AWS"]
""".strip()

def run_llm_baseline(resume_text: str, max_retries: int = 3) -> List[str]:
    """
    Single LLM call baseline using Llama-3.3-70b via Groq.
    Returns list of extracted skill labels.
    """
    client = get_groq_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extract all skills from this resume:\n\n{resume_text[:3000]}"
                    },
                ],
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            # Handle both {"skills": [...]} and direct array responses
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if s]
            elif isinstance(parsed, dict):
                # Try common wrapper keys
                for key in ["skills", "extracted_skills", "skill_list", "result"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return [str(s).strip() for s in parsed[key] if s]
                # Fallback: flatten all list values
                for v in parsed.values():
                    if isinstance(v, list):
                        return [str(s).strip() for s in v if s]

        except json.JSONDecodeError as e:
            logger.warning("[LLM Baseline] JSON parse failed (attempt %d): %s", attempt + 1, e)
        except Exception as e:
            logger.warning("[LLM Baseline] API error (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff

    logger.error("[LLM Baseline] All retries failed.")
    return []


# ── Main Benchmark Loop ───────────────────────────────────────────────────────

def run_benchmark(csv_path: str, n: int = 20, output_path: str = "benchmark_results.json"):
    """
    Run the full benchmark comparing SkillBridge vs LLM baseline.
    
    Args:
        csv_path:    Path to resume_data.csv
        n:           Number of resumes to evaluate
        output_path: Path to write JSON results
    """
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Filter rows that have both skills ground truth and enough resume content
    df = df.dropna(subset=["skills"])
    df = df[df["skills"].str.len() > 10]
    df = df[df["career_objective"].notna() | df["responsibilities"].notna()]

    # Sample n rows — use a fixed seed for reproducibility
    if len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
    else:
        logger.warning("Dataset has only %d valid rows (requested %d)", len(df), n)
        n = len(df)

    logger.info("Running benchmark on %d resumes", n)

    skillbridge_results = []
    llm_results = []
    per_resume_data = []

    for idx, row in tqdm(df.iterrows(), total=n, desc="Benchmarking"):
        resume_text = build_resume_text(row)
        ground_truth = parse_skills_column(row["skills"])

        if not resume_text.strip() or not ground_truth:
            logger.debug("Skipping row %d — empty resume or ground truth", idx)
            continue

        # ── SkillBridge pipeline ───────────────────────────────────────────
        sb_labels = run_skillbridge_pipeline(resume_text)
        sb_metrics = compute_metrics(sb_labels, ground_truth)

        # ── LLM baseline ──────────────────────────────────────────────────
        llm_labels = run_llm_baseline(resume_text)
        llm_metrics = compute_metrics(llm_labels, ground_truth)

        skillbridge_results.append(sb_metrics)
        llm_results.append(llm_metrics)

        per_resume_data.append({
            "resume_index":      int(idx),
            "ground_truth_count": len(ground_truth),
            "ground_truth":      ground_truth[:10],  # first 10 for readability
            "skillbridge": {
                "extracted_count": len(sb_labels),
                "extracted_sample": sb_labels[:10],
                **sb_metrics,
            },
            "llm_baseline": {
                "extracted_count": len(llm_labels),
                "extracted_sample": llm_labels[:10],
                **llm_metrics,
            },
        })

        # Rate limit safety — Groq free tier
        time.sleep(0.5)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def avg(results: List[dict], key: str) -> float:
        vals = [r[key] for r in results if r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    sb_agg = {
        "precision": avg(skillbridge_results, "precision"),
        "recall":    avg(skillbridge_results, "recall"),
        "f1":        avg(skillbridge_results, "f1"),
        "total_tp":  sum(r["tp"] for r in skillbridge_results),
        "total_fp":  sum(r["fp"] for r in skillbridge_results),
        "total_fn":  sum(r["fn"] for r in skillbridge_results),
    }

    llm_agg = {
        "precision": avg(llm_results, "precision"),
        "recall":    avg(llm_results, "recall"),
        "f1":        avg(llm_results, "f1"),
        "total_tp":  sum(r["tp"] for r in llm_results),
        "total_fp":  sum(r["fp"] for r in llm_results),
        "total_fn":  sum(r["fn"] for r in llm_results),
    }

    # ── Print results table ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SKILLBRIDGE EXTRACTION BENCHMARK RESULTS")
    print(f"  Evaluated on {len(per_resume_data)} resumes")
    print("="*60)
    print(f"\n{'Metric':<20} {'SkillBridge':>15} {'LLM Baseline':>15}")
    print("-"*52)
    print(f"{'Precision':<20} {sb_agg['precision']:>14.1%} {llm_agg['precision']:>14.1%}")
    print(f"{'Recall':<20} {sb_agg['recall']:>14.1%} {llm_agg['recall']:>14.1%}")
    print(f"{'F1 Score':<20} {sb_agg['f1']:>14.1%} {llm_agg['f1']:>14.1%}")
    print("-"*52)
    print(f"{'True Positives':<20} {sb_agg['total_tp']:>15} {llm_agg['total_tp']:>15}")
    print(f"{'False Positives':<20} {sb_agg['total_fp']:>15} {llm_agg['total_fp']:>15}")
    print(f"{'False Negatives':<20} {sb_agg['total_fn']:>15} {llm_agg['total_fn']:>15}")
    print("="*60)

    # Highlight the win
    if sb_agg["precision"] > llm_agg["precision"]:
        delta = sb_agg["precision"] - llm_agg["precision"]
        print(f"\n  ✓ SkillBridge precision is {delta:.1%} higher than LLM baseline")
    if sb_agg["recall"] > llm_agg["recall"]:
        delta = sb_agg["recall"] - llm_agg["recall"]
        print(f"  ✓ SkillBridge recall is {delta:.1%} higher than LLM baseline")
    if sb_agg["f1"] > llm_agg["f1"]:
        delta = sb_agg["f1"] - llm_agg["f1"]
        print(f"  ✓ SkillBridge F1 is {delta:.1%} higher than LLM baseline")
    print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "n_resumes": len(per_resume_data),
            "dataset":   csv_path,
            "models": {
                "skillbridge": "SkillNer + JobBERT + Groq Llama-3.3-70b (profile classification)",
                "llm_baseline": "Groq Llama-3.3-70b (single zero-shot call)",
            },
            "matching_strategy": "fuzzy token overlap (normalize → substring/token match)",
        },
        "aggregate": {
            "skillbridge": sb_agg,
            "llm_baseline": llm_agg,
        },
        "per_resume": per_resume_data,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    print(f"  Full results saved to: {output_path}")
    print()

    return output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark SkillBridge extraction vs LLM baseline"
    )
    parser.add_argument(
        "--csv",
        default="dataset/archive/resume_data.csv",
        help="Path to resume_data.csv",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of resumes to evaluate (default: 20)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--skillbridge-only",
        action="store_true",
        help="Skip LLM baseline (faster, saves Groq quota)",
    )

    args = parser.parse_args()

    run_benchmark(
        csv_path=args.csv,
        n=args.n,
        output_path=args.output,
    )
