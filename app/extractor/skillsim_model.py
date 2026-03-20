# app/extractor/skillsim_model.py
"""
Semantic skill-to-ESCO mapping using alvperez/skill-sim-model.

Maps raw extracted skill strings to canonical ESCO taxonomy entries
via cosine similarity over sentence embeddings.

CRITICAL FIX (Phase 1 finding):
    util.cos_sim() returns a torch.Tensor of shape [1, 1].
    Calling .item() is mandatory to extract a plain Python float.
    Returning the raw tensor object causes fatal JSON serialization
    errors (TypeError: Object of type Tensor is not JSON serializable).

Performance note:
    The ESCO catalog is pre-encoded once at initialization time.
    For a catalog of ~13,890 ESCO skills, this takes ~8–12s on CPU
    and ~1–2s on GPU. Never re-encode the catalog per request.
"""

import logging
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

MODEL_NAME       = "alvperez/skill-sim-model"
MATCH_THRESHOLD  = 0.65   # Validated in Phase 1 — below this, matches are noise

# Module-level cache for the model and pre-encoded ESCO embeddings
_model           : Optional[SentenceTransformer] = None
_esco_embeddings : Optional[torch.Tensor]        = None
_esco_ids        : Optional[List[str]]           = None
_esco_labels     : Optional[List[str]]           = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("[SkillSim] Loading model %s...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("[SkillSim] Model ready.")
    return _model


def preload_esco_catalog(esco_catalog: Dict[str, dict]) -> None:
    """
    Pre-encode the full ESCO catalog into embeddings.

    Call this ONCE at application startup (e.g., in a FastAPI lifespan
    event), not on every request. Stores results in module-level cache.

    Args:
        esco_catalog: Dict mapping ESCO URI → {"label": str, ...}
                      e.g. {"http://data.europa.eu/esco/skill/abc": {"label": "Python"}}
    """
    global _esco_embeddings, _esco_ids, _esco_labels
    model = _get_model()

    _esco_ids    = list(esco_catalog.keys())
    _esco_labels = [v["label"] for v in esco_catalog.values()]

    logger.info("[SkillSim] Pre-encoding %d ESCO skills...", len(_esco_labels))
    _esco_embeddings = model.encode(
        _esco_labels,
        convert_to_tensor = True,
        show_progress_bar = True,
        batch_size        = 128,
    )
    logger.info("[SkillSim] ESCO catalog pre-encoding complete.")


def map_to_esco(
    extracted_phrases: List[str],
    esco_catalog     : Dict[str, dict],
) -> List[dict]:
    """
    Map a list of raw extracted skill strings to their best-matching
    ESCO canonical skill entries using cosine similarity.

    Args:
        extracted_phrases: Output of skillner or jobbert extractors.
                           e.g. ["python", "machine learning", "docker"]
        esco_catalog     : Dict mapping ESCO URI → {"label": str, ...}
                           Used to (re-)encode catalog if not already cached.

    Returns:
        List of match dicts for phrases that scored >= MATCH_THRESHOLD.
        Each dict has shape:
            {
                "input_phrase"  : str,   # original extracted string
                "esco_uri"      : str,   # matched ESCO URI
                "label"         : str,   # matched ESCO canonical label
                "similarity"    : float  # plain Python float, NOT a Tensor
            }

        Phrases with no match above threshold are silently omitted.
        Returns [] on empty input or failure.

    Raises:
        Does NOT raise — all exceptions are caught and logged.
    """
    if not extracted_phrases:
        logger.warning("[SkillSim] Empty input phrases, returning [].")
        return []

    if not esco_catalog:
        logger.warning("[SkillSim] Empty ESCO catalog, returning [].")
        return []

    try:
        # Ensure catalog is encoded (idempotent if already cached)
        _ensure_catalog_encoded(esco_catalog)
        model = _get_model()

        # Batch-encode all candidate phrases in one pass
        candidate_embeddings = model.encode(
            extracted_phrases,
            convert_to_tensor = True,
            batch_size        = 64,
        )

        matches: List[dict] = []

        for i, phrase in enumerate(extracted_phrases):
            best_uri, best_label, best_score = _find_best_esco_match(
                candidate_embeddings[i]
            )

            if best_score >= MATCH_THRESHOLD:
                matches.append({
                    "input_phrase": phrase,
                    "esco_uri"    : best_uri,
                    "label"       : best_label,
                    # CRITICAL: .item() converts Tensor → plain Python float
                    # Without this, json.dumps() raises TypeError downstream
                    "similarity"  : float(best_score),
                })
                logger.debug(
                    "[SkillSim] '%s' → '%s' (%.4f)", phrase, best_label, best_score
                )
            else:
                logger.debug(
                    "[SkillSim] '%s' → no match above threshold (best=%.4f)",
                    phrase, best_score
                )

        logger.info(
            "[SkillSim] %d/%d phrases mapped above threshold %.2f.",
            len(matches), len(extracted_phrases), MATCH_THRESHOLD
        )
        return matches

    except Exception as exc:
        logger.error("[SkillSim] Mapping failed: %s", exc, exc_info=True)
        return []


def _ensure_catalog_encoded(esco_catalog: Dict[str, dict]) -> None:
    """
    Encode catalog only if not already cached, or if the catalog
    has changed size (simple cache invalidation heuristic).
    """
    global _esco_ids
    if _esco_embeddings is None or len(_esco_ids or []) != len(esco_catalog):
        preload_esco_catalog(esco_catalog)


def _find_best_esco_match(
    candidate_embedding: torch.Tensor,
) -> tuple[str, str, float]:
    """
    Find the highest cosine similarity ESCO entry for a single embedding.

    Returns:
        Tuple of (esco_uri, label, float_score).
        float_score is a plain Python float extracted via .item().
    """
    # cos_sim returns Tensor shape [1, num_esco_skills]
    similarities = util.cos_sim(candidate_embedding, _esco_embeddings)[0]
    best_idx     = int(similarities.argmax())

    # CRITICAL: .item() → plain Python float, NOT a Tensor
    best_score = similarities[best_idx].item()

    return _esco_ids[best_idx], _esco_labels[best_idx], best_score