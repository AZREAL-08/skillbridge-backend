# app/extractor/jobbert_model.py
"""
Implicit skill extraction using jjzha/jobbert_knowledge_extraction.

Extracts skill phrases from job-responsibility and experience descriptions
via BIO-tagged token classification. Handles two known model quirks
discovered in Phase 1 sandbox testing:

CRITICAL FIX 1 — Label map patch:
    This model's config.id2label is {0: 'B', 1: 'I', 2: 'O'} — bare labels
    without the 'B-SKILL'/'I-SKILL' prefix convention. The HuggingFace
    TokenClassificationPipeline internally filters on this prefix and
    silently discards ALL predictions, returning []. We patch id2label at
    load time to restore correct pipeline behavior.

CRITICAL FIX 2 — B/I span stitching:
    aggregation_strategy='simple' does not correctly merge consecutive B and I
    spans under the bare label scheme, splitting "machine learning" into two
    separate entries. We implement a manual stitching pass using char offsets
    from the original text to recover the exact surface form.
"""

import logging
import re
from typing import List, Optional

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
    Pipeline,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "jjzha/jobbert_knowledge_extraction"

# Module-level singleton
_pipe: Optional[Pipeline] = None


def _get_pipeline() -> Pipeline:
    """
    Lazy-initialize and cache the JobBERT pipeline singleton.
    Applies the id2label patch before handing the model to the pipeline.
    """
    global _pipe
    if _pipe is None:
        logger.info("[JobBERT] Loading model %s...", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

        # CRITICAL FIX 1: Patch bare B/I/O → B-SKILL/I-SKILL/O
        # Without this, HuggingFace pipeline discards all predictions silently.
        model.config.id2label = {0: "B-SKILL", 1: "I-SKILL", 2: "O"}
        model.config.label2id = {"B-SKILL": 0, "I-SKILL": 1, "O": 2}
        logger.info("[JobBERT] id2label patched to %s", model.config.id2label)

        _pipe = pipeline(
            task                 = "token-classification",
            model                = model,
            tokenizer            = tokenizer,
            aggregation_strategy = "simple",
        )
        logger.info("[JobBERT] Pipeline ready.")
    return _pipe


def extract_implicit_skills(text: str) -> List[str]:
    """
    Extract implicitly described skill phrases from experience or
    responsibility-style text using JobBERT token classification.

    Best input style: job-posting or responsibility language.
        GOOD: "Required skills include Python and machine learning."
        WEAK: "I know Python."  (resume-style; lower model confidence)

    Args:
        text: Raw text string — resume bullet points or job descriptions.

    Returns:
        Deduplicated list of stitched skill phrase strings (lowercased).
        e.g. ["python", "machine learning", "rest api development"]

        Returns empty list on empty input or extraction failure.

    Raises:
        Does NOT raise — all exceptions are caught and logged.
    """
    if not text or not text.strip():
        logger.warning("[JobBERT] Received empty text, returning [].")
        return []

    try:
        pipe      = _get_pipeline()
        raw_spans = pipe(text)

        if not raw_spans:
            logger.info("[JobBERT] No spans returned for input text.")
            return []

        # CRITICAL FIX 2: Manually stitch consecutive B-SKILL/I-SKILL spans
        stitched = _stitch_spans(raw_spans, text)
        deduped  = _deduplicate(stitched)

        logger.info("[JobBERT] Extracted %d unique skill phrases.", len(deduped))
        return deduped

    except Exception as exc:
        logger.error("[JobBERT] Extraction failed: %s", exc, exc_info=True)
        return []


def _stitch_spans(spans: list, original_text: str) -> List[str]:
    """
    Merge consecutive B-SKILL and I-SKILL spans into single skill phrases.

    Strategy: Use character offsets (span['start'], span['end']) to slice
    the exact substring from original_text, bypassing WordPiece artifacts
    like '##ing' that may appear in span['word'].

    Args:
        spans        : Raw output list from the HuggingFace pipeline.
        original_text: The original input string passed to the pipeline.

    Returns:
        List of raw skill phrase strings (not yet deduplicated or lowercased).
    """
    phrases: List[str] = []
    buffer_start: Optional[int] = None
    buffer_end  : Optional[int] = None

    for span in spans:
        label = span.get("entity_group") or span.get("entity", "O")

        if label == "B-SKILL":
            # Flush any existing buffer before starting a new phrase
            if buffer_start is not None:
                phrase = _slice_and_clean(original_text, buffer_start, buffer_end)
                if phrase:
                    phrases.append(phrase)

            buffer_start = span["start"]
            buffer_end   = span["end"]

        elif label == "I-SKILL" and buffer_start is not None:
            # Extend the current phrase's end offset
            buffer_end = span["end"]

        else:
            # Non-skill token — flush buffer
            if buffer_start is not None:
                phrase = _slice_and_clean(original_text, buffer_start, buffer_end)
                if phrase:
                    phrases.append(phrase)
                buffer_start = None
                buffer_end   = None

    # Flush any remaining buffer after loop
    if buffer_start is not None:
        phrase = _slice_and_clean(original_text, buffer_start, buffer_end)
        if phrase:
            phrases.append(phrase)

    return phrases


def _slice_and_clean(text: str, start: int, end: int) -> str:
    """
    Slice original_text[start:end] and clean whitespace/punctuation artifacts.
    Using char offsets avoids all WordPiece subword reconstruction issues.
    """
    phrase = text[start:end].strip()
    # Remove leading/trailing punctuation that leaked into the span
    phrase = re.sub(r'^[^\w]+|[^\w]+$', '', phrase)
    return phrase.lower()


def _deduplicate(phrases: List[str]) -> List[str]:
    """Preserve insertion order while removing exact duplicate strings."""
    seen  : set       = set()
    result: List[str] = []
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result