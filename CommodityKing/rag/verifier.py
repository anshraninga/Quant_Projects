"""
RAG verifier — checks agent claims against the vector store and
finds historical analogs for the synthesis node.

Two public functions:
  verify_claims(claims, commodity)   -> list[VerificationResult]
  find_historical_analog(commodity, context_text) -> dict | None
"""

from __future__ import annotations

import logging
from typing import Optional

from models import VerificationResult, HistoricalAnalog
from rag.vector_store import query_news, query_historical
import config

logger = logging.getLogger(__name__)

# Cosine distance thresholds (ChromaDB returns distance, not similarity)
# cosine distance = 1 - cosine_similarity
# distance < 0.25  → similarity > 0.75  → strong support
# distance < 0.40  → similarity > 0.60  → moderate support
# distance > 0.55  → similarity < 0.45  → weak / insufficient

_SUPPORTED_THRESHOLD    = 0.30   # distance ≤ this → SUPPORTED
_CONTRADICTED_THRESHOLD = 0.20   # distance ≤ this AND content contradicts
_INSUFFICIENT_THRESHOLD = 0.55   # distance ≥ this → INSUFFICIENT_EVIDENCE


def verify_claims(
    claims:    list[str],
    commodity: str,
) -> list[VerificationResult]:
    """
    For each claim string, search both news and historical collections.
    Return a VerificationResult for each claim.

    At most 2 claims should be passed (spec: top 2 per agent).
    """
    results = []
    for claim in claims[:2]:
        result = _verify_single_claim(claim, commodity)
        results.append(result)
    return results


def _verify_single_claim(claim: str, commodity: str) -> VerificationResult:
    # Search news first (recent evidence is more relevant)
    news_hits = query_news(claim, commodity, n_results=3)

    # Search historical events as secondary evidence
    hist_hits = query_historical(claim, commodity, n_results=2)

    all_hits = news_hits + hist_hits

    if not all_hits:
        return VerificationResult(
            claim=claim,
            verdict="INSUFFICIENT_EVIDENCE",
            supporting_text="No matching documents found in knowledge base.",
            source="",
            similarity_score=0.0,
        )

    best = min(all_hits, key=lambda h: h["distance"])
    similarity = round(1.0 - best["distance"], 3)

    if best["distance"] <= _SUPPORTED_THRESHOLD:
        verdict        = "SUPPORTED"
        supporting_text = best["text"][:300]
        source          = best.get("source") or best.get("event_short", "")[:80]
    elif best["distance"] >= _INSUFFICIENT_THRESHOLD:
        verdict        = "INSUFFICIENT_EVIDENCE"
        supporting_text = "Closest match is too distant to draw a conclusion."
        source          = ""
    else:
        # Middle range: evidence exists but isn't conclusive
        verdict        = "INSUFFICIENT_EVIDENCE"
        supporting_text = best["text"][:200]
        source          = best.get("source") or ""

    return VerificationResult(
        claim=claim,
        verdict=verdict,
        supporting_text=supporting_text,
        source=source,
        similarity_score=similarity,
    )


def find_historical_analog(
    commodity:    str,
    context_text: str,
) -> Optional[HistoricalAnalog]:
    """
    Search historical_events collection for the closest analog to the
    current market situation described in context_text.

    Returns HistoricalAnalog if similarity > RAG_SIMILARITY_THRESHOLD,
    otherwise returns None.
    """
    hits = query_historical(context_text, commodity, n_results=2)

    if not hits:
        return HistoricalAnalog(found=False)

    best      = hits[0]
    similarity = round(1.0 - best["distance"], 3)

    if similarity < config.RAG_SIMILARITY_THRESHOLD:
        logger.debug(
            "[RAG] Best analog similarity %.3f below threshold %.2f for %s",
            similarity, config.RAG_SIMILARITY_THRESHOLD, commodity
        )
        return HistoricalAnalog(found=False)

    # Parse the structured event text back into fields
    # The text was formatted as: "Commodity: X. Date: Y. Event: Z. ..."
    meta        = best
    event_short = meta.get("event_short", "")
    date        = meta.get("date", "")

    # Extract full fields from the document text
    doc_text      = best.get("text", "")
    price_impact  = _extract_field(doc_text, "Price impact")
    context_str   = _extract_field(doc_text, "Context")
    resolution    = _extract_field(doc_text, "Resolution")

    logger.info(
        "[RAG] Historical analog found for %s: '%s' (similarity=%.3f)",
        commodity, event_short[:60], similarity
    )

    return HistoricalAnalog(
        found=True,
        event_id=meta.get("id", ""),
        event=event_short,
        date=date,
        similarity=similarity,
        price_impact_then=price_impact,
        context=context_str,
        resolution=resolution,
    )


def _extract_field(text: str, field_name: str) -> str:
    """
    Extract 'Field name: value.' from formatted event text.
    Returns empty string if field not found.
    """
    marker = f"{field_name}: "
    start  = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end    = text.find(". ", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()


def summarise_rag_status(results: list[VerificationResult]) -> str:
    """
    Collapse a list of VerificationResults into a single RAGStatus string
    for the AgentThesis.rag_status field.

    Logic:
      Any SUPPORTED   → SUPPORTED
      Any CONTRADICTED → CONTRADICTED
      All INSUFFICIENT → INSUFFICIENT_EVIDENCE
    """
    verdicts = {r.verdict for r in results}
    if not verdicts:
        return "NOT_CHECKED"
    if "CONTRADICTED" in verdicts:
        return "CONTRADICTED"
    if "SUPPORTED" in verdicts:
        return "SUPPORTED"
    return "INSUFFICIENT_EVIDENCE"
