"""
verifier.py
-----------
RAG verifier and Corrective RAG logic.

verify_claim()        → query ChromaDB, ask LLM to judge support
apply_corrective_rag() → if CONTRADICTED, revise AgentThesis via LLM
"""

import logging

import llm_client
from models import AgentThesis, VerificationResult
from alpha_seeker.rag.vector_store import query_docs
import config

logger = logging.getLogger(__name__)

_STATUS_TOKENS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")


def verify_claim(claim: str, symbol: str) -> VerificationResult:
    """
    Embed the claim, retrieve top-5 recent docs filtered by symbol,
    and ask the LLM whether they support, contradict, or are insufficient.
    """
    docs = query_docs(claim, symbol, n_results=5, max_age_hours=config.RAG_WINDOW_HOURS)

    if not docs:
        return VerificationResult(
            status="INSUFFICIENT_EVIDENCE",
            explanation="No recent news found for this symbol.",
            n_docs_checked=0,
        )

    news_block = "\n".join(
        f"[{i+1}] ({d['source']}) {d['text'][:200]}" for i, d in enumerate(docs)
    )

    prompt = f"""Claim: '{claim}'

Recent news about {symbol}:
{news_block}

Does the news SUPPORT, CONTRADICT, or provide INSUFFICIENT_EVIDENCE for this claim?

Answer with exactly one of: SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
Then a single sentence explanation.

Format:
STATUS: <one of the three>
EXPLANATION: <one sentence>"""

    response = llm_client.call_llm(prompt, max_tokens=120)
    status, explanation = _parse_verification(response)

    return VerificationResult(
        status=status,
        explanation=explanation,
        source_urls=[d["url"] for d in docs if d.get("url")],
        n_docs_checked=len(docs),
    )


def _parse_verification(text: str) -> tuple[str, str]:
    status      = "INSUFFICIENT_EVIDENCE"
    explanation = text.strip()

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("STATUS:"):
            candidate = line.split(":", 1)[1].strip().upper()
            if candidate in _STATUS_TOKENS:
                status = candidate
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    # Fallback: scan for keywords if structured parse failed
    if status == "INSUFFICIENT_EVIDENCE":
        upper = text.upper()
        if "CONTRADICTED" in upper:
            status = "CONTRADICTED"
        elif "SUPPORTED" in upper:
            status = "SUPPORTED"

    return status, explanation


def apply_corrective_rag(
    thesis:   AgentThesis,
    results:  list[VerificationResult],
) -> AgentThesis:
    """
    If any verification result is CONTRADICTED, revise the thesis via LLM
    and reduce confidence by 0.15 (floor 0.10).

    Returns the (possibly unchanged) thesis.
    """
    contradictions = [r for r in results if r.status == "CONTRADICTED"]
    if not contradictions:
        return thesis

    evidence_block = "\n".join(
        f"- {r.explanation}" for r in contradictions
    )

    prompt = f"""You are the {thesis.agent} agent. Your original thesis was:

"{thesis.reasoning}"

However, recent news contradicts part of your analysis:
{evidence_block}

Revise your thesis in 2-3 sentences acknowledging this contradiction.
Keep your direction ({thesis.direction}) only if you still have strong reasons; otherwise update it.
Maintain a concise, evidence-based tone."""

    revised = llm_client.call_llm(prompt, max_tokens=200)
    if not revised:
        revised = thesis.reasoning

    new_confidence = max(0.10, thesis.confidence - 0.15)

    return AgentThesis(
        agent=thesis.agent,
        direction=thesis.direction,
        confidence=new_confidence,
        reasoning=revised,
        key_signals=thesis.key_signals,
        raw_data=thesis.raw_data,
        rag_status="CONTRADICTED",
        revision_flag=True,
    )


def verify_thesis(thesis: AgentThesis, symbol: str) -> tuple[AgentThesis, list[VerificationResult]]:
    """
    Verify the top-2 key signals of a thesis, apply Corrective RAG if needed.
    Returns the (possibly revised) thesis and the list of VerificationResults.
    """
    claims  = thesis.key_signals[:2]
    results = [verify_claim(c, symbol) for c in claims]

    # Mark RAG status on thesis (best/worst of the results)
    if any(r.status == "CONTRADICTED" for r in results):
        thesis = apply_corrective_rag(thesis, results)
    elif all(r.status == "SUPPORTED" for r in results):
        thesis.rag_status = "SUPPORTED"
    else:
        thesis.rag_status = "INSUFFICIENT_EVIDENCE"

    return thesis, results
