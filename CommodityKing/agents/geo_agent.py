"""
Geopolitical & Macro Agent.

Data flow:
  1. Fetch price data (yfinance)
  2. Fetch NewsAPI articles — 24h window and 7d window
  3. Fetch RSS articles filtered by commodity keywords
  4. Run kalman_filter on hourly prices
  5. Run bayesian_surprise (24h NewsAPI vs 30d ChromaDB cache)
  6. Build structured LLM prompt
  7. Parse LLM response into AgentThesis
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from agents.base import (
    build_news_query, fetch_newsapi_articles, fetch_price_data,
    fetch_rss_articles, get_date_str, parse_bullet_list,
    parse_confidence, parse_direction, parse_section, strip_markdown, _first_meaningful_line,
)
from llm_client import LLMClient
from models import AgentThesis
from models_quant.quant_models import bayesian_surprise, kalman_filter
from rag.vector_store import get_articles_for_commodity

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a commodity geopolitical analyst. You produce concise, specific, "
    "evidence-based analysis. You name actual events, countries, and people. "
    "You never hedge with vague language when data is available. "
    "You reduce confidence when news is sparse or expected."
)

_ALL_HEADERS = [
    "DIRECTION", "CONFIDENCE", "HEADLINE",
    "24H ANALYSIS", "7D NARRATIVE", "HISTORICAL ANALOG",
    "KEY RISKS",
]


async def run_geo_agent(
    commodity_config: dict,
    llm_client:       LLMClient,
) -> AgentThesis:
    """
    Run the geopolitical & macro agent for one commodity.
    Never raises — returns a neutral thesis on unrecoverable failure.
    """
    symbol = commodity_config.get("symbol", "")
    name   = commodity_config["name"]
    ticker = commodity_config["ticker"]
    unit   = commodity_config["unit"]
    drivers = commodity_config["key_drivers"]

    logger.info("[GeoAgent] Starting for %s", name)

    # ── 1. Price data ─────────────────────────────────────────────────────────
    price_data = fetch_price_data(ticker)
    if price_data["error"]:
        logger.warning("[GeoAgent] Price fetch failed for %s: %s", name, price_data["error"])

    # ── 2. Kalman filter ──────────────────────────────────────────────────────
    kalman_result: dict = {}
    prices_1h = price_data["prices_1h"]
    if len(prices_1h) >= 2:
        kalman_result = kalman_filter(prices_1h)
    else:
        kalman_result = {"trend": "unknown", "signal_sigma": 0.0,
                         "is_signal": False, "latest_signal": 0.0}

    # ── 3. News fetching ──────────────────────────────────────────────────────
    query = build_news_query(name, drivers)
    keywords_for_rss = [name.lower()] + [d.split()[0].lower() for d in drivers[:3]]

    articles_24h = fetch_newsapi_articles(query, from_date=get_date_str(1),
                                          sort_by="publishedAt", page_size=15)
    articles_7d  = fetch_newsapi_articles(query, from_date=get_date_str(7),
                                          sort_by="relevancy",   page_size=25)
    articles_rss = fetch_rss_articles(keywords=keywords_for_rss, max_age_days=7)

    # Merge RSS into 7d bucket (deduplicated by title)
    rss_titles = {a["title"] for a in articles_7d}
    for rss_art in articles_rss:
        if rss_art["title"] not in rss_titles:
            articles_7d.append(rss_art)

    sources_used = []
    if articles_24h or articles_7d:
        sources_used.append("NewsAPI")
    if articles_rss:
        sources_used.append("RSS")

    # ── 4. Bayesian surprise ──────────────────────────────────────────────────
    # 24h corpus: NewsAPI 24h + any RSS articles from last 24h (fallback when
    # NewsAPI is rate-limited). 30d baseline: ChromaDB cache — avoids a second
    # rate-limited NewsAPI call. On first run the cache will be sparse; the
    # rag_node ingests articles after geo_node completes, so by the second run
    # the corpus is meaningful.
    from rag.ingester import get_article_texts
    cutoff_24h = get_date_str(1)  # "YYYY-MM-DD" string, same format as publishedAt
    rss_24h = [
        a for a in articles_rss
        if (a.get("publishedAt") or a.get("published") or "")[:10] >= cutoff_24h
    ]
    texts_24h    = get_article_texts(articles_24h + rss_24h)
    texts_30d    = get_articles_for_commodity(symbol, days_back=30)
    history_source = "chroma_cache" if texts_30d else "insufficient"

    surprise_result = bayesian_surprise(texts_24h, texts_30d, name)
    surprise_score  = surprise_result["surprise_score"]
    surprise_interp = surprise_result["interpretation"]
    surprise_terms  = surprise_result["top_surprise_terms"]

    logger.info(
        "[GeoAgent] Bayesian baseline: history_source=%s, history_docs=%d, 24h_docs=%d",
        history_source, len(texts_30d), len(texts_24h),
    )

    # ── 5. Format news for LLM ────────────────────────────────────────────────
    def _fmt_articles(arts: list[dict], max_items: int = 10) -> str:
        if not arts:
            return "  No articles retrieved."
        lines = []
        for a in arts[:max_items]:
            title = a.get("title") or ""
            desc  = (a.get("description") or "")[:120]
            src   = a.get("source") or ""
            pub   = (a.get("publishedAt") or "")[:10]
            lines.append(f"  [{pub}] [{src}] {title}. {desc}")
        return "\n".join(lines)

    news_24h_text = _fmt_articles(articles_24h)
    news_7d_text  = _fmt_articles(articles_7d, max_items=15)

    # ── 6. LLM prompt ─────────────────────────────────────────────────────────
    prompt = f"""You are a commodity geopolitical analyst with expertise in {name} markets.
Analyse the following data and produce a structured research note.

PRICE ACTION:
  Current price:  {price_data['current_price']} {unit}
  24h change:     {price_data['change_24h_pct']:+.2f}%
  7d change:      {price_data['change_7d_pct']:+.2f}%
  Kalman trend:   {kalman_result.get('trend', 'unknown')} \
(signal: {kalman_result.get('signal_sigma', 0.0):.2f}σ)
  Is signal:      {kalman_result.get('is_signal', False)}

NEWS — LAST 24 HOURS ({len(articles_24h)} articles):
{news_24h_text}

NEWS — LAST 7 DAYS ({len(articles_7d)} articles):
{news_7d_text}

NEWS SURPRISE SCORE: {surprise_score:.3f} ({surprise_interp})
Top unusual terms vs 30-day baseline: {', '.join(surprise_terms) or 'none'}

KEY GEOPOLITICAL DRIVERS FOR {name.upper()}:
{chr(10).join(f'  - {d}' for d in drivers)}

Produce your analysis with EXACTLY these labelled sections:

DIRECTION: [BULLISH | BEARISH | NEUTRAL]
CONFIDENCE: [0.0-1.0] — reduce if surprise score is low (expected news already priced in)

HEADLINE:
[One sentence — the single most important geopolitical development right now]

24H ANALYSIS:
[2-3 sentences on breaking events from the last 24 hours]

7D NARRATIVE:
[2-3 sentences on the developing trend over the last 7 days]

HISTORICAL ANALOG:
[If applicable: what past event does this resemble? What happened then? \
Write "None identified" if no clear analog.]

KEY RISKS:
- [Risk 1 to this view]
- [Risk 2 to this view]

Be specific. Name actual events, countries, organisations. \
If news is sparse, say so and reduce confidence accordingly."""

    # ── 7. LLM call ───────────────────────────────────────────────────────────
    try:
        response_text = await llm_client.agent(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            purpose="geo_agent",
            max_tokens=900,
        )
    except Exception as exc:
        logger.error("[GeoAgent] LLM call failed for %s: %s", name, exc)
        return _neutral_thesis(name, str(exc))

    # ── 8. Parse response ─────────────────────────────────────────────────────
    direction  = parse_direction(response_text)
    confidence = parse_confidence(response_text)
    headline   = _first_meaningful_line(parse_section(response_text, "HEADLINE", _ALL_HEADERS))
    analysis_24h = parse_section(response_text, "24H ANALYSIS", _ALL_HEADERS)
    analysis_7d  = parse_section(response_text, "7D NARRATIVE", _ALL_HEADERS)
    hist_analog  = parse_section(response_text, "HISTORICAL ANALOG", _ALL_HEADERS)
    risks_raw    = parse_section(response_text, "KEY RISKS", ["END", ""])
    key_risks    = parse_bullet_list(risks_raw) or ["Insufficient data to assess risks"]

    logger.info(
        "[GeoAgent] %s → %s (conf=%.2f, surprise=%.3f, articles=%d)",
        name, direction, confidence, surprise_score, len(articles_24h) + len(articles_7d)
    )

    return AgentThesis(
        agent="geopolitical",
        direction=direction,
        confidence=confidence,
        headline=headline or f"Geopolitical analysis of {name}",
        analysis_24h=analysis_24h,
        analysis_7d=analysis_7d,
        historical_analog=hist_analog,
        key_risks=key_risks,
        surprise_score=surprise_score,
        surprise_interp=surprise_interp,
        surprise_terms=surprise_terms,
        surprise_kl=float(surprise_result.get("kl_divergence", 0.0)),
        surprise_history_source=history_source,
        kalman_result=kalman_result,
        n_articles=len(articles_24h) + len(articles_7d),
    )


def _neutral_thesis(name: str, error: str) -> AgentThesis:
    return AgentThesis(
        agent="geopolitical",
        direction="NEUTRAL",
        confidence=0.0,
        headline=f"Geopolitical data unavailable for {name}",
        error=error,
    )
