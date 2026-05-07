"""
sentiment_agent.py
------------------
Calls NewsAnalysis and SentimentAnalysis, then synthesises via LLM.
"""

import logging

from existing_modules.NewsAnalysis      import run_news_analysis
from existing_modules.SentimentAnalysis import run_sentiment_analysis
from models import AgentThesis
from alpha_seeker.agents.base import call_agent_llm, parse_llm_thesis, _neutral_thesis

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a crypto sentiment analyst. "
    "Ground your view in the actual scores provided; do not speculate beyond the data."
)


def run_sentiment_agent(symbol: str) -> AgentThesis:
    raw: dict = {}
    try:
        raw["news"]      = run_news_analysis(symbol)
        raw["sentiment"] = run_sentiment_analysis(symbol)
    except Exception as exc:
        logger.error("Sentiment tools failed for %s: %s", symbol, exc)
        return _neutral_thesis("sentiment", raw, str(exc))

    n_articles = _count_samples(raw.get("news", {}))
    n_posts    = _count_samples(raw.get("sentiment", {}))
    summary    = _format_raw(raw, n_articles, n_posts)

    prompt = f"""You are a sentiment analyst. Here are the NLP sentiment signals for {symbol}:

{summary}

Write a 2-3 sentence thesis. Include:
DIRECTION: BULLISH | BEARISH | NEUTRAL
CONFIDENCE: (0.0-1.0)
REASONING: (2-3 sentences citing scores and sample sizes)
KEY_SIGNALS: signal 1 | signal 2

Rule: if fewer than 5 total articles/posts were found, cap CONFIDENCE at 0.50."""

    response = call_agent_llm(_SYSTEM, prompt)
    thesis   = parse_llm_thesis("sentiment", response, raw)

    # Enforce low-sample cap
    total_samples = n_articles + n_posts
    if total_samples < 5:
        thesis.confidence = min(thesis.confidence, 0.50)

    return thesis


def _count_samples(data: dict) -> int:
    if not isinstance(data, dict):
        return 0
    # NewsAnalysis stores count inside news_sentiment
    ns = data.get("news_sentiment", {})
    if isinstance(ns, dict) and "total_articles" in ns:
        return int(ns["total_articles"])
    # SentimentAnalysis: reddit sub-dict
    reddit = data.get("reddit", {})
    if isinstance(reddit, dict) and "posts_analyzed" in reddit:
        return int(reddit["posts_analyzed"])
    for key in ("n_articles", "n_posts", "n_samples", "count", "total"):
        if key in data:
            try:
                return int(data[key])
            except (ValueError, TypeError):
                pass
    return 0


def _format_raw(raw: dict, n_articles: int, n_posts: int) -> str:
    lines = [f"News articles found : {n_articles}",
             f"Reddit posts found  : {n_posts}"]

    news = raw.get("news", {})
    if isinstance(news, dict):
        ns = news.get("news_sentiment", {})
        if ns:
            lines.append(
                f"[NEWS]  direction={ns.get('overall','?').upper()}  "
                f"bull={ns.get('bullish_pct','?')}%  bear={ns.get('bearish_pct','?')}%"
            )
            for hl in news.get("top_headlines", [])[:3]:
                if isinstance(hl, dict):
                    lines.append(f"  headline({hl.get('sentiment','?')}): {str(hl.get('title',''))[:80]}")

    sent = raw.get("sentiment", {})
    if isinstance(sent, dict):
        lines.append(
            f"[SOCIAL]  direction={sent.get('combined_overall','?').upper()}  "
            f"bull={sent.get('combined_bullish_pct','?')}%  bear={sent.get('combined_bearish_pct','?')}%"
        )
        reddit = sent.get("reddit", {})
        if isinstance(reddit, dict) and reddit.get("posts_analyzed", 0):
            lines.append(
                f"  Reddit({reddit.get('posts_analyzed',0)} posts): "
                f"bull={reddit.get('bullish_pct','?')}%  bear={reddit.get('bearish_pct','?')}%"
            )

    return "\n".join(lines)
